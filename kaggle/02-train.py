"""
Kaggle cell 2 of 2 — training.

Settings: Accelerator GPU T4 x2, Internet ON, Persistence "Variables and
Files". Inputs: the 'gedit-data' Dataset from 01-prep.py, plus — on any run
after the first — the previous run's output as 'gedit-ckpt'.

Same resumable design as MicroG: everything needed to continue lands in
/kaggle/working/run every CKPT_EVERY steps; add that as input to the next
session and it picks up mid-stride.
"""

import glob
import os
import shutil
import subprocess
import sys

REPO = "https://github.com/JerzySukiennik/g-images.git"
WORK = "/kaggle/working"
OUT = f"{WORK}/run"

# Cross-attention + zero-init, measured 2026-07-22: ~0.75s/step on T4x2.
# Step 24000 (~12.9 epochs) checkpoint DOES respond to text/image
# conditioning (verified with a fixed-seed real-vs-zero-vs-random-text
# test) and shows real structure at 100 DDIM sampling steps (20 steps had
# been hiding it) — but still doesn't reliably follow specific instructions
# ("make it black and white" didn't desaturate). Not evidence of a broken
# architecture, more likely still just needs more training for a harder
# task than FiLM's global tone-shift.
#
# STEPS raised to 70000 (~37.5 epochs on the 60k-pair set, Jurek's choice
# after weighing an 80000-step option against repetition risk — no data
# augmentation exists here, so this is already a lot of repeats without it).
# CHECK QUALITY AT INTERVALS (~every 16000 steps) rather than running this
# to completion blind — there's no guarantee more steps keeps helping past
# some point, and catching a plateau/regression early avoids wasting
# further GPU quota chasing it. Still a ceiling, not a target — ckpt.pt is
# written every CKPT_EVERY steps regardless, safe to grab and stop early.
# 2026-07-27, after step 48400 still would not follow "make it black and
# white". A guidance sweep on that very checkpoint (runtime/cfg_test.py) found
# the biggest single problem was never the training at all: every judgement so
# far was made at guidance scale 1.0, i.e. the raw model prediction. With
# guidance the prompt's effect is large and correctly directed (b&w went grey,
# "add a hat" kept its colour) — but above ~3 the image saturates into colour
# garbage, because nothing ever trained the unconditional branch that guidance
# interpolates from.
#
# So train/train.py now defaults to conditioning dropout (5% text, 5% image),
# EMA weights, Min-SNR-gamma loss weighting, cosine LR decay and h-flip
# augmentation — no architecture change, so the step-48400 checkpoint resumes
# straight into it (verified locally). Nothing below needs new flags; those are
# the defaults. What this run has to prove is that guidance 5-7.5 becomes
# usable instead of saturating.
#
# LR_DECAY_STEPS is separate from STEPS on purpose: STEPS is a ceiling that
# gets raised whenever quality is still climbing, and tying the LR schedule to
# it would jump the LR back up on every raise.
#
# --- 2026-07-28, result of the 70000-step run -------------------------------
# That run finished clean (48400 -> 70000, 5.6h, val loss flat at 0.0164) and
# guidance at scale 1-3 is visibly better: "make it black and white" finally
# desaturates and holds facial structure. But scales 5-7.5 STILL saturate into
# flat colour, which was the whole point of adding dropout.
#
# Ruled out as the cause, by local sweeps on that checkpoint (no GPU spent):
# guidance rescale (Lin et al., phi=0.7) and Imagen dynamic thresholding
# (0.995) both failed to fix it, and thresholding made it worse. So this is
# not the x0 clamp flattening an overshoot — the guided direction itself is
# bad at high scale.
#
# The arithmetic points at the unconditional branch simply being starved.
# Dropout only existed for the last 21600 of 70000 steps, so at 5% it saw
# ~34.5k null-text samples against the conditional branch's 2.24M — about 1.5%
# as much. InstructPix2Pix and Stable Diffusion run dropout from step 0 for the
# entire schedule; here it arrived at the very end.
#
# Hence TEXT_DROPOUT 0.2 for a catch-up phase (Jurek's call). At 30000 more
# steps that is ~192k null-text samples, roughly 6x the current total. Image
# dropout deliberately stays at the 0.05 default: image_guidance is not used
# at sampling time yet, so raising it would only spend capacity on a branch
# nothing reads.
#
# STEPS 100000 lands exactly on LR_DECAY_STEPS, so the cosine decay finishes at
# its floor rather than being cut off mid-slope. ~30000 steps at ~0.75s/step is
# ~6.3h, inside the 12h session cap.
# --- v3, 2026-07-28: from scratch, discrete edit types --------------------------
# The 20% dropout run (to step 94200) settled it: cos(e_cond, e_null) = 1.000 at
# every timestep, difference vector ~1-2% of signal. The model had learned to
# ignore its text conditioning, so guidance had nothing to amplify and no amount
# of dropout or steps was going to change that.
#
# kaggle/04-probe-types.py then explained why, by counting the real corpus:
# "black and white" has 588 examples in 60000, sepia 182, "brighter" 3 — while 23%
# of prompts fall in a `replace_with` catch-all ("make her a panda") that names no
# single transformation. We had been grading the model all day on an edit worth 1%
# of its data, and feeding it a third of a corpus of mutually contradictory
# targets.
#
# v3 therefore: conditioning is a discrete edit type with its own learned tokens
# (data/edit_types.py, model/unet.py's TypeTokens); 13 of the types are exact
# image functions generated on the fly (data/synthetic_edits.py), giving unlimited
# perfectly-consistent pairs for precisely the colour/tone edits the corpus lacked;
# 10 semantic types keep their scraped pairs; types are sampled uniformly so the
# rare ones are learnable; cosine schedule + v-prediction replace linear + epsilon,
# which had the model predicting noise of std 0.19 where truth was 1.0 at the
# timestep sampling starts from; and the U-Net gains a fourth level, moving the
# bottleneck from 32x32 to 16x16.
#
# None of that is weight-compatible, so this run starts from zero and gedit-ckpt is
# detached from the kernel's inputs. STEPS is a ceiling as always; check quality at
# intervals. The synthetic types should be learnable within a few thousand steps —
# they are exact functions — so an early check is genuinely informative here,
# unlike the semantic ones.
# SESSION_STEPS caps how far ONE session runs, on top of the global TOTAL_STEPS
# ceiling. Two runs now have died the same way: a normal pace of ~0.84s/step for
# ~20000 steps, then a single 8409-second gap between two consecutive 20-step log
# lines, then a kill. That gap is the process thrashing before it gets OOM-killed
# — 2.3 hours of GPU quota spent computing nothing, on a quota shared with
# G-Micro and G-Mini.
#
# The suspect is page-cache pressure from randomly reading a 9.8 GB memmap for
# hours. Rather than chase that, end each session voluntarily while it is still
# healthy: train/train.py exits cleanly at --max-steps, the checkpoint lands, and
# the next session resumes. Costs one extra checkpoint upload, saves hours of
# thrashing.
# --- v5, 2026-08-02: bigger model, from scratch ---------------------------------
# The 22.4M model reached step 59200 and answered the open question negatively:
# doubling the object types' exposure did NOT sharpen them, and it visibly
# DEGRADED black_and_white, which had been clean at 41200 (colour speckles appeared
# where the ground truth is flat grey). Capacity contention across 53 sampled
# types, not undertraining.
#
# Size picked by measuring T4x2, not by parameter count (kaggle/08-size-probe.py):
#
#     22.4M anchor   0.64 s/step     (real sessions ran 0.78-0.84 with data loading)
#     64.8M          2.00 s/step
#     70.5M          1.69 s/step  <- more parameters than 64.8M, and FASTER
#     83.8M          1.92 s/step
#
# 70.5M wins because (1,2,3,4) keeps capacity away from the 128px level, where
# UNet compute actually goes. Adding ~30% for data loading puts it near 2.2 s/step,
# so 40000 steps is ~24h — one week's quota with room left for G-Micro, where 60000
# steps would have needed 36h and starved the other projects.
#
# 40000 rather than 60000 is a considered trade: the 22.4M model had clean filters
# by step 24000, so a 3x larger one reaching them inside 40000 is a fair
# expectation. If objects are still blobs at 40000, the answer is not "more steps"
# — it is latent-space diffusion, which is the next real option.
#
# Architecture changed, so nothing resumes: this trains from zero and the kernel
# must have gedit-ckpt DETACHED. The step-59200 checkpoint survives as a dataset
# version if the old model is ever wanted back.
# --- 2026-08-08: Jurek granted 15h of quota to push quality as far as it goes ----
#
# Two things had to change first, or the hours would have bought nothing.
#
# 1. LR_DECAY_STEPS was 40000 and training is AT 40000, so the cosine schedule sat
#    on its 1e-5 floor. Continuing would have run 28000 steps at a learning rate
#    that barely moves the weights. Extending the horizon to 68000 warm-restarts
#    it to ~8e-5 and decays to the floor exactly at the new target.
#
# 2. Sampling was uniform over 53 types, so the 13 synthetic filters took 25% of
#    every batch. Those match their exact ground truth pixel for pixel by now;
#    objects are the weak point. SYNTHETIC_SHARE drops them to 10%, moving that
#    capacity to the AnyEdit types. Not to zero: they are the only types with a
#    ground truth, so they double as the regression test that says the model
#    hasn't drifted.
# --- 2026-08-08: 98.3M od zera, z mieszana precyzja --------------------------
#
# Jurek: "ten skok z 20 do 70 byl ogromny, dociagnijmy do 100-150M". Dwie sondy
# na prawdziwym T4 ustawily te decyzje:
#
#   fp32:  kazda konfiguracja powyzej 70.5M konczyla sie OutOfMemory przy batchu 32
#   AMP:   70.5M 1.63 -> 0.96 s/krok, VRAM 12.7 -> 10.7 GB
#          98.3M  1.83 s/krok, 12.8 GB  <- miesci sie
#          119.4M 2.05 s/krok, 14.4 GB  <- miesci sie, ale 14.4 z 16 to zaden zapas
#          142.5M OOM nawet z AMP
#
# Petla treningowa NIGDY nie uzywala mieszanej precyzji — fp32 na kartach, ktorych
# rdzenie tensor istnieja wlasnie po to. To bylo przeoczenie kosztujace polowe
# predkosci przez caly projekt, i jednoczesnie powod, dla ktorego wieksze modele
# wygladaly na niemozliwe.
#
# 98.3M zamiast 119.4M: roznica pojemnosci niewielka, a 14.4/16 GB oznacza, ze
# dowolny skok zuzycia wywala sesje w polowie.
#
# 36000 krokow to tyle, ile miesci sie w 20h przy ~1.98 s/krok z narzutem danych.
# Uczciwie: model 70.5M potrzebowal ~40000 krokow, zeby dojsc tam gdzie jest, wiec
# ten skonczy mniej wiecej w tym samym miejscu, tylko z 40% wieksza pojemnoscia.
# Checkpoint 70.5M zostaje nietkniety jako punkt odniesienia i awaryjny powrot.
BATCH, ACCUM, WARMUP = 32, 1, 500
BASE_CHANNELS = 152
TOTAL_STEPS = 36000
# 12000, nie 15000: sesja 0->15000 dostala SIGKILL na kroku 14800 po 8h11m, przy
# zupelnie plaskim tempie (1922 s/1000 na starcie, 2007 s/1000 na koncu) i bez
# jednego ostrzezenia o pamieci. Rowne tempo az do naglej smierci to limit z
# zewnatrz, nie wyciek — 15000 krokow po ~1.99 s/krok po prostu nie miesci sie w
# oknie sesji. 12000 to ~6.6h, czyli z zapasem. Checkpoint co 200 krokow przezyl,
# wiec stracone bylo 200 krokow, ale kernel skonczyl jako ERROR i autochain
# slusznie odmowil lancuchowania dalej.
SESSION_STEPS = 12000
LR_DECAY_STEPS = 36000
TEXT_DROPOUT = 0.1
MIN_SNR_GAMMA = 0.0
SYNTHETIC_SHARE = 0.10

if os.path.exists(f"{WORK}/gedit"):
    subprocess.run(["git", "-C", f"{WORK}/gedit", "pull", "--ff-only"], check=True)
else:
    subprocess.run(["git", "clone", "--depth", "1", REPO, f"{WORK}/gedit"], check=True)
os.chdir(f"{WORK}/gedit")
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "transformers"], check=True)

# Kaggle's input mount depth isn't fixed (seen both /kaggle/input/<slug>/ and
# /kaggle/input/datasets/<owner>/<slug>/ in practice) — recursive search finds
# it regardless.
# Several prep notebooks are attached, each holding one shard of the corpus —
# one Kaggle notebook cannot write the 40 GB the full set needs. Every
# gedit_images.bin found becomes a dataset prefix, and train/train.py
# concatenates them logically.
hits = sorted(glob.glob("/kaggle/input/**/gedit_images.bin", recursive=True))
if not hits:
    print("no gedit_images.bin found. /kaggle/input contains:")
    for root, dirs, files in os.walk("/kaggle/input"):
        depth = root.count("/") - 2
        if depth > 3:
            continue
        print("  " * depth + os.path.basename(root) + "/")
        for f in sorted(files)[:12]:
            print("  " * (depth + 1) + f)
    raise SystemExit("attach the gedit-anyedit-* notebook outputs")

data_prefixes = [h.replace("_images.bin", "") for h in hits]
print(f"{len(data_prefixes)} data shards:")
for d in data_prefixes:
    print("   ", d)

os.makedirs(OUT, exist_ok=True)

# CKPT_URL exists because Kaggle has TWO different notions of "a kernel's output".
# kernel_sources mounts the last version that COMPLETED; the REST API serves the
# last version that RAN. A session that trains for 8h and then dies leaves its
# checkpoint reachable only by the second one — mounting silently hands you a
# much older checkpoint from a previous architecture instead, which is exactly
# what happened on 2026-08-09 (mounted step 55000 / 70.5M over step 14800 /
# 98.3M). Set CKPT_URL to the signed URL from
#   GET /api/v1/kernels/output?userName=..&kernelSlug=..  ->  files[].url
# to pull that checkpoint straight into the kernel over the internet, with no
# detour through a home connection. The URL is time-limited: generate it right
# before pushing, not hours ahead.
ckpt_url = os.environ.get("CKPT_URL", "").strip()
resume = []
if ckpt_url:
    print("downloading checkpoint from CKPT_URL")
    subprocess.run(["curl", "-fsSL", ckpt_url, "-o", f"{OUT}/ckpt.pt"], check=True)
    resume = ["--resume"]
    print(f"downloaded {os.path.getsize(f'{OUT}/ckpt.pt')/1e9:.2f} GB")
else:
    hits_ckpt = sorted(glob.glob("/kaggle/input/**/ckpt.pt", recursive=True))
    if hits_ckpt:
        shutil.copy(hits_ckpt[0], f"{OUT}/ckpt.pt")
        resume = ["--resume"]
        print(f"resuming from {hits_ckpt[0]}")
    else:
        print("starting from scratch")

# Read the resume point so this session's ceiling is start + SESSION_STEPS rather
# than the global total — see the SESSION_STEPS comment above.
start_step = 0
if resume:
    import torch
    start_step = torch.load(f"{OUT}/ckpt.pt", map_location="cpu").get("step", 0)

# Refuse to run backwards. A checkpoint from beyond TOTAL_STEPS means the wrong
# one got attached — on 2026-08-09 that printed the nonsense "session: step 55000
# -> 36000" and burned a GPU slot before train.py caught the mismatch a minute
# later. The step number alone is enough to know something is wrong, so say so
# here, loudly, instead of letting the run start.
if start_step >= TOTAL_STEPS:
    raise SystemExit(
        f"checkpoint is at step {start_step}, at or past the target {TOTAL_STEPS}. "
        "Almost certainly the wrong checkpoint — check what is mounted under "
        "/kaggle/input, or set CKPT_URL to the one you actually want."
    )

STEPS = min(TOTAL_STEPS, start_step + SESSION_STEPS)
print(f"session: step {start_step} -> {STEPS} (global target {TOTAL_STEPS})")

cmd = [sys.executable, "train/train.py",
       "--data", *data_prefixes,
       "--out", OUT,
       "--batch-size", str(BATCH),
       "--grad-accum", str(ACCUM),
       "--max-steps", str(STEPS),
       "--lr-decay-steps", str(LR_DECAY_STEPS),
       "--text-dropout", str(TEXT_DROPOUT),
       "--min-snr-gamma", str(MIN_SNR_GAMMA),
       "--synthetic-share", str(SYNTHETIC_SHARE),
       "--base-channels", str(BASE_CHANNELS),
       "--warmup", str(WARMUP),
       "--eval-every", "200",
       "--ckpt-every", "200",
       "--log-every", "20"] + resume
print(" ".join(cmd), flush=True)
subprocess.run(cmd, check=True)

print("\nsave this notebook's output as a Dataset ('gedit-ckpt') to continue "
      "in the next session, or download run/ckpt.pt if training finished.")
