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

REPO = "https://github.com/JerzySukiennik/gedit.git"
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
BATCH, ACCUM, STEPS, WARMUP = 32, 1, 60000, 200
LR_DECAY_STEPS = 60000
TEXT_DROPOUT = 0.1

if os.path.exists(f"{WORK}/gedit"):
    subprocess.run(["git", "-C", f"{WORK}/gedit", "pull", "--ff-only"], check=True)
else:
    subprocess.run(["git", "clone", "--depth", "1", REPO, f"{WORK}/gedit"], check=True)
os.chdir(f"{WORK}/gedit")
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "transformers"], check=True)

# Kaggle's input mount depth isn't fixed (seen both /kaggle/input/<slug>/ and
# /kaggle/input/datasets/<owner>/<slug>/ in practice) — recursive search finds
# it regardless.
hits = glob.glob("/kaggle/input/**/gedit_images.bin", recursive=True)
if not hits:
    print("gedit_images.bin not found. /kaggle/input contains:")
    for root, dirs, files in os.walk("/kaggle/input"):
        depth = root.count("/") - 2
        if depth > 3:
            continue
        print("  " * depth + os.path.basename(root) + "/")
        for f in sorted(files)[:12]:
            print("  " * (depth + 1) + f)
    raise SystemExit("attach the gedit-data dataset, or wait for it to finish building")

data_prefix = hits[0].replace("_images.bin", "")
print(f"data: {data_prefix}")

os.makedirs(OUT, exist_ok=True)
hits_ckpt = sorted(glob.glob("/kaggle/input/**/ckpt.pt", recursive=True))
resume = []
if hits_ckpt:
    shutil.copy(hits_ckpt[0], f"{OUT}/ckpt.pt")
    resume = ["--resume"]
    print(f"resuming from {hits_ckpt[0]}")
else:
    print("starting from scratch")

cmd = [sys.executable, "train/train.py",
       "--data", data_prefix,
       "--out", OUT,
       "--batch-size", str(BATCH),
       "--grad-accum", str(ACCUM),
       "--max-steps", str(STEPS),
       "--lr-decay-steps", str(LR_DECAY_STEPS),
       "--text-dropout", str(TEXT_DROPOUT),
       "--warmup", str(WARMUP),
       "--eval-every", "200",
       "--ckpt-every", "200",
       "--log-every", "20"] + resume
print(" ".join(cmd), flush=True)
subprocess.run(cmd, check=True)

print("\nsave this notebook's output as a Dataset ('gedit-ckpt') to continue "
      "in the next session, or download run/ckpt.pt if training finished.")
