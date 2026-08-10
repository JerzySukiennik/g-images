"""Kaggle CPU prep for AnyEdit — replaces kaggle/01-prep.py's IP2P download.

Settings: GPU off, Internet ON. Expect ~3h per shard.

SHARD comes from the environment so the bootstrap kernels differ by one line.
Each shard takes a disjoint slice of the per-type quota, so the three of them
together give ~350k pairs: enough for ~5 epochs at 60000 steps x batch 32, and
far short of the ~190 GB the full corpus would cost for one epoch.
"""

import os
import subprocess
import sys

REPO = "https://github.com/JerzySukiennik/g-images.git"
WORK = "/kaggle/working"

# Resolution and output prefix come from the environment so one script serves
# both corpora. G-Image 2.2 needs 256px sources: the plan is latent diffusion, so
# these pixels train the autoencoder and are then encoded once into latents that
# are ~12x smaller than the images they came from.
#
# The prefix MUST differ between corpora. train/train.py finds data by globbing
# **/<prefix>_images.bin across everything mounted, so a 256px shard sharing the
# 'gedit' prefix would be silently concatenated into a 128px training run — a
# resolution mismatch that no error message would explain.
RES = int(os.environ.get("GEDIT_RES", "128"))
PREFIX = os.environ.get("GEDIT_PREFIX", "gedit")

SHARD = int(os.environ.get("GEDIT_SHARD", "0"))

# Per-shard quotas, each with the number of already-consumed shards to skip.
#
# The skip is not optional bookkeeping: without it every notebook starts at the
# first shard of a type and downloads the SAME rows. The first attempt did
# exactly that — three notebooks produced 150000 `add` pairs of which only 60000
# were unique, caught by comparing their instruction lists (100% of the smaller
# set appeared in the larger). Each quota of 60000 consumes 10 shards, so the
# offsets below step by 10.
#
# `add` dominates the allocation because object addition is the capability this
# dataset switch exists to enable. Measured on the 90000 add pairs fetched so far,
# "hat" appears 580 times, which extrapolates to ~2500 across the full 393629 —
# comfortably trainable, where InstructPix2Pix offered 109.
QUOTAS = [
    (["add=60000", "tune_transfer=60000"], 0),      # done
    (["add=60000"], 10),
    (["add=30000", "style_change=25000",
      "background_change=25000", "replace=20000"], 0),  # done (non-add parts)
    (["add=60000"], 20),
    (["add=60000"], 30),
    (["add=60000"], 40),   # szosty shard: wiecej roznorodnosci obiektow
]

# 256px quotas are smaller per shard for a pure storage reason: a pair costs
# 2 x 256 x 256 x 3 = 384 KB, four times the 96 KB it costs at 128px, so the
# 60000-pair quota that fills 5.8 GB at 128px would want 23 GB — past Kaggle's
# 20 GB notebook output limit. 40000 pairs lands at ~15 GB, with room to spare.
# Offsets continue past the 128px corpus, which consumed shards 0-49 of 'add'.
#
# The types differ per shard here, and that is not arbitrary: **'add' is running
# out.** AnyEdit holds exactly 60 shards of it (6489 rows each) and the 128px
# corpus already consumed 0-49. The first attempt at this table asked for offsets
# 50/57/64/71 by extrapolating the old pattern, and the last two fetched NOTHING
# — both kernels reported COMPLETE while writing 0.0 MB files. Status is not
# content; check `wrote ... for N pairs` in the log.
#
# Drawing from other types costs nothing here, because the first consumer of this
# corpus is the AUTOENCODER, and it does not care what edit an image illustrates —
# it only learns to compress and reconstruct pixels. Diversity is if anything
# better for that. Remaining stock, from data/anyedit_shardmap.json:
# background_change 63 shards, tune_transfer 59+23, color_alter 50 (untouched),
# remove 15, replace 13, appearance_alter 11.
QUOTAS_HI = [
    (["add=40000"], 50),                 # the last of 'add' — shards 50-56
    (["add=40000"], 57),                 # only ~19k really available; 'add' ends here
    (["color_alter=40000"], 0),          # untouched type, 50 shards
    (["background_change=40000"], 10),   # 128px took 0-3; plenty left
]

table = QUOTAS if RES == 128 else QUOTAS_HI
want, skip_shards = table[SHARD]
print(f"res {RES}px, prefix '{PREFIX}', {len(table)}-shard table", flush=True)
print(f"shard {SHARD}: {want}, skipping {skip_shards} shards", flush=True)

if os.path.exists(f"{WORK}/gedit"):
    subprocess.run(["git", "-C", f"{WORK}/gedit", "pull", "--ff-only"], check=True)
else:
    subprocess.run(["git", "clone", "--depth", "1", REPO, f"{WORK}/gedit"], check=True)
os.chdir(f"{WORK}/gedit")
subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                "huggingface_hub", "pyarrow", "pillow"], check=True)

try:
    from kaggle_secrets import UserSecretsClient
    os.environ["HF_TOKEN"] = UserSecretsClient().get_secret("HF_TOKEN")
    print("HF token loaded from Kaggle secrets")
except Exception as e:
    print(f"no HF token ({type(e).__name__}) — downloading anonymously, slower")

subprocess.run([sys.executable, "data/fetch_anyedit.py",
                "--res", str(RES), "--out-prefix", f"{WORK}/{PREFIX}",
                "--skip-shards", str(skip_shards),
                "--want"] + want, check=True)

print(f"\ndone — shard {SHARD}; attach this notebook's output to the training kernel")
for f in sorted(os.listdir(WORK)):
    p = f"{WORK}/{f}"
    if os.path.isfile(p):
        print(f"  {f}  {os.path.getsize(p)/1e6:.1f} MB")
