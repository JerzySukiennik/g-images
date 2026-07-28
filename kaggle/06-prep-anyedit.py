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

REPO = "https://github.com/JerzySukiennik/gedit.git"
WORK = "/kaggle/working"
RES = 128

SHARD = int(os.environ.get("GEDIT_SHARD", "0"))

# Per-shard quotas. `add` gets the largest allocation because object addition is
# the capability this whole dataset switch exists to enable — it had 109 usable
# examples in InstructPix2Pix and has ~414000 here.
QUOTAS = [
    ["add=60000", "tune_transfer=60000"],
    ["add=60000", "tune_transfer=60000"],
    ["add=30000", "style_change=25000", "background_change=25000", "replace=20000"],
]
want = QUOTAS[SHARD]
print(f"shard {SHARD}: {want}", flush=True)

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
                "--res", str(RES), "--out-prefix", f"{WORK}/gedit",
                "--want"] + want, check=True)

print(f"\ndone — shard {SHARD}; attach this notebook's output to the training kernel")
for f in sorted(os.listdir(WORK)):
    p = f"{WORK}/{f}"
    if os.path.isfile(p):
        print(f"  {f}  {os.path.getsize(p)/1e6:.1f} MB")
