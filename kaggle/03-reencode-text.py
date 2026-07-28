"""
Kaggle cell — re-run ONLY after the cross-attention architecture change
(2026-07-22): recomputes text embeddings for an EXISTING gedit-data dataset
(full CLIP sequence instead of pooled vector) without re-downloading images.

Settings: GPU off (CPU-only, this is just a CLIP forward pass over prompts),
Internet ON (first run downloads the CLIP model itself, then it's cached).
Inputs: your existing 'gedit-data' dataset. Expect a few minutes, not hours —
this is nothing like the multi-hour image download in 01-prep.py.
"""

import glob
import os
import shutil
import subprocess
import sys

REPO = "https://github.com/JerzySukiennik/g-images.git"
WORK = "/kaggle/working"

if os.path.exists(f"{WORK}/gedit"):
    subprocess.run(["git", "-C", f"{WORK}/gedit", "pull", "--ff-only"], check=True)
else:
    subprocess.run(["git", "clone", "--depth", "1", REPO, f"{WORK}/gedit"], check=True)
os.chdir(f"{WORK}/gedit")
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "transformers"], check=True)

hits = glob.glob("/kaggle/input/**/gedit_meta.json", recursive=True)
if not hits:
    raise SystemExit("attach your existing gedit-data dataset as Input first")
data_dir = os.path.dirname(hits[0])
print(f"existing data: {data_dir}")

# Copy the existing images/prompts/meta into /kaggle/working (Kaggle inputs
# are read-only; reencode_text.py needs to write the updated meta + text.bin
# next to them).
for name in ("gedit_images.bin", "gedit_prompts.json", "gedit_meta.json"):
    src = f"{data_dir}/{name}"
    if os.path.exists(src):
        shutil.copy(src, f"{WORK}/{name}")
        print(f"copied {name} ({os.path.getsize(src)/1e6:.1f} MB)")

subprocess.run([sys.executable, "data/reencode_text.py", "--out-prefix", f"{WORK}/gedit"], check=True)

print("\ndone — save this notebook's output as a Dataset named 'gedit-data' "
      "(replacing/updating the old one) before training")
for f in ("gedit_images.bin", "gedit_text.bin", "gedit_meta.json", "gedit_prompts.json"):
    p = f"{WORK}/{f}"
    if os.path.exists(p):
        print(f"  {f}  {os.path.getsize(p)/1e6:.1f} MB")
