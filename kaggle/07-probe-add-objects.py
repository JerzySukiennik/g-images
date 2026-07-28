"""Kaggle CPU probe: what objects does the prepared AnyEdit data actually add?

Reads the prompts/types sidecars from the three gedit-anyedit prep notebooks —
never the 34 GB of images — and prints a head-noun histogram. That decides which
objects earn their own type in the taxonomy, at a threshold of ~1000 examples.

This runs separately from prep on purpose: the prep kernels were already running
with an earlier, broken extractor (it returned "table"/"of"/"taking" — the place
the object went, not the object) and the fix landed afterwards. Since the raw
instruction text is stored alongside the images, the histogram is recomputable
without re-downloading anything, which is exactly why it is stored.

Settings: GPU off, Internet on. Inputs: the three prep notebook outputs.
"""

import glob
import json
import subprocess
import sys
from collections import Counter

REPO = "https://github.com/JerzySukiennik/gedit.git"
BOOT = "/tmp/gedit-probe-add"
subprocess.run(["git", "clone", "--depth", "1", REPO, BOOT], check=True)
sys.path.insert(0, BOOT)
from data.edit_types import add_object

hits = sorted(glob.glob("/kaggle/input/**/gedit_prompts.json", recursive=True))
if not hits:
    raise SystemExit("attach the gedit-anyedit-* notebook outputs")

objects, per_type, total = Counter(), Counter(), 0
for pj in hits:
    tj = pj.replace("_prompts.json", "_types.json")
    prompts = json.load(open(pj))
    types = json.load(open(tj)) if glob.os.path.exists(tj) else ["?"] * len(prompts)
    print(f"{pj}: {len(prompts)} rows")
    for instr, t in zip(prompts, types):
        per_type[t] += 1
        total += 1
        if t == "add":
            head = add_object(instr)
            if head:
                objects[head] += 1

print(f"\ntotal {total} pairs across {len(hits)} shards")
print("per AnyEdit type:")
for t, c in per_type.most_common():
    print(f"  {c:>8}  {t}")

print(f"\nadd-object head nouns ({sum(objects.values())} parsed):")
TH = 1000
for w, c in objects.most_common(80):
    print(f"  {c:>7}  {w}{'   <-- own type' if c >= TH else ''}")

winners = [w for w, c in objects.items() if c >= TH]
print(f"\n{len(winners)} objects at the {TH} threshold: {sorted(winners)}")
print(f"covering {sum(c for w, c in objects.items() if c >= TH)} pairs")
print("\nhat specifically:", objects.get("hat", 0))
