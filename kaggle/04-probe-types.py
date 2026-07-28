"""Kaggle CPU probe: how much of the real 60k prompt set does the edit-type
taxonomy actually cover?

Runs in seconds and downloads nothing — it reads gedit_prompts.json out of the
gedit-prep notebook output and prints counts to the log. The point is to find out
BEFORE spending GPU hours whether data/edit_types.py matches a useful share of
the corpus or only a few percent, and to surface the most common words among the
prompts it misses so the taxonomy can be extended at the places that pay.

Written after a day in which two full training runs (5.6h and 6h of T4x2) were
spent on hypotheses that a few local minutes of measurement would have killed.

Settings: GPU off, Internet off. Input: the gedit-prep notebook's output.
"""

import glob
import json
import os
import subprocess
import sys
from collections import Counter

REPO = "https://github.com/JerzySukiennik/g-images.git"
BOOT = "/tmp/gedit-boot"

subprocess.run(["git", "clone", "--depth", "1", REPO, BOOT], check=True)
sys.path.insert(0, BOOT)
from data.edit_types import RULES, TYPE_NAMES, classify

hits = glob.glob("/kaggle/input/**/gedit_prompts.json", recursive=True)
if not hits:
    print("gedit_prompts.json not found. /kaggle/input contains:")
    for root, dirs, files in os.walk("/kaggle/input"):
        if root.count("/") - 2 > 3:
            continue
        print("  " + root)
        for f in sorted(files)[:12]:
            print("    " + f)
    raise SystemExit("attach the gedit-prep notebook output")

with open(hits[0]) as f:
    prompts = json.load(f)
print(f"{len(prompts)} prompts from {hits[0]}\n")

counts = Counter()
unmatched = []
for p in prompts:
    t = classify(p)
    if t is None:
        unmatched.append(p)
    else:
        counts[TYPE_NAMES[t]] += 1

matched = len(prompts) - len(unmatched)
print(f"matched {matched}/{len(prompts)} ({100*matched/len(prompts):.1f}%)")
print(f"unmatched {len(unmatched)}\n")

print("per type (a type with only a few hundred pairs can't teach a "
      "transformation, so these counts decide which types survive):")
for name, _ in RULES:
    print(f"  {counts.get(name, 0):>7}  {name}")

print("\nmost common words among UNMATCHED prompts — where extending the "
      "taxonomy would pay most:")
stop = {"the", "a", "an", "it", "into", "to", "of", "in", "on", "with", "and",
        "make", "turn", "have", "his", "her", "their", "as", "is", "be", "let",
        "add", "change", "give", "put", "this", "that", "for", "at", "by"}
words = Counter(w.strip(".,!?\"'") for p in unmatched for w in p.lower().split())
for w, c in words.most_common(60):
    if w and w not in stop and len(w) > 2:
        print(f"  {c:>7}  {w}")

print("\nsample of 40 unmatched prompts:")
for p in unmatched[:40]:
    print(f"  {p}")
