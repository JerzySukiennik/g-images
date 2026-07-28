"""Kaggle CPU probe: is object-adding trainable if split into specific objects?

The v3 taxonomy dropped `add_object` because its 4483 prompts name 4483 different
things, and one type covering all of them is the inconsistent supervision that
taught the previous model to ignore conditioning entirely. But the same fix that
worked for filters applies here: one type per concrete transformation. "add a hat"
is a consistent transformation; "add something" is not.

So this counts what actually gets added, across the whole corpus rather than just
the prompts an earlier rule matched. A noun with a few thousand examples could
carry its own type; one with forty cannot. The answer decides whether "add a hat"
comes back into scope or stays out.

Settings: GPU off, Internet on. Input: the gedit-prep notebook output.
"""

import glob
import json
import re
import sys
from collections import Counter

hits = glob.glob("/kaggle/input/**/gedit_prompts.json", recursive=True)
if not hits:
    raise SystemExit("attach the gedit-prep notebook output")
prompts = json.load(open(hits[0]))
print(f"{len(prompts)} prompts\n")

# "add a hat", "give him sunglasses", "put a crown on her" -> the noun phrase that
# follows. Deliberately loose: the point is to see the shape of the distribution,
# not to build the final rule.
ADD = re.compile(
    r"\b(?:add|put|give|place|insert)\s+(?:a|an|some|the)?\s*([a-z]+(?:\s+[a-z]+)?)",
    re.I)

heads = Counter()
adders = 0
for p in prompts:
    m = ADD.search(p)
    if not m:
        continue
    adders += 1
    phrase = m.group(1).lower().split()
    # Head noun: last word of the short phrase, minus a trailing preposition.
    head = phrase[-1] if phrase[-1] not in ("on", "to", "in", "of", "with") else phrase[0]
    heads[head] += 1

print(f"{adders} prompts look like an addition ({100*adders/len(prompts):.1f}%)\n")
print("most-added things — a type needs roughly 400+ to be trainable, going by "
      "what the surviving semantic types have:")
for w, c in heads.most_common(50):
    mark = "  <-- trainable" if c >= 400 else ""
    print(f"  {c:>6}  {w}{mark}")

trainable = [w for w, c in heads.items() if c >= 400]
print(f"\n{len(trainable)} candidate object types at the 400 threshold: {trainable}")
print(f"{sum(c for w, c in heads.items() if c >= 400)} pairs covered by them")

print("\nsample 'add' prompts:")
shown = 0
for p in prompts:
    if ADD.search(p):
        print(f"  {p}")
        shown += 1
        if shown >= 30:
            break
