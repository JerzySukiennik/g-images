"""Dataset builder for AnyEdit — the corpus G-Images moved to for semantic edits.

Why AnyEdit replaced InstructPix2Pix
------------------------------------
IP2P has 313010 pairs total and no type labels, so types had to be guessed from
prompt text. Measured counts there were fatal for what Jurek wanted: "add a hat"
appeared 109 times in 60000 pairs, and no single object reached 400. Object
addition was not a training problem, it was an absent-data problem.

AnyEdit has 2485319 pairs and ships an `edit_type` column. Measured from the
parquet footers (see data/anyedit_shardmap.json, built by scanning all 383 shard
footers rather than guessing):

    tune_transfer      ~542000   weather / season / time of day
    add                ~414000   OBJECT ADDITION
    background_change  ~414000
    color_alter        ~338000
    replace            ~104000
    remove              ~89000
    style_change        ~47000
    visual_*           ~330000   unusable: needs a depth map or sketch as a
                                 second input, which our UNet has no port for

`add` alone is larger than the entire IP2P corpus. That is roughly 3800x the
object-addition data we had.

Why this doesn't download everything
------------------------------------
The full set is ~190 GB. It is also more than the model can consume: at 60000
steps and batch 32 it sees 1.9M samples, so 2M pairs would be a single epoch.
~350k pairs is about five epochs and fits three Kaggle CPU notebooks.

The shards are grouped by edit type, so data/anyedit_shardmap.json lets this pick
only the shards holding the wanted types instead of streaming past everything
else — the reason this is hours rather than days.

Output format matches data/fetch_dataset.py so train/train.py needs no new
loader: a raw uint8 images binary, plus JSON sidecars. The per-row AnyEdit
`edit_type` and the raw instruction text are BOTH stored, deliberately: the final
taxonomy (including splitting `add` into specific objects) is then a training-time
decision that can be revised without re-downloading 34 GB.
"""

import argparse
import io
import json
import os
import sys

import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
SHARDMAP = os.path.join(HERE, "anyedit_shardmap.json")
REPO = "Bin1117/AnyEdit"


def pick_shards(wanted, quota, skip_shards=0):
    """Choose shards covering `wanted` types, up to `quota` rows per type, after
    skipping `skip[type]` rows of that type.

    Returns [(shard_index, {types})]. Shards are homogeneous or nearly so, so
    this is mostly "take shards of type X until the quota for X is met".

    `skip_shards` is what makes several prep notebooks give DISJOINT data.
    Without it every notebook starts at the first shard of each type and
    re-downloads identical rows: three kernels once produced 150000 `add` pairs of
    which only 60000 were unique, confirmed by comparing their instruction lists
    (100% of the smaller set appeared in the larger). Notebook k passes
    skip_shards = k * (shards the previous notebooks consumed).
    """
    with open(SHARDMAP) as f:
        smap = json.load(f)
    taken, got = [], {t: 0 for t in wanted}
    seen_shards = 0
    for idx in sorted(smap, key=int):
        types = (smap[idx].get("types") or {})
        useful = {t: n for t, n in types.items()
                  if t in wanted and got[t] < quota[t]}
        if not useful:
            continue
        # Counted in SHARDS, not rows. A row-based skip does not land on shard
        # boundaries — a quota of 60000 consumes ceil(60000/6489) = 10 shards but
        # only writes 60000 rows, so the next notebook skipping "60000 rows"
        # started inside a shard the previous one had already taken. Whole shards
        # make the slices exactly disjoint.
        seen_shards += 1
        if seen_shards <= skip_shards:
            continue
        taken.append((int(idx), set(useful)))
        for t, n in useful.items():
            got[t] += n
        if all(got[t] >= quota[t] for t in wanted):
            break
    return taken, got


def decode(cell):
    """AnyEdit image columns arrive as dicts with raw bytes under 'bytes'."""
    if isinstance(cell, dict):
        cell = cell.get("bytes") or cell.get("path")
    if isinstance(cell, (bytes, bytearray)):
        return Image.open(io.BytesIO(cell))
    if isinstance(cell, str) and os.path.exists(cell):
        return Image.open(cell)
    raise ValueError(f"cannot decode image cell of type {type(cell)}")


def build(args):
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    quota = {t: int(n) for t, n in (kv.split("=") for kv in args.want)}
    shards, projected = pick_shards(set(quota), quota, args.skip_shards)
    if args.skip_shards:
        print(f"skipping the first {args.skip_shards} useful shards")
    print(f"{len(shards)} shards cover the quota; projected rows per type: {projected}",
          flush=True)

    images_path = f"{args.out_prefix}_images.bin"
    got = {t: 0 for t in quota}
    instructions, types = [], []
    written = 0

    with open(images_path, "wb") as f_img:
        for shard_idx, shard_types in shards:
            if all(got[t] >= quota[t] for t in quota):
                break
            name = f"data/train-{shard_idx:05d}-of-00383.parquet"
            local = None
            try:
                # Whole-file download rather than fsspec random access: we need
                # nearly every row of a selected shard anyway, and range-reading
                # a parquet full of embedded images over HTTP was measured taking
                # minutes per shard where a straight download takes seconds.
                # Deleted immediately after so disk stays bounded regardless of
                # how many shards this quota spans.
                local = hf_hub_download(REPO, name, repo_type="dataset")
                # image_file / edited_file are the before/after images
                # (struct<bytes, path>). NOT `input`/`output` — those are plain
                # strings holding the source and target captions, and reading
                # them as images fails at the first row.
                table = pq.read_table(
                    local, columns=["edit_type", "edit_instruction",
                                     "image_file", "edited_file"])
            except Exception as e:
                print(f"shard {shard_idx}: skipped ({type(e).__name__}: {e})", flush=True)
                continue

            col_t = table.column("edit_type").to_pylist()
            col_i = table.column("edit_instruction").to_pylist()
            col_a = table.column("image_file").to_pylist()
            col_b = table.column("edited_file").to_pylist()

            for t, instr, a, b in zip(col_t, col_i, col_a, col_b):
                if t not in quota or got[t] >= quota[t]:
                    continue
                try:
                    before = decode(a).convert("RGB").resize(
                        (args.res, args.res), Image.LANCZOS)
                    after = decode(b).convert("RGB").resize(
                        (args.res, args.res), Image.LANCZOS)
                except Exception:
                    # A handful of rows in any corpus this size are unreadable;
                    # skipping beats crashing a multi-hour run over a few.
                    continue
                f_img.write(np.array(before, dtype=np.uint8).transpose(2, 0, 1).tobytes())
                f_img.write(np.array(after, dtype=np.uint8).transpose(2, 0, 1).tobytes())
                instructions.append(instr or "")
                types.append(t)
                got[t] += 1
                written += 1
                if written % 2000 == 0:
                    print(f"{written} pairs  {got}", flush=True)
            del table, col_a, col_b
            if local and os.path.exists(local):
                # ~500 MB each; keeping them would blow past the notebook's disk
                # long before the quota is met.
                try:
                    os.remove(local)
                except OSError:
                    pass
            print(f"shard {shard_idx} done — totals {got}", flush=True)

    print(f"\nimages: {written} pairs, {os.path.getsize(images_path)/1e9:.2f} GB")

    with open(f"{args.out_prefix}_meta.json", "w") as f:
        json.dump({"n": written, "val_n": min(args.val_n, written // 10),
                   "res": args.res, "source": REPO, "counts": got}, f, indent=2)
    with open(f"{args.out_prefix}_prompts.json", "w") as f:
        json.dump(instructions, f)
    with open(f"{args.out_prefix}_types.json", "w") as f:
        json.dump(types, f)
    print(f"wrote meta/prompts/types sidecars for {written} pairs")

    # Print the head-noun histogram per AnyEdit type straight into the kernel log.
    # The final taxonomy has to split `add` into specific objects — one type per
    # concrete transformation, since a single `add` covering everything is the
    # inconsistent supervision that made the previous model ignore conditioning.
    # Deciding that split needs these counts, and reading them out of the log
    # costs nothing, whereas downloading the 34 GB output to count locally costs
    # an evening.
    import re
    from collections import Counter
    PAT = {
        "add": re.compile(r"\b(?:add|put|place|insert)\s+(?:a|an|some|the)?\s*"
                           r"([a-z]+(?:\s+[a-z]+)?)", re.I),
        "tune_transfer": re.compile(r"\b(?:to|make|change)\s+(?:the\s+)?"
                                     r"(?:weather|season|time|day)?\s*(?:to\s+)?"
                                     r"([a-z]+)\s*$", re.I),
    }
    for t, pat in PAT.items():
        heads = Counter()
        for instr, typ in zip(instructions, types):
            if typ != t:
                continue
            m = pat.search(instr or "")
            if not m:
                continue
            words = m.group(1).lower().split()
            heads[words[-1] if words else ""] += 1
        if not heads:
            continue
        print(f"\n=== '{t}' head nouns (a type needs ~1000+ to be worth its own "
              f"embedding row) ===")
        for w, c in heads.most_common(60):
            print(f"  {c:>7}  {w}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--want", nargs="+", required=True,
                    help="TYPE=COUNT pairs, e.g. add=150000 tune_transfer=150000")
    p.add_argument("--skip-shards", type=int, default=0,
                    help="skip this many otherwise-useful shards first, so parallel "
                         "prep notebooks fetch DISJOINT slices instead of identical rows")
    p.add_argument("--res", type=int, default=128)
    p.add_argument("--val-n", type=int, default=2000)
    p.add_argument("--out-prefix", default="./gedit")
    build(p.parse_args())
