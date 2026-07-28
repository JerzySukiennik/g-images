"""Kaggle/CPU dataset builder for G-Images.

Streams `timbrooks/instructpix2pix-clip-filtered` from Hugging Face (it's far
too large to keep in full — see SPEC.md #2), takes the first `--n` pairs,
resizes both images to `--res` and writes them as a single raw uint8 binary
(mmap-friendly, same idea as MicroG's pl_train.bin) plus a separate float32
binary of frozen CLIP per-token text sequence embeddings for the edit
instructions (full sequence, not pooled — see model/clip_encoder.py for why).

Run once; the output is meant to become a Kaggle Dataset consumed by
train/train.py.
"""

import argparse
import json
import os
import sys

import numpy as np
from PIL import Image


def build(args):
    images_path = f"{args.out_prefix}_images.bin"
    text_path = f"{args.out_prefix}_text.bin"
    meta_path = f"{args.out_prefix}_meta.json"
    prompts_path = f"{args.out_prefix}_prompts.json"

    sample_bytes = 2 * 3 * args.res * args.res
    if os.path.exists(images_path) and os.path.getsize(images_path) >= args.n * sample_bytes:
        print(f"{images_path} already has {args.n} samples — nothing to do")
        return

    from datasets import load_dataset
    ds = load_dataset("timbrooks/instructpix2pix-clip-filtered", split="train", streaming=True)

    # --skip shards the corpus: the stream order is deterministic, so shard k
    # taking rows [k*n, (k+1)*n) gives disjoint, reproducible slices. Needed
    # because the full 313010 pairs at 128px are ~31 GB, over what one Kaggle
    # notebook may write to its output.
    if args.skip:
        print(f"skipping the first {args.skip} rows...", flush=True)
        ds = ds.skip(args.skip)

    prompts = []
    written = 0
    with open(images_path, "wb") as f_img:
        for row in ds:
            if written >= args.n:
                break
            try:
                before = row["original_image"].convert("RGB").resize(
                    (args.res, args.res), Image.LANCZOS)
                after = row["edited_image"].convert("RGB").resize(
                    (args.res, args.res), Image.LANCZOS)
            except Exception as e:
                # A handful of rows in this dataset have corrupt/missing
                # images — skip rather than crash a multi-hour prep run over
                # a few bad rows.
                print(f"skip malformed row (after {written} good pairs): {type(e).__name__}")
                continue
            before_arr = np.array(before, dtype=np.uint8).transpose(2, 0, 1)
            after_arr = np.array(after, dtype=np.uint8).transpose(2, 0, 1)
            f_img.write(before_arr.tobytes())
            f_img.write(after_arr.tobytes())
            prompts.append(row["edit_prompt"])
            written += 1
            if written % 1000 == 0:
                print(f"{written}/{args.n}", flush=True)

    print(f"images done: {written} pairs, {os.path.getsize(images_path)/1e9:.2f} GB")

    # --- text embeddings (frozen CLIP) --------------------------------------
    # Off by default since v3: conditioning is a discrete edit type with its own
    # learned tokens (data/edit_types.py), so nothing reads these any more. They
    # were 3.9 GB per 60k pairs and a CLIP forward pass over every prompt — at
    # the full 313010-pair scale that is ~20 GB and hours of CPU spent producing
    # a file no code opens. --with-clip-text restores them for anyone comparing
    # against the old text-conditioned model.
    meta = {
        "n": written,
        "val_n": min(args.val_n, written // 10),
        "res": args.res,
        "prompts_sample": prompts[:20],
    }
    if args.with_clip_text:
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
        from model.clip_encoder import ClipTextEncoder

        encoder = ClipTextEncoder(device="cpu")
        emb = encoder.encode(prompts, batch_size=args.clip_batch)
        emb.numpy().astype(np.float32).tofile(text_path)
        print(f"text embeddings done: {tuple(emb.shape)}, "
              f"{os.path.getsize(text_path)/1e6:.1f} MB")
        meta.update(text_dim=encoder.embed_dim, seq_len=encoder.seq_len,
                    clip_model=encoder.model.name_or_path)
    else:
        print("skipping CLIP text embeddings (not used since v3)")

    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"wrote {meta_path}")

    # Full prompt list, index-aligned with the images/text binaries — lets
    # runtime/sample_check.py show the actual instruction next to each
    # generated image instead of just judging structural reconstruction.
    with open(prompts_path, "w") as f:
        json.dump(prompts, f)
    print(f"wrote {prompts_path} ({len(prompts)} prompts)")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=20000)
    p.add_argument("--skip", type=int, default=0,
                    help="rows to skip before collecting; use k*n for shard k")
    p.add_argument("--with-clip-text", action="store_true",
                    help="also write the CLIP text embedding binary (unused since v3)")
    p.add_argument("--res", type=int, default=128)
    p.add_argument("--val-n", type=int, default=300)
    p.add_argument("--clip-batch", type=int, default=64)
    p.add_argument("--out-prefix", default="./gedit")
    build(p.parse_args())
