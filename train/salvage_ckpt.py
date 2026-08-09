"""Report what survived in a checkpoint whose write was cut short, and salvage it
if the weights are among it.

Written after 2026-08-09, when a Kaggle session was killed 7 seconds after
logging step 14800 — 8h11m of training, a 412 MB file where a whole one is
1.57 GB. torch.load refuses such a file outright ("unexpected EOF"), which says
nothing about what is actually in it.

The tempting assumption is that torch writes the checkpoint dict section by
section, so a file cut inside 'opt' would still hold all of 'model'. **That is
false.** torch's legacy format sorts storages by key — and the keys are id-like
numbers, so the on-disk order has no relationship to the dict's structure. In the
2026-08-09 file the 531 storages below the cut were optimizer moments, every one
of them, and not a single one of the 321 weight tensors. Nothing was recoverable.

So this tool answers the question rather than assuming it: it maps each storage
to its section (traversal order, from the pickle) and to its position on disk
(sorted order, from the key list), then reports per-section survival. If and only
if every 'model' storage is below the cut does it rebuild a loadable file.

Repair, where possible, is by construction rather than reinterpretation: each
storage is an 8-byte little-endian element count followed by its elements, and
torch validates every header, so padding the tail with raw zeros fails. This
finishes the storage that was cut and appends well-formed empty records for the
rest.

The real lesson lives in train.py: save to a temp file and rename. Then a kill
mid-write costs one checkpoint interval instead of the whole run.

Usage:
    python train/salvage_ckpt.py <truncated.pt> [<out.pt>]
"""

import argparse
import collections
import os
import pickle
import shutil
import struct
import sys

ELEM_SIZE = {
    "FloatStorage": 4, "DoubleStorage": 8, "HalfStorage": 2, "BFloat16Storage": 2,
    "LongStorage": 8, "IntStorage": 4, "ShortStorage": 2,
    "ByteStorage": 1, "CharStorage": 1, "BoolStorage": 1,
}

_SAFE = {
    ("collections", "OrderedDict"): collections.OrderedDict,
    ("builtins", "dict"): dict, ("builtins", "list"): list,
    ("builtins", "set"): set, ("builtins", "tuple"): tuple,
    ("builtins", "int"): int, ("builtins", "float"): float,
    ("builtins", "str"): str, ("builtins", "bool"): bool,
    ("argparse", "Namespace"): argparse.Namespace,
}


def _named_stub(name):
    class _Stub:
        __tname__ = name

        def __init__(self, *a, **k):
            pass

        def __setstate__(self, s):
            pass
    return _Stub


def scan(path):
    """Layout of the file, without materializing a single tensor.

    Returns (obj, disk_keys, meta, data_start, traversal) where `disk_keys` is
    on-disk order (sorted by key) and `traversal` is the order the storages were
    reached while pickling — the only thing that ties a storage to its section.
    """
    records = []

    class _U(pickle.Unpickler):
        def find_class(self, module, name):
            return _SAFE.get((module, name)) or _named_stub(name)

        def persistent_load(self, pid):
            records.append(pid)
            return ("storage", pid[2])

    with open(path, "rb") as f:
        for _ in range(3):              # magic, protocol, sys_info
            _U(f).load()
        obj = _U(f).load()              # the dict — written before any tensor data
        disk_keys = _U(f).load()        # storage keys, in on-disk order
        data_start = f.tell()

    meta = {r[2]: (getattr(r[1], "__tname__", "FloatStorage"), r[4]) for r in records}
    return obj, disk_keys, meta, data_start, [r[2] for r in records]


def analyse(path):
    obj, disk_keys, meta, data_start, traversal = scan(path)
    size = os.path.getsize(path)

    plan, pos, cut_at = [], data_start, None
    for i, k in enumerate(disk_keys):
        tname, numel = meta.get(k, ("FloatStorage", 0))
        nbytes = numel * ELEM_SIZE.get(tname, 4)
        plan.append((k, numel, nbytes))
        if cut_at is None and pos + 8 + nbytes > size:
            cut_at = i
        pos += 8 + nbytes

    # Section membership: persistent_load fires in traversal order, so the
    # per-section tensor counts partition that sequence in the dict's own order.
    sections, at = {}, 0
    for key, value in obj.items():
        n = len(value) if hasattr(value, "keys") and key != "arch" else 0
        if key == "opt":
            n = len(traversal) - sum(
                len(v) for k2, v in obj.items()
                if k2 != "opt" and hasattr(v, "keys") and k2 != "arch")
        if n:
            sections[key] = traversal[at:at + n]
            at += n
    return obj, plan, cut_at, sections, pos, size


def main(src, dst=None):
    obj, plan, cut_at, sections, required, size = analyse(src)
    print(f"{src}: {size/1e9:.2f} GB present, {required/1e9:.2f} GB declared")

    if cut_at is None:
        print("file is complete — not truncated")
        if dst:
            shutil.copy(src, dst)
        return 0

    survived = {k for k, _, _ in plan[:cut_at]}
    print(f"cut at storage {cut_at}/{len(plan)} (on-disk order is sorted by key, "
          "so this does not respect section boundaries)")
    for name, keys in sections.items():
        have = len(set(keys) & survived)
        print(f"  {name}: {have}/{len(keys)} storages survived "
              f"({100*have/max(len(keys),1):.0f}%)")

    model_keys = set(sections.get("model", []))
    if not model_keys or model_keys - survived:
        missing = len(model_keys - survived)
        print(f"\nNOT SALVAGEABLE: {missing} of {len(model_keys)} weight tensors "
              "fell past the cut. Optimizer moments without weights are worthless.")
        return 1

    if not dst:
        print("\nsalvageable — rerun with an output path to write it")
        return 0

    shutil.copy(src, dst + ".tmp")
    with open(dst + ".tmp", "ab") as f:
        head = sum(8 + b for _, _, b in plan[:cut_at])
        _, numel, nbytes = plan[cut_at]
        written = size - (required - sum(8 + b for _, _, b in plan[cut_at:]))
        del head
        if written >= 8:
            f.write(b"\0" * (8 + nbytes - written))
        else:
            f.write(struct.pack("<q", numel))
            f.write(b"\0" * nbytes)
        for _, numel, nbytes in plan[cut_at + 1:]:
            f.write(struct.pack("<q", numel))
            f.write(b"\0" * nbytes)

    import torch
    try:
        ckpt = torch.load(dst + ".tmp", map_location="cpu", weights_only=False)
    finally:
        os.remove(dst + ".tmp")

    out = {"model": ckpt["model"], "step": ckpt["step"]}
    if "arch" in ckpt:
        out["arch"] = ckpt["arch"]
    tensors = [(k, v) for k, v in out["model"].items() if torch.is_tensor(v)]
    bad = [k for k, v in tensors if not torch.isfinite(v).all()]
    if bad:
        raise SystemExit(f"{len(bad)} weight tensors hold inf/nan, first {bad[:3]}")
    zero = [k for k, v in tensors if v.numel() > 1 and float(v.abs().sum()) == 0.0]
    if zero:
        raise SystemExit(f"{len(zero)} tensors are entirely zero, first {zero[:3]}")

    total = sum(v.numel() for _, v in tensors)
    print(f"\nsalvaged step {out['step']}: {len(tensors)} tensors, "
          f"{total/1e6:.1f}M parameters, all finite and non-zero")
    print("dropped: optimizer state and EMA")
    torch.save(out, dst, _use_new_zipfile_serialization=False)
    print(f"wrote {dst} ({os.path.getsize(dst)/1e9:.2f} GB)")
    return 0


if __name__ == "__main__":
    if len(sys.argv) not in (2, 3):
        raise SystemExit(__doc__)
    sys.exit(main(sys.argv[1], sys.argv[2] if len(sys.argv) == 3 else None))
