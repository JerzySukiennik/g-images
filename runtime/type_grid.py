"""The v3 acceptance test: one photo, every edit type, side by side.

This replaces guessing from one or two prompts. Because 13 of the types are exact
functions of the input (data/synthetic_edits.py), this grid can show the model's
output next to the GROUND TRUTH for those — the first time in this project that a
checkpoint can be graded against a correct answer rather than by eye alone.

The synthetic rows are the honest measure of whether conditioning works at all:
if the model can't desaturate on demand when it has seen unlimited exact examples
of desaturation, no amount of guidance tuning will save the semantic types.

  python runtime/type_grid.py --ckpt run/ckpt.pt --image photo.jpg --guidance 3
"""

import argparse
import os
import sys

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from model.unet import UNet
from model.scheduler import DiffusionSchedule
from data.edit_types import NULL_TYPE, N_TYPES, SYNTHETIC_TYPES, TYPE_ID, TYPE_NAMES
from data.synthetic_edits import SYNTHETIC


def to_img(t):
    return ((t.clamp(-1, 1) + 1) * 127.5).byte().cpu().numpy().transpose(1, 2, 0)


def main(args):
    device = "cuda" if torch.cuda.is_available() else (
        "mps" if torch.backends.mps.is_available() else "cpu")
    ckpt = torch.load(args.ckpt, map_location=device)
    model = UNet(**ckpt.get("arch", dict(n_types=N_TYPES))).to(device).eval()
    if "ema" in ckpt and not args.no_ema:
        model.load_state_dict({k: v.to(model.state_dict()[k].dtype)
                                for k, v in ckpt["ema"].items()})
        print(f"step {ckpt['step']} (EMA)")
    else:
        model.load_state_dict(ckpt["model"])
        print(f"step {ckpt['step']} (raw)")

    img = Image.open(args.image).convert("RGB").resize((args.res, args.res), Image.LANCZOS)
    before = (torch.from_numpy(np.array(img, dtype=np.uint8).transpose(2, 0, 1)).float()
              / 127.5 - 1.0).unsqueeze(0).to(device)

    schedule = DiffusionSchedule(schedule=args.noise_schedule,
                                  prediction=args.prediction, device=device)
    null_ids = torch.tensor([NULL_TYPE], device=device)

    names = args.types or [n for n in TYPE_NAMES if n != "null"]
    rows = []
    for name in names:
        torch.manual_seed(args.seed)  # same noise for every row
        with torch.no_grad():
            out = schedule.ddim_sample(
                model, before, torch.tensor([TYPE_ID[name]], device=device),
                steps=args.steps, device=device,
                text_uncond=null_ids if args.guidance != 1.0 else None,
                guidance=args.guidance, guidance_rescale=args.guidance_rescale)
        cells = [to_img(before[0]), to_img(out[0])]
        truth = ""
        if name in SYNTHETIC:
            cells.append(to_img(SYNTHETIC[name](before[0])))
            truth = "  (3rd cell = exact ground truth)"
        else:
            # Keep every row the same width so the grid stacks.
            cells.append(np.zeros_like(cells[0]))
        rows.append(np.concatenate(cells, axis=1))
        print(f"  {name}{truth}")
        rows.append(np.full((6, rows[-1].shape[1], 3), 32, dtype=np.uint8))

    Image.fromarray(np.concatenate(rows[:-1], axis=0)).save(args.out)
    print(f"\nsaved {args.out}")
    print("columns: input | model output | ground truth (synthetic types only)")
    print(f"rows, top to bottom: {', '.join(names)}")
    print(f"guidance {args.guidance}, {args.steps} DDIM steps, seed {args.seed}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--image", required=True)
    p.add_argument("--types", nargs="+", help="subset of type names; default is all")
    p.add_argument("--res", type=int, default=128)
    p.add_argument("--steps", type=int, default=100)
    p.add_argument("--guidance", type=float, default=3.0)
    p.add_argument("--guidance-rescale", type=float, default=0.0)
    p.add_argument("--noise-schedule", choices=("cosine", "linear"), default="cosine")
    p.add_argument("--prediction", choices=("v", "eps"), default="v")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no-ema", action="store_true")
    p.add_argument("--out", default="./type_grid.png")
    main(p.parse_args())
