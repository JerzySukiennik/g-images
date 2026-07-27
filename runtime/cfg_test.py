"""Sweep classifier-free guidance scales on an existing checkpoint.

Why this exists: every judgement of Gedit's quality so far was made at
guidance = 1.0, i.e. sampling the model's raw prediction. That is not how
text-conditioned diffusion is sampled — InstructPix2Pix's own results use
text guidance ~7.5. With `before` concatenated on the channel axis, copying
the input is always the cheapest way to lower the training MSE, so the text
signal is real but weak (the fixed-seed real-vs-zero-vs-random-text test on
2026-07-22 confirmed it is measurably there) and needs amplifying at sampling
time to actually show up.

This sweeps several scales side by side on the SAME noise seed, so the only
thing varying across a row is the guidance strength. Runs on CPU/MPS against
a checkpoint that is already trained — no GPU quota needed to find out whether
guidance is the missing piece before spending another Kaggle session.

Caveat: train/train.py does not yet train with conditioning dropout, so the
null-text branch is extrapolation rather than something the model was taught.
Text-only guidance (the default here) is the honest measurement; --image-guidance
additionally zeroes `before`, which is further off-distribution.

  python runtime/cfg_test.py --ckpt ~/Downloads/ckpt.pt --image photo.jpg \
      --prompt "make it black and white"
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
from model.clip_encoder import ClipTextEncoder


def load_image(path, res):
    img = Image.open(path).convert("RGB").resize((res, res), Image.LANCZOS)
    arr = np.array(img, dtype=np.uint8).transpose(2, 0, 1)
    return torch.from_numpy(arr).float() / 127.5 - 1.0


def to_img(t):
    return ((t.clamp(-1, 1) + 1) * 127.5).byte().cpu().numpy().transpose(1, 2, 0)


def label_strip(width, text, height=16):
    """Tiny text-free separator carrying the column order in the filename
    instead — PIL's default font is unreliable across installs, and the point
    of this script is the pixels, not the chrome."""
    return np.full((height, width, 3), 32, dtype=np.uint8)


def main(args):
    device = "cuda" if torch.cuda.is_available() else (
        "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"device: {device}")

    ckpt = torch.load(args.ckpt, map_location=device)
    model = UNet().to(device).eval()
    if "ema" in ckpt and not args.no_ema:
        model.load_state_dict({k: v.to(model.state_dict()[k].dtype) for k, v in ckpt["ema"].items()})
        print(f"checkpoint step {ckpt['step']} (EMA weights)")
    else:
        model.load_state_dict(ckpt["model"])
        print(f"checkpoint step {ckpt['step']} (raw weights)")

    print("loading frozen CLIP text encoder...")
    encoder = ClipTextEncoder(device=device)
    prompts = args.prompt
    text_seq = encoder.encode(prompts).to(device)
    text_uncond_one = encoder.encode([""]).to(device)

    before = load_image(args.image, args.res).unsqueeze(0).to(device)
    schedule = DiffusionSchedule(device=device)
    scales = args.scales

    rows = []
    for pi, prompt in enumerate(prompts):
        cols = [to_img(before[0])]
        for s in scales:
            # Same seed for every cell: guidance strength is the only variable.
            torch.manual_seed(args.seed)
            with torch.no_grad():
                out = schedule.ddim_sample(
                    model, before, text_seq[pi:pi + 1], steps=args.steps, device=device,
                    text_uncond=text_uncond_one, guidance=s,
                    image_guidance=args.image_guidance,
                    guidance_rescale=args.guidance_rescale,
                    dynamic_threshold=args.dynamic_threshold)
            cols.append(to_img(out[0]))
            print(f"  [{prompt!r}] guidance={s} done", flush=True)
        rows.append(np.concatenate(cols, axis=1))
        if pi < len(prompts) - 1:
            rows.append(label_strip(rows[-1].shape[1], ""))

    Image.fromarray(np.concatenate(rows, axis=0)).save(args.out)
    print(f"\nsaved {args.out}")
    print(f"columns: input | " + " | ".join(f"g={s}" for s in scales))
    print(f"rows:    " + "  /  ".join(repr(p) for p in prompts))
    print(f"image_guidance={args.image_guidance}, {args.steps} DDIM steps, seed {args.seed}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--image", required=True)
    p.add_argument("--prompt", action="append", required=True,
                    help="repeat for several prompts; each becomes one row")
    p.add_argument("--ckpt", required=True)
    p.add_argument("--res", type=int, default=128)
    p.add_argument("--steps", type=int, default=100)
    p.add_argument("--scales", type=float, nargs="+", default=[1.0, 3.0, 5.0, 7.5, 12.0],
                    help="text guidance scales; 1.0 reproduces every earlier test")
    p.add_argument("--image-guidance", type=float, default=1.0,
                    help=">1.0 adds the zeroed-`before` unconditional branch (3 passes "
                         "per step, and off-distribution until conditioning dropout is trained)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--guidance-rescale", type=float, default=0.0,
                    help="0..1, Lin et al. guidance rescale; 0.7 is the paper's value")
    p.add_argument("--dynamic-threshold", type=float, default=0.0,
                    help="percentile for Imagen dynamic thresholding, e.g. 0.995; "
                         "0 keeps the plain clamp to [-1,1]")
    p.add_argument("--no-ema", action="store_true",
                    help="sample from the raw weights even when the checkpoint has EMA")
    p.add_argument("--out", default="./cfg_sweep.png")
    main(p.parse_args())
