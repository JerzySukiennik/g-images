"""Manual hands-on test: edit ANY photo with ANY free-form instruction using
a trained checkpoint, from the terminal — no gzowo-ai wiring needed yet.

This is the same model/scheduler.py + model/clip_encoder.py pipeline that
will eventually run inside gzowo-ai's Node bridge (via ONNX), just driven
directly in Python so Jurek can try real photos/prompts before that
integration is built (see SPEC.md "Kolejne kroki" #4-6).
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
    arr = ((t.clamp(-1, 1) + 1) * 127.5).byte().cpu().numpy()
    return arr.transpose(1, 2, 0)


def main(args):
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"device: {device}")

    ckpt = torch.load(args.ckpt, map_location=device)
    model = UNet().to(device).eval()
    # Sample from the EMA weights when the checkpoint has them — that's what
    # they're for. Checkpoints from before train/train.py tracked EMA only have
    # "model", so this stays backward compatible.
    if "ema" in ckpt and not args.no_ema:
        model.load_state_dict({k: v.to(model.state_dict()[k].dtype) for k, v in ckpt["ema"].items()})
        print(f"loaded checkpoint at step {ckpt['step']} (EMA weights)")
    else:
        model.load_state_dict(ckpt["model"])
        print(f"loaded checkpoint at step {ckpt['step']} (raw weights)")

    print("loading CLIP text encoder (frozen, first run downloads it)...")
    encoder = ClipTextEncoder(device=device)
    text_seq = encoder.encode([args.prompt]).to(device)
    text_uncond = encoder.encode([""]).to(device) if args.guidance != 1.0 else None

    before = load_image(args.image, args.res).unsqueeze(0).to(device)

    schedule = DiffusionSchedule(device=device)
    print(f"sampling ({args.steps} DDIM steps, guidance {args.guidance})...")
    with torch.no_grad():
        generated = schedule.ddim_sample(
            model, before, text_seq, steps=args.steps, device=device,
            text_uncond=text_uncond, guidance=args.guidance,
            image_guidance=args.image_guidance)

    grid = np.concatenate([to_img(before[0]), to_img(generated[0])], axis=1)
    Image.fromarray(grid).save(args.out)
    print(f"saved {args.out}  (left: input at {args.res}px | right: edited)")
    print(f"prompt: {args.prompt!r}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--image", required=True, help="path to any photo")
    p.add_argument("--prompt", required=True, help="free-form edit instruction")
    p.add_argument("--ckpt", required=True)
    p.add_argument("--res", type=int, default=128, help="must match the resolution the checkpoint was trained at")
    p.add_argument("--steps", type=int, default=100,
                    help="DDIM sampling steps. Measured 2026-07-22: 20 steps gave a "
                         "washed-out, near-identical-regardless-of-prompt result on the "
                         "cross-attention checkpoint that mostly turned out to be a "
                         "sampling-discretization artifact, not (only) undertraining — "
                         "100 steps on the SAME checkpoint showed real structure/color the "
                         "20-step version hid. Don't judge cross-attention quality at <50.")
    p.add_argument("--guidance", type=float, default=1.0,
                    help="classifier-free guidance scale on the text. 1.0 = the raw model "
                         "prediction, which is how every test before 2026-07-27 was run and "
                         "why edits barely showed. InstructPix2Pix uses ~7.5, but until a "
                         "checkpoint is trained with conditioning dropout, scales above ~3 "
                         "saturate into colour garbage (measured on step 48400 — see "
                         "runtime/cfg_test.py). Raise the default once such a checkpoint exists.")
    p.add_argument("--image-guidance", type=float, default=1.0,
                    help="second IP2P scale, on `before` (~1.5 there). Needs a checkpoint "
                         "trained with --image-dropout; costs a third forward pass per step.")
    p.add_argument("--no-ema", action="store_true",
                    help="sample from the raw weights even when the checkpoint has EMA")
    p.add_argument("--out", default="./edited.png")
    main(p.parse_args())
