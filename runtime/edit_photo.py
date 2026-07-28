"""Manual hands-on test: apply one of Gedit's edit types to any photo from the
terminal.

v3 takes --type (a name from data/edit_types.py) instead of a free-form --prompt.
The model no longer encodes sentences: conditioning is a discrete edit type with
its own learned tokens, because measurement showed the CLIP-text version had
learned to ignore the text almost entirely (cos(e_cond, e_null) = 1.000, see
runtime/guidance_probe.py) and the corpus could never have taught it otherwise —
"make it black and white" appears 588 times in 60000 pairs.

Mapping a free-form sentence onto the nearest type is an inference-time concern
and deliberately deferred; --list shows what the model actually knows.

  python runtime/edit_photo.py --image photo.jpg --type black_and_white \
      --ckpt run/ckpt.pt --guidance 3
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
from data.edit_types import NULL_TYPE, N_TYPES, TYPE_ID, TYPE_NAMES


def load_image(path, res):
    img = Image.open(path).convert("RGB").resize((res, res), Image.LANCZOS)
    arr = np.array(img, dtype=np.uint8).transpose(2, 0, 1)
    return torch.from_numpy(arr).float() / 127.5 - 1.0


def to_img(t):
    arr = ((t.clamp(-1, 1) + 1) * 127.5).byte().cpu().numpy()
    return arr.transpose(1, 2, 0)


def load_model(ckpt_path, device, no_ema=False):
    ckpt = torch.load(ckpt_path, map_location=device)
    model = UNet(n_types=N_TYPES).to(device).eval()
    # Sample from the EMA weights when the checkpoint has them — that is what
    # they are for; the live weights keep bouncing around the optimum.
    if "ema" in ckpt and not no_ema:
        state = {k: v.to(model.state_dict()[k].dtype) for k, v in ckpt["ema"].items()}
        which = "EMA"
    else:
        state = ckpt["model"]
        which = "raw"
    model.load_state_dict(state)
    print(f"checkpoint step {ckpt['step']} ({which} weights)")
    return model


def main(args):
    if args.list:
        print("edit types this model knows:")
        for i, name in enumerate(TYPE_NAMES):
            note = "  (null / unconditional branch)" if i == NULL_TYPE else ""
            print(f"  {i:>3}  {name}{note}")
        return

    if args.type not in TYPE_ID:
        raise SystemExit(f"unknown type {args.type!r} — run with --list to see them all")

    device = "cuda" if torch.cuda.is_available() else (
        "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"device: {device}")

    model = load_model(args.ckpt, device, args.no_ema)
    before = load_image(args.image, args.res).unsqueeze(0).to(device)
    type_ids = torch.tensor([TYPE_ID[args.type]], device=device)
    null_ids = torch.tensor([NULL_TYPE], device=device)

    schedule = DiffusionSchedule(schedule=args.noise_schedule,
                                  prediction=args.prediction, device=device)
    print(f"sampling {args.type!r} ({args.steps} DDIM steps, guidance {args.guidance})...")
    with torch.no_grad():
        generated = schedule.ddim_sample(
            model, before, type_ids, steps=args.steps, device=device,
            text_uncond=null_ids if args.guidance != 1.0 else None,
            guidance=args.guidance, image_guidance=args.image_guidance,
            guidance_rescale=args.guidance_rescale)

    grid = np.concatenate([to_img(before[0]), to_img(generated[0])], axis=1)
    Image.fromarray(grid).save(args.out)
    print(f"saved {args.out}  (left: input at {args.res}px | right: edited)")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--image", help="path to any photo")
    p.add_argument("--type", help="edit type name; see --list")
    p.add_argument("--list", action="store_true", help="print the known edit types and exit")
    p.add_argument("--ckpt")
    p.add_argument("--res", type=int, default=128,
                    help="must match the resolution the checkpoint was trained at")
    p.add_argument("--steps", type=int, default=100,
                    help="DDIM sampling steps. Measured: 20 steps hid real structure "
                         "behind a washed-out result that looked the same regardless of "
                         "conditioning. Don't judge quality below ~50.")
    p.add_argument("--guidance", type=float, default=3.0,
                    help="classifier-free guidance on the edit type. 1.0 is the raw "
                         "prediction. On the previous (CLIP text) model anything above 3 "
                         "saturated into colour noise because conditioning was too weak "
                         "for guidance to have anything to amplify; with discrete types "
                         "higher values are worth re-testing.")
    p.add_argument("--image-guidance", type=float, default=1.0)
    p.add_argument("--guidance-rescale", type=float, default=0.0,
                    help="0..1, Lin et al. rescale; 0.7 is the paper's value")
    p.add_argument("--noise-schedule", choices=("cosine", "linear"), default="cosine")
    p.add_argument("--prediction", choices=("v", "eps"), default="v")
    p.add_argument("--no-ema", action="store_true")
    p.add_argument("--out", default="./edited.png")
    args = p.parse_args()
    if not args.list and not (args.image and args.type and args.ckpt):
        p.error("--image, --type and --ckpt are required unless --list is given")
    main(args)
