"""Render the evaluation grid ON Kaggle and emit only the PNG.

Downloading a checkpoint to judge it does not scale: the Kaggle CLI buffers the
whole file in memory before writing (`out.write(download_response.content)`), and
a 1.13 GB checkpoint against ~1 GB of free RAM on the Mac thrashes for twenty
minutes and may never finish. The model is already sitting on Kaggle next to a
GPU; bringing the weights home to look at six pictures is backwards.

This runs there instead and writes one PNG, which is a few hundred KB.

The reference photo is fetched from the repo so nothing depends on local files.
Rows are input | model | exact ground truth, where a synthetic type has one.
"""

import os
import sys

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.unet import UNet
from model.scheduler import DiffusionSchedule
from data.edit_types import N_TYPES, TYPE_ID, NULL_TYPE
from data.synthetic_edits import SYNTHETIC

TYPES = ["black_and_white", "inverted", "add_hat", "add_cat", "add_balloon",
         "season_winter", "time_night", "style_change"]


def main(ckpt_path, image_path, out_path, guidance=3.0, steps=100):
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    # Same sanity check the local scripts now carry: a backend that silently
    # returns zeros produces confident-looking nonsense.
    assert float(torch.randn(1, 3, 128, 128, device=dev).norm()) > 50, "backend zwraca zera"

    ck = torch.load(ckpt_path, map_location=dev)
    model = UNet(n_types=N_TYPES).to(dev).eval()
    state = ck.get("ema", ck["model"])
    model.load_state_dict({k: v.to(model.state_dict()[k].dtype) for k, v in state.items()})
    print(f"step {ck['step']} on {dev}", flush=True)

    img = Image.open(image_path).convert("RGB").resize((128, 128), Image.LANCZOS)
    before = (torch.from_numpy(np.array(img, dtype=np.uint8).transpose(2, 0, 1)).float()
              / 127.5 - 1.0).unsqueeze(0).to(dev)
    sched = DiffusionSchedule(schedule="cosine", prediction="v", device=dev)

    def to_img(t):
        return ((t.clamp(-1, 1) + 1) * 127.5).byte().cpu().numpy().transpose(1, 2, 0)

    rows = []
    for name in TYPES:
        torch.manual_seed(0)  # same noise every row: only the type varies
        with torch.no_grad():
            out = sched.ddim_sample(
                model, before, torch.tensor([TYPE_ID[name]], device=dev),
                steps=steps, device=dev,
                text_uncond=torch.tensor([NULL_TYPE], device=dev), guidance=guidance)
        cells = [to_img(before[0]), to_img(out[0])]
        cells.append(to_img(SYNTHETIC[name](before[0])) if name in SYNTHETIC
                     else np.zeros_like(cells[0]))
        rows.append(np.concatenate(cells, axis=1))
        rows.append(np.full((6, rows[-1].shape[1], 3), 32, dtype=np.uint8))
        print(f"  {name}", flush=True)

    Image.fromarray(np.concatenate(rows[:-1], axis=0)).save(out_path)
    print(f"saved {out_path}; rows: {', '.join(TYPES)}")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--image", required=True)
    p.add_argument("--out", default="/kaggle/working/eval.png")
    p.add_argument("--guidance", type=float, default=3.0)
    p.add_argument("--steps", type=int, default=100)
    a = p.parse_args()
    main(a.ckpt, a.image, a.out, a.guidance, a.steps)
