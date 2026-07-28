"""Export a trained UNet checkpoint to ONNX for inference in gzowo-ai's Node
bridge via onnxruntime-node (SPEC.md #3).

The frozen CLIP text encoder is NOT exported here — the Node side uses
transformers.js to tokenize and encode the free-form instruction directly in
JS (it already ships CLIP text models runnable in Node/browser), so only the
part of G-Images that's actually trained from scratch needs converting.
"""

import argparse
import os
import sys

import numpy as np
import onnxruntime as ort
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from model.unet import UNet


def main(args):
    ckpt = torch.load(args.ckpt, map_location="cpu")
    model = UNet(text_dim=args.text_dim).eval()
    model.load_state_dict(ckpt["model"])
    print(f"loaded checkpoint at step {ckpt['step']}")

    dummy_x = torch.randn(1, 6, args.res, args.res)
    dummy_t = torch.zeros(1, dtype=torch.long)
    dummy_text = torch.randn(1, args.seq_len, args.text_dim)

    torch.onnx.export(
        model, (dummy_x, dummy_t, dummy_text), args.out,
        input_names=["x", "t", "text_seq"], output_names=["pred_noise"],
        dynamic_axes={"x": {0: "batch"}, "t": {0: "batch"},
                       "text_seq": {0: "batch"}, "pred_noise": {0: "batch"}},
        opset_version=17,
    )
    print(f"exported {args.out}")

    # Sanity check: the exported graph must actually run and match PyTorch —
    # an export that "succeeds" but silently diverges (e.g. a traced branch
    # that doesn't generalize) is worse than a loud failure.
    sess = ort.InferenceSession(args.out)
    onnx_out = sess.run(None, {"x": dummy_x.numpy(), "t": dummy_t.numpy(),
                                "text_seq": dummy_text.numpy()})[0]
    with torch.no_grad():
        torch_out = model(dummy_x, dummy_t, dummy_text).numpy()
    diff = np.abs(onnx_out - torch_out).max()
    print(f"max abs diff onnx vs torch: {diff:.6f}")
    if diff > 1e-3:
        raise SystemExit(f"ONNX output diverges from PyTorch by {diff} — do not ship this export")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--out", default="./gedit_unet.onnx")
    p.add_argument("--res", type=int, default=128)
    p.add_argument("--text-dim", type=int, default=512)
    p.add_argument("--seq-len", type=int, default=32)
    main(p.parse_args())
