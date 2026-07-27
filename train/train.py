"""Gedit training loop — diffusion noise-prediction loss on (before, after,
text_seq) triples prepared by data/fetch_dataset.py. Mirrors MicroG's
train/train.py conventions: gradient accumulation, DataParallel across
Kaggle's T4x2, checkpointing every --ckpt-every steps so a killed 12h session
resumes clean.

Revision 2026-07-27, after step 48400 still failed to follow instructions like
"make it black and white". Four additions, none of which change the
architecture — an existing checkpoint resumes into this loop cleanly:

  1. Conditioning dropout (--text-dropout / --image-dropout). Randomly
     replaces the prompt with the null-text embedding and/or `before` with
     zeros, which is what teaches the model the unconditional distribution
     that classifier-free guidance interpolates away from. Without it,
     sampling with guidance (model/scheduler.py) is extrapolating into inputs
     the model never saw. InstructPix2Pix trains exactly this way.
  2. EMA weights (--ema-decay). Diffusion models are sampled from an
     exponential moving average of the weights, not the raw ones; skipping it
     costs visible sharpness. Stored alongside `model` in the checkpoint.
  3. Min-SNR-gamma loss weighting (--min-snr-gamma). Plain MSE under uniform
     `t` is dominated by high-noise timesteps, where predicting roughly the
     mean is already a large win — which is why loss flattened around step
     300 while visual quality kept improving for thousands of steps after.
     Down-weighting those re-aims the gradient at the steps that carry image
     content.
  4. Horizontal-flip augmentation (--no-flip to disable). 60k pairs over tens
     of epochs with zero augmentation; flipping both images together is free
     2x data. Skipped for prompts mentioning left/right, where it would
     contradict the instruction.
"""

import argparse
import json
import math
import os
import random
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# Running this as `python train/train.py` puts train/ on sys.path, not the
# repo root — the same fix data/fetch_dataset.py already needed for its
# `from model.clip_encoder import ...` import.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from model.unet import UNet
from model.scheduler import DiffusionSchedule


class PairDataset(Dataset):
    def __init__(self, data_prefix, split="train", flip=False):
        with open(f"{data_prefix}_meta.json") as f:
            meta = json.load(f)
        self.res = meta["res"]
        self.text_dim = meta["text_dim"]
        self.seq_len = meta["seq_len"]
        n, val_n = meta["n"], meta["val_n"]
        self.images = np.memmap(f"{data_prefix}_images.bin", dtype=np.uint8, mode="r",
                                 shape=(n, 2, 3, self.res, self.res))
        self.text = np.memmap(f"{data_prefix}_text.bin", dtype=np.float32, mode="r",
                               shape=(n, self.seq_len, self.text_dim))
        self.idx = range(0, n - val_n) if split == "train" else range(n - val_n, n)
        # Never augment the validation split — its whole job is being the same
        # measurement every time.
        self.flip = flip and split == "train"

        # A flip contradicts any prompt that names a side ("move the cup to the
        # left"), so those rows opt out. The prompt list is index-aligned with
        # the binaries; datasets prepared before it was added simply don't get
        # flipped, rather than getting flipped unsafely.
        self.flippable = None
        prompts_path = f"{data_prefix}_prompts.json"
        if self.flip:
            if os.path.exists(prompts_path):
                with open(prompts_path) as f:
                    prompts = json.load(f)
                self.flippable = [
                    not any(w in p.lower() for w in ("left", "right"))
                    for p in prompts
                ]
                blocked = len(self.flippable) - sum(self.flippable)
                print(f"flip augmentation on ({blocked} prompts opted out for left/right)")
            else:
                print(f"flip augmentation requested but {prompts_path} is missing — "
                      f"disabled (can't tell which prompts name a side)")
                self.flip = False

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, i):
        j = self.idx[i]
        before = torch.from_numpy(self.images[j, 0].copy()).float() / 127.5 - 1.0
        after = torch.from_numpy(self.images[j, 1].copy()).float() / 127.5 - 1.0
        text = torch.from_numpy(self.text[j].copy())
        if self.flip and self.flippable[j] and random.random() < 0.5:
            # Both images flipped together — the edit relationship has to survive.
            before = torch.flip(before, dims=[-1])
            after = torch.flip(after, dims=[-1])
        return before, after, text


class EMA:
    """Shadow copy of the weights, updated multiplicatively each step. Sampling
    from this instead of the live weights is standard for diffusion models —
    the live weights keep bouncing around the optimum, the average sits in it.
    """

    def __init__(self, model, decay):
        self.decay = decay
        self.shadow = {k: v.detach().clone().float()
                       for k, v in model.state_dict().items()}

    @torch.no_grad()
    def update(self, model):
        for k, v in model.state_dict().items():
            if v.dtype.is_floating_point:
                self.shadow[k].mul_(self.decay).add_(v.detach().float(), alpha=1 - self.decay)
            else:
                self.shadow[k].copy_(v.detach())

    def state_dict(self):
        return self.shadow

    def load_state_dict(self, sd):
        self.shadow = {k: v.clone().float() for k, v in sd.items()}


def load_null_text(seq_len, text_dim, device):
    """CLIP embedding of the empty string — the target that conditioning
    dropout replaces real prompts with, and the branch guidance interpolates
    from at sampling time. Computed here rather than baked into the dataset so
    the existing 60k Kaggle Dataset doesn't need regenerating; it's one forward
    pass through the already-frozen encoder.
    """
    from model.clip_encoder import ClipTextEncoder
    enc = ClipTextEncoder(device="cpu")
    if enc.seq_len != seq_len or enc.embed_dim != text_dim:
        raise SystemExit(
            f"null-text shape mismatch: encoder gives ({enc.seq_len}, {enc.embed_dim}), "
            f"dataset was built with ({seq_len}, {text_dim}). The dataset and "
            f"model/clip_encoder.py disagree — rebuild the text embeddings "
            f"(data/reencode_text.py) before training.")
    return enc.encode([""])[0].to(device)  # [seq_len, text_dim]


def min_snr_weights(schedule, t, gamma):
    """min(SNR_t, gamma) / SNR_t — the eps-prediction form of Min-SNR-gamma
    (Hang et al. 2023). SNR_t = acp_t / (1 - acp_t): huge at low noise, tiny at
    high noise. Unweighted MSE therefore spends most of its gradient on the
    high-noise steps where any near-mean guess scores well, which is exactly
    the "loss flattened at step 300 but quality kept climbing" mismatch seen
    here. Clipping the weight at gamma (5 is the paper's default) hands that
    budget back to the steps that decide image content.
    """
    acp = schedule.alphas_cumprod[t]
    snr = acp / (1.0 - acp)
    return (snr.clamp(max=gamma) / snr).view(-1, 1, 1, 1)


def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_ds = PairDataset(args.data, "train", flip=not args.no_flip)
    val_ds = PairDataset(args.data, "val")
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                               num_workers=2, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    model = UNet(text_dim=train_ds.text_dim).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"UNet params: {n_params/1e6:.1f}M")

    raw_model = model
    if torch.cuda.device_count() > 1 and not args.single_gpu:
        model = torch.nn.DataParallel(model)
        print(f"DataParallel across {torch.cuda.device_count()} GPUs")

    schedule = DiffusionSchedule(device=device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    null_text = None
    if args.text_dropout > 0:
        null_text = load_null_text(train_ds.seq_len, train_ds.text_dim, device)
        print(f"conditioning dropout: text {args.text_dropout}, image {args.image_dropout}")

    step = 0
    os.makedirs(args.out, exist_ok=True)
    ckpt_path = f"{args.out}/ckpt.pt"
    ema = EMA(raw_model, args.ema_decay) if args.ema_decay > 0 else None
    if args.resume and os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device)
        raw_model.load_state_dict(ckpt["model"])
        opt.load_state_dict(ckpt["opt"])
        step = ckpt["step"]
        if ema is not None:
            if "ema" in ckpt:
                ema.load_state_dict(ckpt["ema"])
            else:
                # Checkpoint from before EMA existed: seed the average from the
                # weights we just loaded rather than from a fresh init, so it
                # starts useful instead of needing thousands of steps to catch
                # up to where training already is.
                ema = EMA(raw_model, args.ema_decay)
                print("no EMA in checkpoint — seeded from the loaded weights")
        print(f"resumed from step {step}")

    def cycle(loader):
        while True:
            for batch in loader:
                yield batch

    data_iter = cycle(train_loader)
    t0 = time.time()

    while step < args.max_steps:
        opt.zero_grad()
        loss_accum = 0.0
        for _ in range(args.grad_accum):
            before, after, text_seq = next(data_iter)
            before, after, text_seq = before.to(device), after.to(device), text_seq.to(device)
            b = before.shape[0]

            if null_text is not None:
                # Independent masks, so all four combinations occur: full
                # conditioning, image-only, text-only, and neither. Guidance at
                # sampling time needs every one of those branches to be
                # something the model was actually trained on.
                drop_text = torch.rand(b, device=device) < args.text_dropout
                drop_image = torch.rand(b, device=device) < args.image_dropout
                text_seq = torch.where(drop_text[:, None, None], null_text.expand_as(text_seq),
                                       text_seq)
                before = torch.where(drop_image[:, None, None, None],
                                     torch.zeros_like(before), before)

            t = torch.randint(0, schedule.timesteps, (b,), device=device)
            noisy_after, noise = schedule.q_sample(after, t)
            pred = model(torch.cat([noisy_after, before], dim=1), t, text_seq)
            if args.min_snr_gamma > 0:
                w = min_snr_weights(schedule, t, args.min_snr_gamma)
                loss = (w * (pred - noise) ** 2).mean()
            else:
                loss = F.mse_loss(pred, noise)
            loss = loss / args.grad_accum
            loss.backward()
            loss_accum += loss.item()

        # Linear warmup, then cosine decay to --lr-min. A constant LR all the
        # way to a 70k-step ceiling leaves the last few percent of quality on
        # the table; decaying lets the weights settle instead of continuing to
        # bounce at full step size.
        #
        # The decay horizon is --lr-decay-steps, deliberately NOT --max-steps:
        # max-steps is used here as a ceiling that gets raised between sessions
        # whenever quality is still climbing (5 times so far). Tying the
        # schedule to it would make every raise jump the LR back up, which is
        # the opposite of a decay.
        if step < args.warmup:
            lr = args.lr * (step + 1) / args.warmup
        else:
            horizon = args.lr_decay_steps or args.max_steps
            progress = min(1.0, (step - args.warmup) / max(1, horizon - args.warmup))
            lr = args.lr_min + 0.5 * (args.lr - args.lr_min) * (1 + math.cos(math.pi * progress))
        for g in opt.param_groups:
            g["lr"] = lr
        opt.step()
        step += 1
        if ema is not None:
            ema.update(raw_model)

        if step % args.log_every == 0:
            print(f"step {step}/{args.max_steps}  loss {loss_accum:.4f}  "
                  f"{time.time()-t0:.0f}s elapsed", flush=True)

        if step % args.eval_every == 0:
            model.eval()
            vloss, n_batches = 0.0, 0
            # Fixed timestep draw: previously val loss re-randomised `t` every
            # evaluation, so consecutive numbers differed by more sampling noise
            # than actual progress. Same t's every time makes it comparable.
            gen = torch.Generator(device="cpu").manual_seed(1234)
            with torch.no_grad():
                for before, after, text_seq in val_loader:
                    before, after, text_seq = before.to(device), after.to(device), text_seq.to(device)
                    b = before.shape[0]
                    t = torch.randint(0, schedule.timesteps, (b,), generator=gen).to(device)
                    noisy_after, noise = schedule.q_sample(after, t)
                    pred = model(torch.cat([noisy_after, before], dim=1), t, text_seq)
                    vloss += F.mse_loss(pred, noise).item()
                    n_batches += 1
            print(f"  val loss {vloss / max(n_batches, 1):.4f}")
            model.train()

        if step % args.ckpt_every == 0 or step == args.max_steps:
            # Old (pre-1.6) pickle serialization, not torch's default zip
            # format: a zip-shaped ckpt.pt gets auto-exploded into its
            # internal data.pkl/data/N files the moment it's downloaded from
            # or uploaded to Kaggle as a Dataset, breaking the exact-filename
            # match every resume step depends on.
            blob = {"model": raw_model.state_dict(), "opt": opt.state_dict(), "step": step}
            if ema is not None:
                blob["ema"] = ema.state_dict()
            torch.save(blob, ckpt_path, _use_new_zipfile_serialization=False)

    print(f"done — {args.max_steps} steps in {(time.time()-t0)/3600:.1f}h")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True, help="prefix used by fetch_dataset.py's --out-prefix")
    p.add_argument("--out", default="./run")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--grad-accum", type=int, default=1)
    p.add_argument("--max-steps", type=int, default=8000)
    p.add_argument("--warmup", type=int, default=200)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--lr-min", type=float, default=1e-5,
                    help="floor of the post-warmup cosine decay")
    p.add_argument("--lr-decay-steps", type=int, default=0,
                    help="horizon of the cosine decay; 0 falls back to --max-steps. Set this "
                         "to the total training length you actually intend, so raising the "
                         "--max-steps ceiling between sessions doesn't reset the LR upward")
    p.add_argument("--text-dropout", type=float, default=0.05,
                    help="probability of replacing the prompt with null text, so "
                         "classifier-free guidance has a trained unconditional branch. "
                         "0 disables conditioning dropout entirely (and skips loading CLIP)")
    p.add_argument("--image-dropout", type=float, default=0.05,
                    help="probability of zeroing `before`; enables the image-guidance "
                         "scale in model/scheduler.py's sampler")
    p.add_argument("--ema-decay", type=float, default=0.9995,
                    help="EMA decay for the sampling weights; 0 disables EMA")
    p.add_argument("--min-snr-gamma", type=float, default=5.0,
                    help="Min-SNR-gamma loss weighting (paper default 5); 0 = plain MSE")
    p.add_argument("--no-flip", action="store_true",
                    help="disable horizontal-flip augmentation")
    p.add_argument("--log-every", type=int, default=20)
    p.add_argument("--eval-every", type=int, default=200)
    p.add_argument("--ckpt-every", type=int, default=200)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--single-gpu", action="store_true")
    main(p.parse_args())
