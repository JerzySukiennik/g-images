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
from data.edit_types import (NULL_TYPE, N_TYPES, SYNTHETIC_TYPES, TYPE_ID,
                              TYPE_NAMES, real_index_by_type)
from data.synthetic_edits import SYNTHETIC


class PairDataset(Dataset):
    """Mixes two supervision sources under one balanced type distribution.

    Synthetic types apply an exact function (data/synthetic_edits.py) to a real
    `before` image, so every one of the 60k images is a valid pair for every such
    type — effectively unlimited, perfectly consistent data. Real types read the
    scraped `after` image for pairs whose prompt named that transformation.

    Types are sampled UNIFORMLY, not in proportion to available data. Without
    that, `painting` (6290 real pairs) would outvote `sunset_sunrise` (585) more
    than tenfold, and the synthetic types — which can supply a pair for every
    image — would swamp everything. Uniform sampling is why the rare-but-coherent
    types are trainable at all.

    Training draws at random and ignores the index (the epoch is a formality when
    one source is unbounded); validation walks a fixed deterministic list so the
    number means the same thing every time it is printed.
    """

    def __init__(self, data_prefix, split="train", flip=False, val_pairs=8,
                 nominal_len=100_000):
        with open(f"{data_prefix}_meta.json") as f:
            meta = json.load(f)
        self.res = meta["res"]
        n, val_n = meta["n"], meta["val_n"]
        self.images = np.memmap(f"{data_prefix}_images.bin", dtype=np.uint8, mode="r",
                                 shape=(n, 2, 3, self.res, self.res))
        self.n = n
        self.split = split
        self.flip = flip and split == "train"

        with open(f"{data_prefix}_prompts.json") as f:
            prompts = json.load(f)
        by_type = real_index_by_type(prompts)

        # Hold the tail out of training entirely, for both sources.
        train_cut = n - val_n
        self.image_pool = (list(range(train_cut)) if split == "train"
                           else list(range(train_cut, n)))
        self.real_by_type = {
            t: [i for i in idxs if (i < train_cut) == (split == "train")]
            for t, idxs in by_type.items()
        }
        self.real_by_type = {t: v for t, v in self.real_by_type.items() if v}

        self.synth_ids = [TYPE_ID[name] for name in SYNTHETIC_TYPES]
        self.synth_fn = {TYPE_ID[name]: SYNTHETIC[name] for name in SYNTHETIC_TYPES}
        self.sampleable = self.synth_ids + sorted(self.real_by_type)

        if split == "train":
            counts = ", ".join(f"{TYPE_NAMES[t]}={len(v)}"
                               for t, v in sorted(self.real_by_type.items()))
            print(f"{len(self.synth_ids)} synthetic types (unlimited pairs each)")
            print(f"{len(self.real_by_type)} real types: {counts}")
            print(f"sampling types uniformly over {len(self.sampleable)} of them")

        # Deterministic validation list: every sampleable type, in order, repeated
        # until val_pairs*len(types) examples exist.
        self.val_items = []
        if split != "train":
            for k in range(val_pairs):
                for t in self.sampleable:
                    pool = self.real_by_type.get(t, self.image_pool)
                    self.val_items.append((t, pool[k % len(pool)]))

        self.nominal_len = nominal_len if split == "train" else len(self.val_items)

    def __len__(self):
        return self.nominal_len

    def _load(self, j, which):
        arr = self.images[j, which].copy()
        return torch.from_numpy(arr).float() / 127.5 - 1.0

    def _build(self, type_id, j):
        before = self._load(j, 0)
        if type_id in self.synth_fn:
            after = self.synth_fn[type_id](before)
        else:
            after = self._load(j, 1)
        return before, after

    def __getitem__(self, i):
        if self.split == "train":
            type_id = random.choice(self.sampleable)
            pool = self.real_by_type.get(type_id, self.image_pool)
            j = random.choice(pool)
        else:
            type_id, j = self.val_items[i]

        before, after = self._build(type_id, j)

        # Horizontal flip is safe for every type kept here: none of them name a
        # side, and the synthetic functions are all per-pixel or symmetric, so
        # flipping both images preserves the relationship exactly.
        if self.flip and random.random() < 0.5:
            before = torch.flip(before, dims=[-1])
            after = torch.flip(after, dims=[-1])
        return before, after, torch.tensor(type_id, dtype=torch.long)


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


def min_snr_weights(schedule, t, gamma, prediction="v"):
    """Min-SNR-gamma (Hang et al. 2023). SNR_t = acp_t / (1 - acp_t): huge at low
    noise, tiny at high noise. Unweighted MSE therefore lets the low-noise steps
    dominate the gradient, where the task is nearly trivial; clipping the weight
    at gamma (5 is the paper's default) hands that budget back to the steps that
    decide image content.

    The two parameterizations need different denominators — min(SNR,g)/SNR for
    epsilon, min(SNR,g)/(SNR+1) for v — because v-prediction already carries an
    implicit SNR-dependent weighting of its own. Using the epsilon form with
    v-prediction double-counts it.

    (A note against a mistake made once while reading this: min-SNR does NOT
    down-weight the high-noise end. At t=999, SNR is ~4e-5, far below gamma, so
    min(SNR,gamma)/SNR = 1 — full weight. It is the low-noise steps that get
    suppressed.)
    """
    acp = schedule.alphas_cumprod[t]
    snr = acp / (1.0 - acp)
    if prediction == "eps":
        w = snr.clamp(max=gamma) / snr
    else:
        w = snr.clamp(max=gamma) / (snr + 1.0)
    return w.view(-1, 1, 1, 1)


def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_ds = PairDataset(args.data, "train", flip=not args.no_flip)
    val_ds = PairDataset(args.data, "val")
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                               num_workers=2, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    model = UNet(n_types=N_TYPES).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"UNet params: {n_params/1e6:.1f}M  ({N_TYPES} edit types)")

    raw_model = model
    if torch.cuda.device_count() > 1 and not args.single_gpu:
        model = torch.nn.DataParallel(model)
        print(f"DataParallel across {torch.cuda.device_count()} GPUs")

    schedule = DiffusionSchedule(schedule=args.noise_schedule,
                                  prediction=args.prediction, device=device)
    print(f"schedule: {args.noise_schedule}, predicting: {args.prediction}")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    if args.text_dropout > 0:
        print(f"conditioning dropout: type -> null {args.text_dropout}, "
              f"image -> zeros {args.image_dropout}")

    step = 0
    os.makedirs(args.out, exist_ok=True)
    ckpt_path = f"{args.out}/ckpt.pt"
    ema = EMA(raw_model, args.ema_decay) if args.ema_decay > 0 else None
    if args.resume and os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device)
        # A checkpoint from a previous architecture generation resumes into a
        # wall of shape errors that says nothing useful. v3 changed the
        # conditioning, the depth and the parameterization all at once, and the
        # v2 checkpoints (up to step 94200) are still lying around in Kaggle
        # datasets, so name the problem explicitly.
        want = raw_model.state_dict()
        got = dict(ckpt["model"])

        # Growing the taxonomy changes only the embedding table's row count.
        # Every other weight is untouched, so copying the old rows into the
        # larger table carries over everything the model already learned — worth
        # doing, since the filter types took ~40000 steps to get right and the
        # new rows are additions, not replacements. Relies on new types being
        # APPENDED to TYPE_NAMES: reordering would silently attach learned
        # behaviour to the wrong type, so verify that before trusting this path.
        key = "type_tokens.emb.weight"
        if key in got and key in want and got[key].shape != want[key].shape:
            old_rows, new_rows = got[key].shape[0], want[key].shape[0]
            if got[key].shape[1:] == want[key].shape[1:] and new_rows > old_rows:
                grown = want[key].clone()
                grown[:old_rows] = got[key]
                got[key] = grown
                print(f"grew the edit-type table {old_rows} -> {new_rows} rows, "
                      f"keeping the learned ones (new rows start from init)")

        bad = [k for k in want if k not in got or want[k].shape != got[k].shape]
        if bad:
            raise SystemExit(
                f"checkpoint at {ckpt_path} (step {ckpt.get('step')}) does not match "
                f"this model: {len(bad)} parameters differ or are missing, first few "
                f"{bad[:3]}. It is almost certainly from an earlier architecture — "
                f"train from scratch instead of resuming, or detach the stale "
                f"checkpoint dataset from the kernel's inputs.")
        raw_model.load_state_dict(got)
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
            before, after, type_ids = next(data_iter)
            before, after, type_ids = before.to(device), after.to(device), type_ids.to(device)
            b = before.shape[0]

            if args.text_dropout > 0:
                # Independent masks, so all four combinations occur: full
                # conditioning, image-only, type-only, and neither. Guidance at
                # sampling time needs every one of those branches to be something
                # the model was actually trained on.
                drop_type = torch.rand(b, device=device) < args.text_dropout
                drop_image = torch.rand(b, device=device) < args.image_dropout
                type_ids = torch.where(drop_type,
                                       torch.full_like(type_ids, NULL_TYPE), type_ids)
                before = torch.where(drop_image[:, None, None, None],
                                     torch.zeros_like(before), before)

            t = torch.randint(0, schedule.timesteps, (b,), device=device)
            noisy_after, noise = schedule.q_sample(after, t)
            target = schedule.target(after, noise, t)
            pred = model(torch.cat([noisy_after, before], dim=1), t, type_ids)
            if args.min_snr_gamma > 0:
                w = min_snr_weights(schedule, t, args.min_snr_gamma,
                                     prediction=args.prediction)
                loss = (w * (pred - target) ** 2).mean()
            else:
                loss = F.mse_loss(pred, target)
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
                for before, after, type_ids in val_loader:
                    before, after, type_ids = (before.to(device), after.to(device),
                                                type_ids.to(device))
                    b = before.shape[0]
                    t = torch.randint(0, schedule.timesteps, (b,), generator=gen).to(device)
                    noisy_after, noise = schedule.q_sample(after, t)
                    target = schedule.target(after, noise, t)
                    pred = model(torch.cat([noisy_after, before], dim=1), t, type_ids)
                    vloss += F.mse_loss(pred, target).item()
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
    p.add_argument("--noise-schedule", choices=("cosine", "linear"), default="cosine")
    p.add_argument("--prediction", choices=("v", "eps"), default="v")
    p.add_argument("--text-dropout", type=float, default=0.1,
                    help="probability of replacing the edit type with the null type, so "
                         "classifier-free guidance has a trained unconditional branch. "
                         "Runs from step 0 here — added late to the previous model, it "
                         "never caught up (see kaggle/02-train.py's history)")
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
