"""G-Images training loop — diffusion noise-prediction loss on (before, after,
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
                              TYPE_NAMES, classify_anyedit)
from data.synthetic_edits import SYNTHETIC


def grow_to(old, target):
    """Pad `old` with zeros so it matches `target`'s shape, keeping old values in
    the leading rows. Returns `old` unchanged when the shapes already agree.

    Growing the edit-type taxonomy changes one parameter's row count, and EVERY
    structure holding a per-parameter tensor has to follow. That is three places,
    and this project found them one crash at a time: the model weights, AdamW's
    two momentum buffers, and the EMA shadow copy. Hence one helper used by all
    three rather than three ad-hoc patches — the next satellite structure added
    should use it too.
    """
    if old.shape == target.shape:
        return old
    grown = torch.zeros_like(target)
    grown[:old.shape[0]] = old.to(grown.dtype)
    return grown


class PairDataset(Dataset):
    """Mixes synthetic and AnyEdit supervision across several prep shards.

    Data now arrives as several independent prep-notebook outputs (five, for
    400000 pairs), each with its own images binary and sidecars, because one
    Kaggle notebook may not write 40 GB. They are concatenated logically here:
    an item is (shard, row), and nothing downstream needs to know.

    Two supervision sources, as before:
      - synthetic types apply an exact function (data/synthetic_edits.py) to any
        `before` image from any shard, so their data is unlimited and perfect;
      - AnyEdit types read the scraped `after` image for rows whose (edit_type,
        instruction) pair names one transformation, per classify_anyedit.

    Types are sampled UNIFORMLY. With `add_person` at 9764 pairs and `add_hat` at
    1849, proportional sampling would bury the rarer objects five to one; uniform
    sampling is why rare-but-coherent types are learnable at all, and it is what
    made the filters work.
    """

    def __init__(self, data_prefixes, split="train", flip=False, val_pairs=4,
                 nominal_len=100_000, synthetic_share=0.25):
        self.synthetic_share = synthetic_share
        self.split = split
        self.flip = flip and split == "train"
        self.shards = []
        self.by_type = {}          # type_id -> [(shard, row)]
        self.image_pool = []       # (shard, row) usable as a synthetic source

        for si, prefix in enumerate(data_prefixes):
            with open(f"{prefix}_meta.json") as f:
                meta = json.load(f)
            n, val_n, res = meta["n"], meta.get("val_n", 0), meta["res"]
            images = np.memmap(f"{prefix}_images.bin", dtype=np.uint8, mode="r",
                                shape=(n, 2, 3, res, res))
            self.shards.append(images)
            self.res = res

            with open(f"{prefix}_prompts.json") as f:
                prompts = json.load(f)
            types_path = f"{prefix}_types.json"
            # Shards built by data/fetch_anyedit.py carry the corpus's own
            # edit_type; the older IP2P prep had none, and those rows can still
            # serve as synthetic sources even though no semantic type claims them.
            raw_types = json.load(open(types_path)) if os.path.exists(types_path) else None

            train_cut = n - val_n
            for i in range(n):
                in_train = i < train_cut
                if in_train != (split == "train"):
                    continue
                self.image_pool.append((si, i))
                if raw_types is None:
                    continue
                t = classify_anyedit(raw_types[i], prompts[i] if i < len(prompts) else "")
                if t is not None:
                    self.by_type.setdefault(t, []).append((si, i))

        self.synth_ids = [TYPE_ID[name] for name in SYNTHETIC_TYPES]
        self.synth_fn = {TYPE_ID[name]: SYNTHETIC[name] for name in SYNTHETIC_TYPES}
        # A type with too few rows cannot teach a transformation and would just be
        # memorized, so it is excluded rather than trained badly.
        self.by_type = {t: v for t, v in self.by_type.items() if len(v) >= 50}
        self.sampleable = self.synth_ids + sorted(self.by_type)

        if split == "train":
            print(f"{len(self.shards)} shards, {len(self.image_pool)} images in split")
            print(f"{len(self.synth_ids)} synthetic types (unlimited pairs each)")
            print(f"{len(self.by_type)} AnyEdit types; synthetic share "
                  f"{self.synthetic_share:.0%}, rest spread over AnyEdit types")
            rare = sorted(self.by_type.items(), key=lambda kv: len(kv[1]))[:5]
            print("smallest: " + ", ".join(f"{TYPE_NAMES[t]}={len(v)}" for t, v in rare))

        self.val_items = []
        if split != "train":
            for k in range(val_pairs):
                for t in self.sampleable:
                    pool = self.by_type.get(t) or self.image_pool
                    if pool:
                        self.val_items.append((t, pool[k % len(pool)]))
        self.nominal_len = nominal_len if split == "train" else len(self.val_items)

    def __len__(self):
        return self.nominal_len

    def _load(self, ref, which):
        si, i = ref
        return torch.from_numpy(self.shards[si][i, which].copy()).float() / 127.5 - 1.0

    def __getitem__(self, i):
        if self.split == "train":
            # Sampling is no longer uniform over all types. By step 40000 the 13
            # synthetic filters match their exact ground truth pixel for pixel,
            # while object types are the weak point — so spending 13/53 = 25% of
            # every batch re-teaching solved transformations is waste. The
            # synthetic share is capped at --synthetic-share and the remainder is
            # spread over the AnyEdit types.
            #
            # It is not dropped to zero: these types are the only ones with an
            # exact ground truth, so they are also the project's regression test.
            # Losing them would remove the one signal that tells us the model has
            # not drifted.
            if self.by_type and random.random() >= self.synthetic_share:
                type_id = random.choice(sorted(self.by_type))
            else:
                type_id = random.choice(self.synth_ids)
            pool = self.by_type.get(type_id) or self.image_pool
            ref = random.choice(pool)
        else:
            type_id, ref = self.val_items[i]

        before = self._load(ref, 0)
        after = (self.synth_fn[type_id](before) if type_id in self.synth_fn
                 else self._load(ref, 1))

        # Safe for every surviving type: none names a side, and the synthetic
        # functions are per-pixel or symmetric.
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

    def adapt_to(self, model):
        """Resize any shadow entry whose parameter changed shape — otherwise the
        first update() dies on a 24-vs-64 row mismatch, which is exactly how the
        first attempt at resuming into a larger taxonomy failed."""
        for k, v in model.state_dict().items():
            cur = self.shadow.get(k)
            if cur is None:
                self.shadow[k] = v.detach().clone().float()
            elif cur.shape != v.shape:
                self.shadow[k] = grow_to(cur, v.detach().float())
                print(f"grew EMA {k} {tuple(cur.shape)} -> {tuple(v.shape)}")


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
    train_ds = PairDataset(args.data, "train", flip=not args.no_flip,
                            synthetic_share=args.synthetic_share)
    val_ds = PairDataset(args.data, "val")
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                               num_workers=2, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    arch = dict(n_types=N_TYPES, base_channels=args.base_channels,
                channel_mults=tuple(args.channel_mults))
    model = UNet(**arch).to(device)
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
    # Mixed precision. The T4s have tensor cores that fp32 training never touches:
    # measured on this exact model, AMP took 1.63 s/step down to 0.96 and peak VRAM
    # from 12.7 GB to 10.7. That memory is what decides whether a bigger model fits
    # at all — every config above 70.5M OOM'd in fp32 and 98.3M fits comfortably
    # with AMP. Parameters stay fp32 (autocast only casts operations), so EMA and
    # the optimizer are unaffected.
    use_amp = args.amp and device == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    if use_amp:
        print("mixed precision: on")

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
                got[key] = grow_to(got[key], want[key])
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
        if "opt" in ckpt:
            opt.load_state_dict(ckpt["opt"])
        else:
            # A checkpoint salvaged from a run that was killed mid-save can hold
            # complete weights and nothing else — torch writes the sections in
            # order, so 'model' lands whole before 'opt' is even started. Fresh
            # AdamW moments cost a short transient at resume; refusing to start
            # would cost every step that produced these weights.
            print("no optimizer state in checkpoint — AdamW moments start at rest")

        # Growing the embedding table is only half the job: AdamW keeps two
        # momentum tensors PER PARAMETER, and the ones just restored are still
        # shaped for the old row count. Without this the first opt.step() dies
        # inside _multi_tensor_adam on a size mismatch — which is exactly how the
        # first v4 run failed, after everything else had already gone right.
        # Padding with zeros rather than dropping the state keeps the momentum
        # learned for the existing types; new rows simply start at rest.
        emb = raw_model.type_tokens.emb.weight
        st = opt.state.get(emb)
        if st is not None:
            for key in ("exp_avg", "exp_avg_sq"):
                v = st.get(key)
                if v is not None and v.shape != emb.shape:
                    st[key] = grow_to(v, emb)
                    print(f"grew optimizer {key} {tuple(v.shape)} -> {tuple(emb.shape)}")
        step = ckpt["step"]
        if ema is not None:
            if "ema" in ckpt:
                ema.load_state_dict(ckpt["ema"])
                ema.adapt_to(raw_model)
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
            with torch.cuda.amp.autocast(enabled=use_amp):
                pred = model(torch.cat([noisy_after, before], dim=1), t, type_ids)
            # Loss in fp32 even under autocast: the squared error of small
            # residuals is exactly where fp16 loses the digits that matter.
            pred = pred.float()
            if args.min_snr_gamma > 0:
                w = min_snr_weights(schedule, t, args.min_snr_gamma,
                                     prediction=args.prediction)
                loss = (w * (pred - target) ** 2).mean()
            else:
                loss = F.mse_loss(pred, target)
            loss = loss / args.grad_accum
            scaler.scale(loss).backward()
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
        scaler.step(opt)
        scaler.update()
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
                    with torch.cuda.amp.autocast(enabled=use_amp):
                        pred = model(torch.cat([noisy_after, before], dim=1), t, type_ids)
                    vloss += F.mse_loss(pred.float(), target).item()
                    n_batches += 1
            print(f"  val loss {vloss / max(n_batches, 1):.4f}")
            model.train()

        if step % args.ckpt_every == 0 or step == args.max_steps:
            # Old (pre-1.6) pickle serialization, not torch's default zip
            # format: a zip-shaped ckpt.pt gets auto-exploded into its
            # internal data.pkl/data/N files the moment it's downloaded from
            # or uploaded to Kaggle as a Dataset, breaking the exact-filename
            # match every resume step depends on.
            # The architecture travels WITH the weights. Two model sizes now exist
            # (70.5M and 98.3M) and an evaluation script that guesses the width
            # simply fails to load — or worse, a future size guesses wrong silently.
            blob = {"model": raw_model.state_dict(), "opt": opt.state_dict(),
                    "step": step, "arch": arch}
            if ema is not None:
                blob["ema"] = ema.state_dict()

            # Write to a temp file and rename, rather than saving over ckpt.pt.
            # Saving in place means the only copy is a half-written file for the
            # ~20s the save takes, and on 2026-08-09 a Kaggle session was killed
            # inside exactly that window: 8h11m of training, 412 MB of a 1.57 GB
            # file, and nothing recoverable. Not "most of the weights" — torch
            # orders storages on disk by key, not by section, so the surviving
            # 531 storages were all optimizer moments and not one of the 321
            # weight tensors. rename() is atomic, so ckpt.pt is always either the
            # previous complete checkpoint or the new one, never a stump.
            tmp_path = ckpt_path + ".tmp"
            torch.save(blob, tmp_path, _use_new_zipfile_serialization=False)
            os.replace(tmp_path, ckpt_path)

    print(f"done — {args.max_steps} steps in {(time.time()-t0)/3600:.1f}h")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True, nargs="+",
                    help="one or more dataset prefixes; several prep shards are "
                         "concatenated logically (one notebook cannot write 40 GB)")
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
    p.add_argument("--min-snr-gamma", type=float, default=0.0,
                    help="Min-SNR-gamma loss weighting; 0 = plain MSE, which is the right "
                         "choice with v-prediction. MEASURED 2026-08-06: with gamma=5 and the "
                         "cosine schedule, the v-form weight min(SNR,g)/(SNR+1) is 3.75e-33 at "
                         "t=999 — the timestep DDIM starts from. The step-30000 model had 26% "
                         "relative error there against 1-2% everywhere else, so the first "
                         "sampling step (which sets global structure) was computed from a bad "
                         "prediction and guidance amplified it into noise. Min-SNR was designed "
                         "for epsilon-prediction; v-prediction already equalizes difficulty "
                         "across timesteps, and stacking them suppresses high noise twice.")
    p.add_argument("--synthetic-share", type=float, default=0.25,
                    help="fraction of each batch spent on the 13 exact-function filter "
                         "types. They are solved by step 40000, so later runs lower this "
                         "to redirect capacity at objects; kept above zero because they "
                         "are the only types with a ground truth to regression-test against")
    p.add_argument("--base-channels", type=int, default=128,
                    help="UNet width. 128 -> 70.5M, 152 -> 98.3M. Must be divisible by 8 "
                         "(GroupNorm groups). Anything above ~168 OOMs on a T4 even with AMP")
    p.add_argument("--channel-mults", type=int, nargs="+", default=[1, 2, 3, 4])
    p.add_argument("--amp", type=int, default=1,
                    help="mixed precision (1/0). Measured on T4x2: 1.63 -> 0.96 s/step "
                         "and 12.7 -> 10.7 GB peak. Off only for debugging a numerical "
                         "problem you suspect fp16 caused")
    p.add_argument("--no-flip", action="store_true",
                    help="disable horizontal-flip augmentation")
    p.add_argument("--log-every", type=int, default=20)
    p.add_argument("--eval-every", type=int, default=200)
    p.add_argument("--ckpt-every", type=int, default=200)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--single-gpu", action="store_true")
    main(p.parse_args())
