"""Kaggle GPU probe: real seconds-per-step on T4x2 for candidate model sizes.

Scaling up is a from-scratch retrain, so the size has to fit the weekly GPU quota
BEFORE committing to it — 30h/week is shared with G-Micro and G-Mini. Local MPS
timings are useless for this: forward time there went 150ms -> 176ms across a 4x
parameter increase, because the Mac is bandwidth-bound at this size while a T4 is
not. This measures the machine we will actually train on.

The 22.4M config is included as an ANCHOR. We know from three real sessions that
it runs at ~0.82 s/step on T4x2; if the probe reproduces that, the other numbers
can be trusted. (G-Mini used the same trick and it caught an interpolation that
would have crashed on the first step.)

Settings: GPU T4 x2, Internet ON. No dataset needed — synthetic batches, since
this measures compute, not data loading.
"""

import subprocess
import sys
import time

REPO = "https://github.com/JerzySukiennik/g-images.git"
BOOT = "/tmp/g-images-probe"
subprocess.run(["git", "clone", "--depth", "1", REPO, BOOT], check=True)
sys.path.insert(0, BOOT)

import torch
import torch.nn.functional as F
from model.unet import UNet
from model.scheduler import DiffusionSchedule
from data.edit_types import N_TYPES

BATCH, WARMUP, MEASURE = 32, 5, 25
CONFIGS = [
    ("anchor 22.4M", dict(base_channels=64, channel_mults=(1, 2, 4, 4))),
    ("64.8M", dict(base_channels=112, channel_mults=(1, 2, 4, 4))),
    ("70.5M", dict(base_channels=128, channel_mults=(1, 2, 3, 4))),
    ("83.8M", dict(base_channels=128, channel_mults=(1, 2, 4, 4))),
]

dev = "cuda"
print(f"GPUs: {torch.cuda.device_count()} x {torch.cuda.get_device_name(0)}\n")
print(f"{'config':<16}{'params':>10}{'s/step':>10}{'60k steps':>12}{'VRAM GB':>10}")

for name, kw in CONFIGS:
    try:
        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
        model = UNet(n_types=N_TYPES, **kw).to(dev)
        n = sum(p.numel() for p in model.parameters())
        if torch.cuda.device_count() > 1:
            model = torch.nn.DataParallel(model)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
        sched = DiffusionSchedule(device=dev)

        before = torch.randn(BATCH, 3, 128, 128, device=dev)
        after = torch.randn(BATCH, 3, 128, 128, device=dev)
        ids = torch.randint(0, N_TYPES, (BATCH,), device=dev)

        def step():
            opt.zero_grad()
            t = torch.randint(0, sched.timesteps, (BATCH,), device=dev)
            noisy, noise = sched.q_sample(after, t)
            target = sched.target(after, noise, t)
            pred = model(torch.cat([noisy, before], dim=1), t, ids)
            F.mse_loss(pred, target).backward()
            opt.step()

        for _ in range(WARMUP):
            step()
        torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(MEASURE):
            step()
        torch.cuda.synchronize()
        s = (time.time() - t0) / MEASURE
        vram = torch.cuda.max_memory_allocated() / 1e9
        print(f"{name:<16}{n/1e6:>8.1f}M{s:>10.2f}{s*60000/3600:>10.1f}h{vram:>10.1f}",
              flush=True)
        del model, opt
    except RuntimeError as e:
        print(f"{name:<16}  FAILED: {str(e)[:60]}", flush=True)
        torch.cuda.empty_cache()

print("\n30h/week quota is shared with G-Micro and G-Mini — anything whose 60k-step "
      "column approaches 30h consumes an entire week for one training run.")
