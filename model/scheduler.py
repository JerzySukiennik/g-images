"""Gaussian diffusion schedule: linear beta_t noise schedule (DDPM) with a
DDIM sampler for fast few-step inference. Hand-rolled rather than imported —
same reasoning as MicroG: understanding every layer matters as much as a
working model (see SPEC.md).
"""

import torch


def _quantile(flat, q):
    """Per-row quantile of a [B, N] tensor. torch.quantile isn't implemented
    for every MPS build and silently differs across versions, so this uses
    sort+index, which behaves identically everywhere this runs (Mac MPS/CPU
    for the local checks, CUDA on Kaggle).
    """
    n = flat.shape[1]
    k = max(0, min(n - 1, int(round(q * (n - 1)))))
    return flat.sort(dim=1).values[:, k]


class DiffusionSchedule:
    def __init__(self, timesteps=1000, beta_start=1e-4, beta_end=2e-2, device="cpu"):
        self.timesteps = timesteps
        betas = torch.linspace(beta_start, beta_end, timesteps, device=device)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        self.betas = betas
        self.alphas_cumprod = alphas_cumprod
        self.sqrt_alphas_cumprod = alphas_cumprod.sqrt()
        self.sqrt_one_minus_alphas_cumprod = (1.0 - alphas_cumprod).sqrt()

    def q_sample(self, x0, t, noise=None):
        """Forward process: x_t = sqrt(acp_t) * x0 + sqrt(1 - acp_t) * noise."""
        if noise is None:
            noise = torch.randn_like(x0)
        sqrt_acp = self.sqrt_alphas_cumprod[t].view(-1, 1, 1, 1)
        sqrt_1macp = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1)
        return sqrt_acp * x0 + sqrt_1macp * noise, noise

    @torch.no_grad()
    def ddim_sample(self, model, before, text_emb, steps=100, device="cpu",
                    text_uncond=None, guidance=1.0, image_guidance=1.0,
                    guidance_rescale=0.0, dynamic_threshold=0.0,
                    generator=None):
        """Deterministic (eta=0) DDIM sampler, `steps` << self.timesteps.

        `steps` trades quality for the seconds-per-edit budget on a CPU/MPS
        Mac (SPEC.md #5 leaves the exact count open until real inference is
        timed). Measured 2026-07-22 on the cross-attention checkpoint: 20
        steps gave a washed-out result that looked nearly identical
        regardless of prompt; 100 steps on the SAME checkpoint revealed real
        structure and color the 20-step version was hiding. Don't judge
        output quality below ~50.

        Classifier-free guidance (added 2026-07-27). Sampling the model's raw
        prediction (guidance=1.0, everything before this change) is why edits
        barely showed up: with `before` concatenated on the channel axis, just
        copying the input is always the cheapest way to lower the MSE, so the
        text signal stays weak unless it's explicitly amplified at sampling
        time. InstructPix2Pix uses the two-scale form implemented here —

            e = e_uncond
                + image_guidance * (e_image - e_uncond)
                + guidance       * (e_full  - e_image)

        where e_full conditions on both `before` and the prompt, e_image drops
        the prompt (null text), and e_uncond drops both (`before` zeroed too).
        IP2P's own defaults are guidance~7.5 / image_guidance~1.5.

        `text_uncond` is the null-text embedding (encode [""]); pass it to turn
        guidance on. Until train/train.py trains with conditioning dropout the
        null branches are extrapolation — the empty prompt is still a valid
        CLIP embedding, but a zeroed `before` is off-distribution, so a plain
        text-only run (image_guidance=1.0) is the trustworthy measurement.
        """
        b = before.shape[0]
        seq = torch.linspace(self.timesteps - 1, 0, steps, dtype=torch.long, device=device)
        x = torch.randn(before.shape, device=device, dtype=before.dtype, generator=generator)
        acp = self.alphas_cumprod.to(device)
        do_cfg = text_uncond is not None and (guidance != 1.0 or image_guidance != 1.0)
        zeros_before = torch.zeros_like(before) if do_cfg and image_guidance != 1.0 else None
        for i, t in enumerate(seq):
            t_batch = t.repeat(b)
            if do_cfg:
                e_full = model(torch.cat([x, before], dim=1), t_batch, text_emb)
                e_image = model(torch.cat([x, before], dim=1), t_batch, text_uncond)
                if zeros_before is None:
                    # Text-only guidance: no unconditional (zeroed-image)
                    # branch, so the formula collapses to the familiar
                    # single-scale one and costs 2 passes instead of 3.
                    pred_noise = e_image + guidance * (e_full - e_image)
                else:
                    e_uncond = model(torch.cat([x, zeros_before], dim=1), t_batch, text_uncond)
                    pred_noise = (e_uncond
                                  + image_guidance * (e_image - e_uncond)
                                  + guidance * (e_full - e_image))
                if guidance_rescale > 0:
                    # Guidance rescale (Lin et al. 2024). Amplifying the
                    # conditional difference also inflates the magnitude of the
                    # predicted noise; the reconstructed x0 then overshoots
                    # [-1,1] and the clamp below flattens whole regions to
                    # channel extremes — the flat psychedelic patches seen at
                    # scale >= 5 on the step-70000 checkpoint. Scaling the
                    # guided prediction back to the conditional branch's own
                    # standard deviation removes the overshoot without
                    # weakening the direction of the edit.
                    std_cond = e_full.std(dim=(1, 2, 3), keepdim=True)
                    std_cfg = pred_noise.std(dim=(1, 2, 3), keepdim=True).clamp(min=1e-6)
                    pred_noise = (guidance_rescale * (pred_noise * std_cond / std_cfg)
                                  + (1 - guidance_rescale) * pred_noise)
            else:
                pred_noise = model(torch.cat([x, before], dim=1), t_batch, text_emb)
            a_t = acp[t]
            x0_pred = (x - (1 - a_t).sqrt() * pred_noise) / a_t.sqrt()
            if dynamic_threshold > 0:
                # Imagen's dynamic thresholding: instead of hard-clipping the
                # overshoot away (which destroys contrast by pinning pixels to
                # +-1), rescale the whole image by its own high percentile so
                # the outliers come back into range and relative structure
                # survives.
                flat = x0_pred.abs().reshape(b, -1)
                s = _quantile(flat, dynamic_threshold).clamp(min=1.0).view(-1, 1, 1, 1)
                x0_pred = x0_pred.clamp(-s, s) / s
            else:
                x0_pred = x0_pred.clamp(-1, 1)
            if i == len(seq) - 1:
                x = x0_pred
                break
            a_prev = acp[seq[i + 1]]
            x = a_prev.sqrt() * x0_pred + (1 - a_prev).sqrt() * pred_noise
        return x
