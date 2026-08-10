"""Convolutional autoencoder — the compressor that makes G-Image 2.2 affordable.

Everything before 2.2 ran diffusion directly on pixels. At 128x128 that is 16384
points per channel, and the measured cost was 1314 ms per forward pass for the
98.3M model on CPU. Going to 256x256 in pixels would have quadrupled that. The
whole reason Stable Diffusion can synthesize objects at all is that it does not
do this: it trains a separate model to compress the image first, then runs
diffusion on the compressed codes.

Here: 256x256x3 pixels -> 32x32x4 latent. That is 8x smaller per side, 48x fewer
numbers, and 16x fewer spatial positions than the 128x128 pixel grid 2.1 works
on. So 2.2 gets HIGHER resolution and a CHEAPER diffusion model at the same time,
which is not a trade-off anyone gets to make by tuning width.

Trained from scratch like everything else here. In particular there is no VGG
perceptual loss, which is the standard ingredient at this step — it would import
pretrained ImageNet weights and break the one rule this project has held since
the beginning. L1 plus a light KL term instead; if reconstructions come out
soft, the next lever is a small patch discriminator trained alongside, also from
scratch.

The KL term is deliberately tiny (~1e-6). Its job is not to build a generative
VAE — the diffusion model is the generative part — but to stop the encoder from
answering with arbitrarily large values, which would leave the latent space full
of holes that diffusion trajectories then wander into.

One number here matters more than it looks: `scale`. Diffusion assumes its input
has roughly unit variance, and raw latents do not — Stable Diffusion carries a
hardcoded 0.18215 for exactly this. Ours is measured after training by
train/train_ae.py and stored in the checkpoint. Getting it wrong does not crash
anything; it quietly puts the noise schedule at the wrong scale, which reads as
"the diffusion model just does not learn well" and is nearly impossible to
diagnose from the loss curve.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    """GroupNorm-SiLU-conv twice, plus a skip. No time conditioning: unlike the
    U-Net, the autoencoder sees clean images only and has no notion of a step."""

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.norm2 = nn.GroupNorm(8, out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x):
        h = self.conv1(F.silu(self.norm1(x)))
        h = self.conv2(F.silu(self.norm2(h)))
        return h + self.skip(x)


class AttnBlock(nn.Module):
    """Self-attention at the bottleneck only. At 32x32 that is 1024 positions —
    affordable. Applying it at 256x256 would be 65536 positions and quadratic."""

    def __init__(self, ch):
        super().__init__()
        self.norm = nn.GroupNorm(8, ch)
        self.qkv = nn.Conv2d(ch, ch * 3, 1)
        self.proj = nn.Conv2d(ch, ch, 1)
        self.ch = ch

    def forward(self, x):
        b, c, h, w = x.shape
        q, k, v = self.qkv(self.norm(x)).reshape(b, 3, c, h * w).unbind(1)
        att = torch.softmax(q.transpose(1, 2) @ k / (c ** 0.5), dim=-1)
        out = (v @ att.transpose(1, 2)).reshape(b, c, h, w)
        return x + self.proj(out)


class Encoder(nn.Module):
    def __init__(self, base=64, mults=(1, 2, 4), latent_ch=4):
        super().__init__()
        self.conv_in = nn.Conv2d(3, base, 3, padding=1)
        chs, ch = [], base
        blocks = []
        for i, m in enumerate(mults):
            out = base * m
            blocks += [ResBlock(ch, out), ResBlock(out, out)]
            ch = out
            if i < len(mults) - 1 or True:      # three downsamples: 256 -> 32
                blocks.append(nn.Conv2d(ch, ch, 3, stride=2, padding=1))
            chs.append(ch)
        self.blocks = nn.Sequential(*blocks)
        self.mid = nn.Sequential(ResBlock(ch, ch), AttnBlock(ch), ResBlock(ch, ch))
        self.norm_out = nn.GroupNorm(8, ch)
        # Two latent channels' worth of parameters per channel: mean and logvar.
        self.conv_out = nn.Conv2d(ch, latent_ch * 2, 3, padding=1)

    def forward(self, x):
        h = self.mid(self.blocks(self.conv_in(x)))
        mean, logvar = self.conv_out(F.silu(self.norm_out(h))).chunk(2, dim=1)
        return mean, logvar.clamp(-30, 20)


class Decoder(nn.Module):
    def __init__(self, base=64, mults=(1, 2, 4), latent_ch=4):
        super().__init__()
        ch = base * mults[-1]
        self.conv_in = nn.Conv2d(latent_ch, ch, 3, padding=1)
        self.mid = nn.Sequential(ResBlock(ch, ch), AttnBlock(ch), ResBlock(ch, ch))
        blocks = []
        for m in reversed(mults):
            out = base * m
            blocks += [ResBlock(ch, out), ResBlock(out, out),
                       nn.Upsample(scale_factor=2, mode="nearest"),
                       nn.Conv2d(out, out, 3, padding=1)]
            ch = out
        self.blocks = nn.Sequential(*blocks)
        self.norm_out = nn.GroupNorm(8, ch)
        self.conv_out = nn.Conv2d(ch, 3, 3, padding=1)

    def forward(self, z):
        h = self.blocks(self.mid(self.conv_in(z)))
        return self.conv_out(F.silu(self.norm_out(h)))


class Autoencoder(nn.Module):
    """256x256x3 <-> 32x32x4.

    `scale` multiplies the latent on encode and divides on decode, so that what
    the diffusion model sees has roughly unit variance. It is 1.0 until
    train/train_ae.py measures the real value over a batch of data and writes it
    into the checkpoint.
    """

    def __init__(self, base=64, mults=(1, 2, 4), latent_ch=4, scale=1.0):
        super().__init__()
        self.encoder = Encoder(base, mults, latent_ch)
        self.decoder = Decoder(base, mults, latent_ch)
        self.latent_ch = latent_ch
        self.register_buffer("scale", torch.tensor(float(scale)))
        self.arch = dict(base=base, mults=tuple(mults), latent_ch=latent_ch)

    def encode(self, x, sample=True):
        """x in [-1, 1] -> scaled latent. Sampling during training, mean at
        inference: a fixed encoding is what the diffusion model should see."""
        mean, logvar = self.encoder(x)
        z = mean + torch.randn_like(mean) * (0.5 * logvar).exp() if sample else mean
        return z * self.scale, mean, logvar

    def decode(self, z):
        return self.decoder(z / self.scale)

    def forward(self, x):
        z, mean, logvar = self.encode(x)
        return self.decode(z), mean, logvar


def kl_loss(mean, logvar):
    """Per-element KL against a unit Gaussian, averaged. Kept tiny by its weight
    in the training loop — see the module docstring for why it is here at all."""
    return 0.5 * (mean.pow(2) + logvar.exp() - 1.0 - logvar).mean()
