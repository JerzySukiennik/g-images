"""Small conditional U-Net — the part of G-Images that IS trained from scratch.
Predicts the target added to `after`, conditioned on `before` (concatenated on
the channel axis — same img2img formulation InstructPix2Pix itself uses) and on
a discrete edit type, whose learned token sequence feeds cross-attention.

Architecture history
--------------------
v1 used FiLM: a global scale/shift from one pooled CLIP text vector. It handled
whole-image style/colour edits but could not localize object edits ("add a hat"
gave a global tone shift, not a hat), because a pooled vector carries no
per-word or spatial information.

v2 replaced that with cross-attention over the full per-token CLIP sequence, so
each spatial position could attend to the specific word driving it. Trained to
step 94200. Then measurement killed it: runtime/guidance_probe.py showed
cos(e_cond, e_null) = 1.000 at every timestep, with the conditional/unconditional
difference at ~1-2% of signal magnitude. The model had learned to almost entirely
ignore the text, so classifier-free guidance had nothing to amplify — raising its
scale produced only saturated colour noise, and raising conditioning dropout from
5% to 20% (a 4.5x increase in null-text exposure) changed nothing.

v3 (this file) keeps the cross-attention machinery untouched and swaps out what
it attends to: each edit type owns N_TOKENS learned vectors (TypeTokens below)
instead of a CLIP encoding of a sentence. The model no longer has to learn what
English means — a hopeless task from 60k pairs, which is why InstructPix2Pix
fine-tunes Stable Diffusion rather than training from scratch — and instead
learns a fixed set of named transformations, each backed by thousands of
consistent examples. Conditioning becomes unambiguous, and spatial routing
survives (a return to FiLM would have thrown it away again).

Timestep conditioning stays FiLM throughout: "how much noise" genuinely is a
whole-image quantity, unlike the edit itself.

Also v3: channel_mults gains a fourth level, taking the bottleneck from 32x32
down to 16x16. At 128px input, three levels left the deepest features with a
receptive field too small to reason about scene-scale structure, with only one
self-attention layer to compensate.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def timestep_embedding(t, dim):
    """Sinusoidal embedding, same construction as the original DDPM paper."""
    half = dim // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device).float() / half)
    args = t.float()[:, None] * freqs[None]
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        emb = F.pad(emb, (0, 1))
    return emb


class TimeResBlock(nn.Module):
    """GroupNorm -> SiLU -> Conv, twice, with FiLM (scale, shift) from the
    timestep embedding applied after the first norm — this one genuinely is
    a whole-image quantity ("how much noise to remove"), unlike text. 1x1
    conv on the residual path when the channel count changes.
    """

    def __init__(self, in_ch, out_ch, time_dim):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.film = nn.Linear(time_dim, out_ch * 2)
        self.norm2 = nn.GroupNorm(8, out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, t_emb):
        h = self.conv1(F.silu(self.norm1(x)))
        scale, shift = self.film(t_emb).chunk(2, dim=-1)
        h = h * (1 + scale[:, :, None, None]) + shift[:, :, None, None]
        h = self.conv2(F.silu(self.norm2(h)))
        return h + self.skip(x)


class SelfAttention2d(nn.Module):
    """Plain multi-head self-attention over flattened spatial positions.
    Only used at the bottleneck — cheapest resolution, biggest payoff for
    long-range mixing on a small model.
    """

    def __init__(self, channels, heads=4):
        super().__init__()
        self.heads = heads
        self.norm = nn.GroupNorm(8, channels)
        self.qkv = nn.Conv2d(channels, channels * 3, 1)
        self.proj = nn.Conv2d(channels, channels, 1)
        # Zero-init: at step 0 this block is an exact identity (x + 0 = x).
        # Without it, a randomly-initialized proj injects noise into the
        # residual stream from the very first step; the network then has to
        # spend early training just undoing that instead of learning
        # anything useful. Standard practice for new attention blocks added
        # to a diffusion U-Net.
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.qkv(self.norm(x)).reshape(b, 3, self.heads, c // self.heads, h * w)
        q, k, v = qkv.unbind(1)
        attn = torch.softmax(q.transpose(-1, -2) @ k / math.sqrt(c // self.heads), dim=-1)
        out = (attn @ v.transpose(-1, -2)).transpose(-1, -2).reshape(b, c, h, w)
        return x + self.proj(out)


class TypeTokens(nn.Module):
    """One learned token sequence per edit type — the conditioning signal that
    replaced CLIP text embeddings in v3.

    Shape-compatible with what CrossAttention2d already consumed ([B, L, dim]),
    so the attention path needed no changes at all. The difference is what the
    tokens mean: previously an encoding of an English sentence the model had no
    hope of grounding from 60k pairs, now a set of free parameters the model
    shapes itself, trained by thousands of consistent examples per type.

    N_TOKENS > 1 matters. A single vector per type would collapse this back to
    something FiLM-like, with nothing for different image regions to attend to
    differently. Several tokens let the model split a transformation into parts
    that address different content ("sky" vs "foreground" for a sunset), which is
    what cross-attention is for.

    Type 0 is the null type used by conditioning dropout and as the guidance
    baseline. It gets a real learned row like any other, so the unconditional
    branch is trained rather than extrapolated.
    """

    def __init__(self, n_types, n_tokens, dim):
        super().__init__()
        self.n_tokens = n_tokens
        self.dim = dim
        self.emb = nn.Embedding(n_types, n_tokens * dim)
        nn.init.normal_(self.emb.weight, std=0.02)

    def forward(self, type_ids):
        """type_ids: [B] long -> [B, n_tokens, dim]."""
        return self.emb(type_ids).view(-1, self.n_tokens, self.dim)


class CrossAttention2d(nn.Module):
    """Each spatial position (query) attends over the conditioning token
    sequence (key/value) — this is what lets one part of a transformation drive
    one region of the image, instead of everything blending into a single global
    scale/shift the way FiLM did.
    """

    def __init__(self, channels, text_dim, heads=4):
        super().__init__()
        self.heads = heads
        self.head_dim = channels // heads
        self.norm = nn.GroupNorm(8, channels)
        self.to_q = nn.Conv2d(channels, channels, 1)
        self.to_k = nn.Linear(text_dim, channels)
        self.to_v = nn.Linear(text_dim, channels)
        self.proj = nn.Conv2d(channels, channels, 1)
        # Zero-init, same reasoning as SelfAttention2d — with NINE of these
        # blocks chained through the U-Net (4 down + bottleneck + 4 up), an
        # un-zeroed proj compounds noise at every one of them before the
        # network has learned anything, which is why an earlier run produced
        # near-pure static instead of a blurry-but-coherent image the way the
        # FiLM model did at a comparable stage.
        #
        # Note the flip side, learned the hard way: zero-init also lets the model
        # LEAVE these blocks near zero if conditioning isn't worth using, which is
        # exactly what happened with CLIP text conditioning. It is the right
        # initialization, but it only pays off when the conditioning signal is
        # strong enough to beat the copy-the-input shortcut — hence v3's discrete
        # types.
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x, text_seq):
        """x: [B,C,H,W]. text_seq: [B,L,text_dim]."""
        b, c, h, w = x.shape
        L = text_seq.shape[1]
        q = self.to_q(self.norm(x)).reshape(b, self.heads, self.head_dim, h * w)
        k = self.to_k(text_seq).reshape(b, L, self.heads, self.head_dim).permute(0, 2, 3, 1)
        v = self.to_v(text_seq).reshape(b, L, self.heads, self.head_dim).permute(0, 2, 1, 3)
        attn = torch.softmax(q.transpose(-1, -2) @ k / math.sqrt(self.head_dim), dim=-1)  # b,heads,hw,L
        out = (attn @ v).transpose(-1, -2).reshape(b, c, h, w)  # b,heads,hw,head_dim -> b,c,h,w
        return x + self.proj(out)


class Downsample(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.op = nn.Conv2d(ch, ch, 3, stride=2, padding=1)

    def forward(self, x):
        return self.op(x)


class Upsample(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.op = nn.Conv2d(ch, ch, 3, padding=1)

    def forward(self, x):
        return self.op(F.interpolate(x, scale_factor=2, mode="nearest"))


class UNet(nn.Module):
    """base_channels=128, channel_mults=(1,2,3,4) -> 128/256/384/512 at
    128/64/32/16px, 2 ResBlocks per level. 70.5M parameters.

    v5 sizing, chosen by measurement on real T4x2 (kaggle/08-size-probe.py) rather
    than by parameter count. The 22.4M predecessor learned filters cleanly but
    could not synthesize objects, and adding 40 object/semantic types visibly
    DEGRADED the filters it already knew — capacity contention, not undertraining.

    The mults matter more than the width. A 64.8M config at base 112 with
    (1,2,4,4) measured 2.00 s/step, while this 70.5M one measured 1.69 — more
    parameters, less time — because UNet cost is dominated by the high-resolution
    levels, and (1,2,3,4) moves capacity down to where the feature maps are small.
    Parameter count is a poor proxy for what a config costs to train. Cross-attention to the edit type's
    token sequence after every resolution level's ResBlocks (down and up) plus
    the bottleneck, so a transformation can be driven at multiple spatial scales
    rather than only the coarsest. Self-attention (spatial-only) also at the
    bottleneck, where it is cheapest and buys the most long-range mixing.
    """

    def __init__(self, n_types, base_channels=128, channel_mults=(1, 2, 3, 4),
                 token_dim=256, n_tokens=8, time_dim=256):
        super().__init__()
        self.base_channels = base_channels
        self.n_types = n_types
        self.time_mlp = nn.Sequential(
            nn.Linear(base_channels, time_dim), nn.SiLU(), nn.Linear(time_dim, time_dim))
        self.type_tokens = TypeTokens(n_types, n_tokens, token_dim)
        text_dim = token_dim

        self.in_conv = nn.Conv2d(6, base_channels, 3, padding=1)

        chs = [base_channels * m for m in channel_mults]
        self.down_blocks = nn.ModuleList()
        self.down_attns = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        self.skip_channels = []
        in_ch = base_channels
        for i, ch in enumerate(chs):
            self.down_blocks.append(nn.ModuleList([
                TimeResBlock(in_ch, ch, time_dim),
                TimeResBlock(ch, ch, time_dim),
            ]))
            self.down_attns.append(CrossAttention2d(ch, text_dim))
            self.skip_channels.append(ch)
            in_ch = ch
            self.downsamples.append(Downsample(ch) if i < len(chs) - 1 else nn.Identity())

        self.mid_block1 = TimeResBlock(in_ch, in_ch, time_dim)
        self.mid_self_attn = SelfAttention2d(in_ch)
        self.mid_cross_attn = CrossAttention2d(in_ch, text_dim)
        self.mid_block2 = TimeResBlock(in_ch, in_ch, time_dim)

        self.up_blocks = nn.ModuleList()
        self.up_attns = nn.ModuleList()
        self.upsamples = nn.ModuleList()
        for i, ch in reversed(list(enumerate(chs))):
            skip_ch = self.skip_channels[i]
            self.up_blocks.append(nn.ModuleList([
                TimeResBlock(in_ch + skip_ch, ch, time_dim),
                TimeResBlock(ch + skip_ch, ch, time_dim),
            ]))
            self.up_attns.append(CrossAttention2d(ch, text_dim))
            in_ch = ch
            self.upsamples.append(Upsample(ch) if i > 0 else nn.Identity())

        self.out_norm = nn.GroupNorm(8, in_ch)
        self.out_conv = nn.Conv2d(in_ch, 3, 3, padding=1)

    def forward(self, x, t, type_ids):
        """x: [B,6,H,W] (noisy_after || before). t: [B] long.
        type_ids: [B] long, edit type indices (0 = null / unconditional).
        """
        t_emb = self.time_mlp(timestep_embedding(t, self.base_channels))
        text_seq = self.type_tokens(type_ids)

        h = self.in_conv(x)
        skips = []
        for (b1, b2), attn, down in zip(self.down_blocks, self.down_attns, self.downsamples):
            h = b1(h, t_emb)
            h = b2(h, t_emb)
            h = attn(h, text_seq)
            skips.append(h)
            h = down(h)

        h = self.mid_block1(h, t_emb)
        h = self.mid_self_attn(h)
        h = self.mid_cross_attn(h, text_seq)
        h = self.mid_block2(h, t_emb)

        for (b1, b2), attn, up in zip(self.up_blocks, self.up_attns, self.upsamples):
            skip = skips.pop()
            h = b1(torch.cat([h, skip], dim=1), t_emb)
            h = b2(torch.cat([h, skip], dim=1), t_emb)
            h = attn(h, text_seq)
            h = up(h)

        return self.out_conv(F.silu(self.out_norm(h)))
