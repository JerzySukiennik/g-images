"""Deterministic image edits, generated on the fly — perfect supervision for
every transformation that is actually a known function of the input.

Why this exists
---------------
The coverage probe (kaggle/04-probe-types.py) counted how the real 60k
InstructPix2Pix prompts distribute across edit types, and the answer reframed the
whole project: "make it black and white" — the instruction used as the acceptance
test all day — has 588 examples in 60000. Sepia has 182. Brighter has 3. The
corpus is dominated by semantic subject swaps ("make her a panda", "make it the
Taj Mahal"), with 23% of it landing in a catch-all that describes no single
transformation at all.

So the model was being asked to learn colour and tone filters from a few hundred
noisy examples each. That was never going to work, and no amount of training
steps or conditioning dropout was going to fix it.

But these particular edits don't need scraped data. Desaturating an image,
warming it, brightening it, blurring it — each is an exact function we can apply
ourselves. Pairing a real photo with its own programmatically edited version
gives supervision that is unlimited, perfectly consistent, and free: every one of
the 60k `before` images already on disk becomes a training pair for every filter
here. That is strictly better data than the scraped kind for this class of edit,
not a compromise.

The semantic types (painting, snow, cartoon...) still come from the real pairs,
where a deterministic function genuinely can't produce the target. The two sources
mix in one training set; see train/train.py's PairDataset.

Everything operates on [-1,1] CHW float tensors, matching the training format, and
uses only torch ops so it runs inside DataLoader workers without a PIL round-trip.
"""

import torch
import torch.nn.functional as F

# Rec. 601 luma weights — the same ones PIL and ffmpeg use for grayscale, so a
# model trained here matches what a user gets from any other tool.
_LUMA = torch.tensor([0.299, 0.587, 0.114]).view(3, 1, 1)


def _luma(img):
    return (img * _LUMA.to(img.device, img.dtype)).sum(dim=0, keepdim=True)


def grayscale(img):
    return _luma(img).expand(3, -1, -1).clone()


def sepia(img):
    g = _luma(img)
    # Warm tone applied to the luma channel: lift red, leave green, drop blue.
    tint = torch.tensor([1.07, 0.94, 0.72]).view(3, 1, 1).to(img.device, img.dtype)
    return (g.expand(3, -1, -1) * tint).clamp(-1, 1)


def brighter(img):
    # In [-1,1] space a gain toward +1 brightens without clipping mid-tones as
    # hard as a plain additive offset would.
    return (img * 0.75 + 0.35).clamp(-1, 1)


def darker(img):
    return (img * 0.75 - 0.3).clamp(-1, 1)


def more_saturated(img):
    g = _luma(img)
    return (g + (img - g) * 1.9).clamp(-1, 1)


def less_saturated(img):
    g = _luma(img)
    return (g + (img - g) * 0.35).clamp(-1, 1)


def _gauss_kernel(sigma, device, dtype):
    radius = max(1, int(3 * sigma))
    xs = torch.arange(-radius, radius + 1, device=device, dtype=dtype)
    k = torch.exp(-(xs ** 2) / (2 * sigma ** 2))
    return (k / k.sum()), radius


def blur(img, sigma=2.5):
    """Separable Gaussian — two 1D passes instead of one 2D kernel, which is
    both cheaper and exactly equivalent for a Gaussian."""
    k, r = _gauss_kernel(sigma, img.device, img.dtype)
    x = img.unsqueeze(0)
    x = F.conv2d(F.pad(x, (r, r, 0, 0), mode="reflect"),
                 k.view(1, 1, 1, -1).expand(3, 1, 1, -1), groups=3)
    x = F.conv2d(F.pad(x, (0, 0, r, r), mode="reflect"),
                 k.view(1, 1, -1, 1).expand(3, 1, -1, 1), groups=3)
    return x.squeeze(0)


def sharpen(img):
    """Unsharp mask: push the image away from its own blurred version."""
    return (img + (img - blur(img, sigma=1.5)) * 1.2).clamp(-1, 1)


def warmer(img):
    tint = torch.tensor([1.12, 1.0, 0.85]).view(3, 1, 1).to(img.device, img.dtype)
    return (img * tint).clamp(-1, 1)


def cooler(img):
    tint = torch.tensor([0.85, 0.98, 1.15]).view(3, 1, 1).to(img.device, img.dtype)
    return (img * tint).clamp(-1, 1)


def invert(img):
    return -img


def high_contrast(img):
    return (img * 1.7).clamp(-1, 1)


def low_contrast(img):
    return img * 0.5


# Name -> function. Names are the edit-type names in data/edit_types.py; keeping
# them identical means the taxonomy has exactly one entry per transformation
# regardless of whether its data is synthetic or scraped.
SYNTHETIC = {
    "black_and_white": grayscale,
    "sepia_vintage": sepia,
    "brighter": brighter,
    "darker": darker,
    "more_colorful": more_saturated,
    "less_colorful": less_saturated,
    "blur_background": blur,
    "sharper": sharpen,
    "warmer": warmer,
    "cooler": cooler,
    "inverted": invert,
    "high_contrast": high_contrast,
    "low_contrast": low_contrast,
}
