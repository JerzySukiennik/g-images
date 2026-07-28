"""Discrete edit-type taxonomy — the conditioning signal for Gedit v2.

Why this replaces free-form CLIP text conditioning
--------------------------------------------------
Measured on the step-94200 checkpoint (runtime/guidance_probe.py): the cosine
similarity between the model's noise prediction given the real prompt and given
an empty prompt is 1.000 at every timestep, and the difference vector is ~1-2%
of the signal magnitude. The model had learned to almost entirely ignore the
text. Classifier-free guidance then has nothing to amplify, which is why raising
its scale only ever produced saturated colour noise, and why raising conditioning
dropout from 5% to 20% changed nothing.

That is not a bug to fix with more steps. With `before` concatenated on the
channel axis, copying the input is by far the cheapest way to lower the MSE, so
the only reason to learn language would be to explain the residual — and 60k
pairs is orders of magnitude too little to learn what English phrases mean from
scratch. InstructPix2Pix does not do this either: it fine-tunes Stable Diffusion,
which already understands text from billions of image-caption pairs, and adds
450k edit pairs on top.

So the model stops learning language and learns a small number of concrete
transformations instead. Each type gets its own learned token sequence (see
model/unet.py's TypeTokens) feeding the same cross-attention the CLIP sequence
used to feed — conditioning becomes unambiguous and strong, while spatial
routing (the whole reason cross-attention replaced FiLM) is preserved. Mapping a
user's free-form sentence onto one of these types is an inference-time concern,
deliberately deferred.

Rules are ordered: the first match wins, so put specific patterns above generic
ones. classify() returns None for prompts that match nothing, and those pairs are
dropped from training — a pair whose transformation we cannot name is exactly the
kind of inconsistent supervision that taught the model to ignore conditioning.
"""

# NULL_TYPE is the trained unconditional branch: conditioning dropout swaps a
# real type for this one, and sampling uses it as the guidance baseline. It is a
# real embedding row, not zeros, so the branch is something the model was taught
# rather than something it extrapolates.
NULL_TYPE = 0

# (name, [substrings]) — matched case-insensitively against the whole prompt.
RULES = [
    ("black_and_white", ["black and white", "grayscale", "greyscale", "monochrome",
                          "desaturate", "remove the color", "remove the colour"]),
    ("sepia_vintage", ["sepia", "vintage", "retro", "old photo", "aged photo",
                        "antique"]),
    ("painting", ["painting", "oil paint", "watercolor", "watercolour", "as a canvas",
                   "impressionist", "van gogh", "picasso", "monet"]),
    ("drawing_sketch", ["sketch", "pencil drawing", "charcoal", "line drawing",
                         "as a drawing", "doodle"]),
    ("cartoon_anime", ["cartoon", "anime", "comic", "cel shad", "pixar",
                        "as an illustration"]),
    ("pixel_art", ["pixel art", "8-bit", "8 bit", "16-bit", "voxel"]),
    ("snow_winter", ["snow", "winter", "blizzard", "frozen", "icy", "frost"]),
    ("rain_storm", ["rain", "storm", "thunder", "lightning", "wet weather"]),
    ("fog_mist", ["fog", "mist", "haze", "smog"]),
    ("sunset_sunrise", ["sunset", "sunrise", "golden hour", "dusk", "dawn"]),
    ("night", ["night", "at midnight", "dark sky", "moonlit", "starry"]),
    ("autumn", ["autumn", "fall colors", "fall colours", "autumnal"]),
    ("spring_bloom", ["spring", "blossom", "in bloom", "flowers everywhere"]),
    ("desert", ["desert", "sand dune", "sahara", "arid"]),
    ("underwater", ["underwater", "under the sea", "submerged", "ocean floor"]),
    ("space_scifi", ["space", "galaxy", "nebula", "sci-fi", "futuristic",
                      "cyberpunk", "neon city"]),
    ("fire_lava", ["fire", "flames", "burning", "lava", "volcanic", "on fire"]),
    ("apocalypse_ruin", ["apocalyp", "ruined", "abandoned", "destroyed",
                          "post-apocalyptic", "wasteland"]),
    ("brighter", ["brighter", "brighten", "more light", "well lit", "increase exposure"]),
    ("darker", ["darker", "darken", "less light", "dim the", "decrease exposure"]),
    ("more_colorful", ["more colorful", "more colourful", "vibrant", "saturate",
                        "vivid", "psychedelic"]),
    ("blur_background", ["blur", "bokeh", "depth of field", "out of focus"]),
    ("add_object", ["add ", "put a ", "put an ", "give it a ", "give him a ",
                     "give her a ", "place a ", "insert a "]),
    ("remove_object", ["remove ", "delete ", "erase ", "take away ", "get rid of"]),
    ("replace_with", ["replace ", "turn the ", "swap ", "change the ", "make the "]),
]

TYPE_NAMES = ["null"] + [name for name, _ in RULES]
N_TYPES = len(TYPE_NAMES)


def classify(prompt):
    """prompt: str -> type id in 1..N_TYPES-1, or None if nothing matches.

    Never returns NULL_TYPE: that id belongs to conditioning dropout, not to any
    real prompt.
    """
    p = prompt.lower()
    for i, (_, needles) in enumerate(RULES, start=1):
        for needle in needles:
            if needle in p:
                return i
    return None


def classify_all(prompts):
    """-> (labels, kept_indices). labels is aligned with kept_indices, not with
    the input, since unmatched prompts are dropped rather than bucketed into a
    catch-all class (a catch-all would be the inconsistent supervision this
    taxonomy exists to remove)."""
    labels, kept = [], []
    for i, p in enumerate(prompts):
        t = classify(p)
        if t is not None:
            labels.append(t)
            kept.append(i)
    return labels, kept
