"""Discrete edit-type taxonomy — the conditioning signal for G-Images v3.

Why this replaced free-form CLIP text conditioning
--------------------------------------------------
Measured on the step-94200 checkpoint (runtime/guidance_probe.py): the cosine
similarity between the model's prediction given the real prompt and given an empty
prompt was 1.000 at every timestep, with the difference vector at ~1-2% of signal
magnitude. The model had learned to almost entirely ignore the text, so
classifier-free guidance had nothing to amplify — which is why raising its scale
only produced saturated colour noise, and why raising conditioning dropout from 5%
to 20% (4.5x more null exposure) changed nothing at all.

That is not fixable with more steps. With `before` concatenated on the channel
axis, copying the input is by far the cheapest way to lower the loss, so learning
language would only pay off for explaining the residual — and 60k pairs is orders
of magnitude too little to learn what English phrases mean from scratch.
InstructPix2Pix does not attempt it either: it fine-tunes Stable Diffusion, which
already understands text from billions of image-caption pairs.

Why the type list looks like this
---------------------------------
kaggle/04-probe-types.py counted the real 60k prompts against an earlier, larger
taxonomy. Two findings reshaped it:

1. The transformations we most wanted are barely present. "black and white" has
   588 examples in 60000; sepia has 182; "brighter" has 3. We spent the entire day
   judging the model on an edit represented by 1% of its training data.

2. 23% of the corpus fell into a `replace_with` bucket ("make it the Taj Mahal",
   "make her a panda") and another 7% into `add_object`. Those name no single
   transformation, so training them as one type would recreate the exact
   inconsistent supervision this taxonomy exists to remove. Both are dropped. That
   does mean "add a hat" is out of scope for now — a deliberate cost of Jurek's
   "make it actually do something, prompts later" call.

So types split by where good supervision comes from:

  SYNTHETIC_TYPES — edits that are exact functions of the input, generated on the
  fly by data/synthetic_edits.py from the `before` images already on disk. Every
  one of the 60k images is a valid pair for every one of these, so the data is
  unlimited and perfectly consistent. For these, synthetic supervision is strictly
  better than the scraped kind, which was both scarce and imperfect (the corpus's
  own "make it black and white" targets are Stable Diffusion outputs, not true
  desaturations).

  REAL_TYPES — edits no deterministic function can produce, kept only where the
  probe found enough consistent examples to teach one. Counts from the probe are
  in the comments; anything under ~400 was dropped as untrainable.

Type 0 is `null`: the trained unconditional branch used by conditioning dropout
and as the guidance baseline.
"""

import re

NULL_TYPE = 0

# Exact image functions — see data/synthetic_edits.py for the implementations.
# Unlimited pairs each, so these carry the burden of proving the model can follow
# conditioning at all.
SYNTHETIC_TYPES = [
    "black_and_white",
    "sepia_vintage",
    "brighter",
    "darker",
    "more_colorful",
    "less_colorful",
    "blur_background",
    "sharper",
    "warmer",
    "cooler",
    "inverted",
    "high_contrast",
    "low_contrast",
]

# Scraped from the real pairs. (probe count in the 60k set)
REAL_RULES = [
    ("painting", ["painting", "oil paint", "watercolor", "watercolour",
                   "impressionist", "van gogh", "monet", "picasso"]),          # 6290
    ("rain_storm", ["rain", "storm", "thunder", "lightning"]),                  # 2116
    ("snow_winter", ["snow", "winter", "blizzard", "frozen", "icy", "frost"]),  # 1497
    ("cartoon_anime", ["cartoon", "anime", "comic", "cel shad", "pixar"]),      # 1291
    ("desert", ["desert", "sand dune", "sahara", "arid"]),                      # 1156
    ("fire_lava", ["fire", "flames", "burning", "lava", "volcanic"]),           # 901
    ("night", ["night", "midnight", "dark sky", "moonlit", "starry"]),          # 792
    ("sunset_sunrise", ["sunset", "sunrise", "golden hour", "dusk", "dawn"]),   # 585
    ("space_scifi", ["space", "galaxy", "nebula", "sci-fi", "futuristic",
                      "cyberpunk", "neon"]),                                    # 486
    ("drawing_sketch", ["sketch", "pencil drawing", "charcoal", "line drawing",
                         "woodcut", "ink drawing"]),                            # 422
]

REAL_TYPES = [name for name, _ in REAL_RULES]
TYPE_NAMES = ["null"] + SYNTHETIC_TYPES + REAL_TYPES
N_TYPES = len(TYPE_NAMES)
TYPE_ID = {name: i for i, name in enumerate(TYPE_NAMES)}


# Boundary words that end the object phrase in an AnyEdit `add` instruction.
# Real examples: "add a fresh fruit bowl ON the table", "add a big red hat ON the
# horse", "add a person TAKING a bath", "add a cup OF coffee in his hand". Without
# cutting here, the head noun comes out as "table"/"horse"/"bath"/"hand" — the
# place the object went, not the object. A first attempt did exactly that and
# produced a histogram of "of", "taking", "eating".
_ADD_BOUNDARY = re.compile(
    r"\b(?:on|in|at|next|between|near|under|with|behind|over|of|to|from|beside|"
    r"against|around|inside|outside|beneath|above|below|along|across|onto|into|"
    r"nearby|overhead|out|off|up|down|away|there|here|"
    r"taking|holding|standing|eating|lying|preparing|watching|sitting|walking|"
    r"playing|wearing|hanging|floating|resting|leaning|flying|perched|waving|"
    r"waiting|swimming|running|jumping|reading|riding|carrying|looking|smiling|"
    r"tied|placed|parked|seated|attached|surrounded|covered|filled)\b")

# Words that are never the object itself. A first pass produced "nearby" (2568),
# "overhead" (1772) and "flying" (1063) as top "objects" — all locative or
# participial tails that survived because the boundary list was too short. Kept
# as a second net so a missed boundary word degrades to a dropped row rather than
# to a bogus type with thousands of contradictory examples.
_NOT_AN_OBJECT = {
    "nearby", "overhead", "out", "above", "below", "here", "there", "away",
    "flying", "perched", "waving", "waiting", "swimming", "running", "tied",
    "placed", "parked", "seated", "attached", "left", "right", "top", "bottom",
    "side", "front", "back", "middle", "corner", "area", "background",
    "foreground", "scene", "image", "picture", "photo", "one", "two", "some",
    "more", "another", "other", "same", "new", "few", "several", "many",
}
_ADD_HEAD = re.compile(r"^\s*(?:add|include|put|place|insert)\s+"
                        r"(?:a|an|the|some)?\s*(.+)$", re.I)


def add_object(instruction):
    """AnyEdit `add` instruction -> head noun of the added object, or None.

    "add a big red hat on the horse" -> "hat"
    "include a beach umbrella"       -> "umbrella"
    "add a cup of coffee in his hand"-> "cup"
    """
    m = _ADD_HEAD.match(instruction or "")
    if not m:
        return None
    phrase = m.group(1).lower()
    cut = _ADD_BOUNDARY.search(phrase)
    if cut:
        phrase = phrase[:cut.start()]
    words = [w for w in re.split(r"[^a-z]+", phrase) if w]
    while words and words[-1] in _NOT_AN_OBJECT:
        words.pop()
    return words[-1] if words else None


def classify_real(prompt):
    """prompt: str -> type id for one of REAL_TYPES, or None.

    Ordered: first match wins, so more specific patterns sit higher. Returns None
    for anything unmatched, and those pairs are dropped rather than swept into a
    catch-all — an unnameable transformation is precisely the supervision that
    taught the previous model to ignore its conditioning.

    Deliberately never returns a synthetic type, even when a prompt mentions one:
    the corpus's own "make it black and white" targets are generated images, not
    true desaturations, so mixing them in would contradict the exact synthetic
    pairs.
    """
    p = prompt.lower()
    for name, needles in REAL_RULES:
        for needle in needles:
            if needle in p:
                return TYPE_ID[name]
    return None


def real_index_by_type(prompts):
    """prompts: list[str] -> {type_id: [indices]} for the real types only."""
    out = {}
    for i, p in enumerate(prompts):
        t = classify_real(p)
        if t is not None:
            out.setdefault(t, []).append(i)
    return out
