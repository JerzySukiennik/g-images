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

# --- v4: AnyEdit taxonomy -----------------------------------------------------
#
# The IP2P types above are kept in the list even though the new corpus does not
# feed them. Their POSITIONS are load-bearing: train/train.py grows the edit-type
# embedding table on resume by copying old rows into a larger one, which is only
# correct if existing ids keep their meaning. Removing them would shift every
# later index and silently re-attach learned filter behaviour to the wrong type.
# They simply never get sampled. New types are APPENDED, never inserted.

# tune_transfer instructions are formulaic ("change the weather to snow", "make
# the season autumn"), so the target word alone identifies the transformation.
# Synonyms are merged: snow/snowy is one edit, not two, and splitting them would
# halve the data for no gain. Counts measured over 60000 tune_transfer pairs.
TUNE_RULES = [
    ("season_winter", ["winter"]),                          # 12600
    ("weather_snow", ["snow", "snowy"]),                    #  8452
    ("season_autumn", ["autumn", "fall"]),                  #  7802
    ("time_night", ["night", "midnight", "nighttime"]),     #  6382
    ("time_evening", ["evening", "dusk", "sunset"]),        #  5512
    ("weather_storm", ["storm", "stormy", "thunderstorm"]), #  4641
    ("weather_fog", ["fog", "foggy", "mist", "misty"]),     #  4108
    ("season_summer", ["summer"]),                          #  2384
]

# Whole AnyEdit types used as-is. `replace` is deliberately excluded: "replace X
# with Y" names no single transformation, exactly like the IP2P `replace_with`
# bucket that taught the previous model to ignore conditioning.
ANYEDIT_WHOLE = ["style_change", "background_change"]

# Top 30 objects by frequency across 270000 AnyEdit `add` pairs, all >= 1779
# examples. In InstructPix2Pix the best-represented object had 351 and "hat" had
# 109; here hat has 1849 and sits at rank 27.
ADD_OBJECTS = [
    "ball",           # 8021
    "bouquet",        # 7534
    "cup",            # 6975
    "balloon",        # 6478
    "seagull",        # 5703
    "butterfly",      # 4980
    "book",           # 4856
    "vase",           # 4835
    "sailboat",       # 4746
    "bird",           # 4032
    "basket",         # 3962
    "glass",          # 3723
    "cat",            # 3717
    "dog",            # 3468
    "guitar",         # 3248
    "slice",          # 3075
    "bear",           # 2875
    "flower",         # 2436
    "tree",           # 2272
    "candle",         # 2266
    "fish",           # 2207
    "cake",           # 2184
    "hat",            # 1877
    "banner",         # 1565
    "performer",      # 1409
    "chef",           # 1382
    "bookshelf",      # 1342
    "surfboard",      # 1298
    "farmer",         # 1237
    "ribbon",         # 1168
]

SEMANTIC_TYPES = [name for name, _ in TUNE_RULES] + ANYEDIT_WHOLE
ADD_TYPES = [f"add_{o}" for o in ADD_OBJECTS]

TYPE_NAMES = TYPE_NAMES + SEMANTIC_TYPES + ADD_TYPES
N_TYPES = len(TYPE_NAMES)
TYPE_ID = {name: i for i, name in enumerate(TYPE_NAMES)}
_TUNE_LOOKUP = {w: name for name, words in TUNE_RULES for w in words}
_ADD_SET = set(ADD_OBJECTS)


def classify_anyedit(anyedit_type, instruction):
    """(AnyEdit edit_type, instruction) -> our type id, or None to drop the row.

    Dropping is the default for anything not confidently nameable — the whole
    point of this taxonomy is that every type means one transformation.
    """
    if anyedit_type == "add":
        obj = add_object(instruction)
        if obj in _ADD_SET:
            return TYPE_ID[f"add_{obj}"]
        return None
    if anyedit_type == "tune_transfer":
        words = [w for w in re.split(r"[^a-z]+", (instruction or "").lower()) if w]
        for w in reversed(words):
            if w in _TUNE_LOOKUP:
                return TYPE_ID[_TUNE_LOOKUP[w]]
        return None
    if anyedit_type in ANYEDIT_WHOLE:
        return TYPE_ID[anyedit_type]
    return None



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
    "shining", "hovering", "standing", "sitting", "large", "small", "big",
    "little", "sky",
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
    if not words:
        return None
    # Singular and plural name the same transformation. Left unmerged, the first
    # taxonomy had both add_seagull (3735) and add_seagulls (1919) as separate
    # types, splitting one edit's data across two embedding rows for no reason.
    w = words[-1]
    if w.endswith("ies") and len(w) > 4:
        return w[:-3] + "y"
    if w.endswith(("ses", "xes", "hes")):
        return w[:-2]
    if w.endswith("s") and not w.endswith("ss") and len(w) > 3:
        return w[:-1]
    return w


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
