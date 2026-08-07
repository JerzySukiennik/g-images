# G-Images — handoff

Read this before touching anything. It covers the state as of **2026-08-03**, the
things that will bite you, and the one open question the project is currently
spending GPU quota to answer.

The vault card (`~/Downloads/Claude/ClaudeMemory/projects/g-images.md`) has the
full chronology; this file is the operating manual.

---

## What this is

A photo-editing diffusion model trained **from scratch** — no pretrained weights
except a CLIP text encoder that v3 removed. Called from Gzowo AI by voice. Repo:
`github.com/JerzySukiennik/g-images` (renamed from `gedit` on 2026-07-28).

It went through four architectures. Knowing why matters, because the dead ends
are easy to walk back into:

- **v1 FiLM** — one pooled text vector scaling the whole image. Could shift tone,
  could not localize. "Add a hat" gave a global colour shift.
- **v2 cross-attention over CLIP tokens** — trained to step 94200. Then
  `runtime/guidance_probe.py` measured `cos(e_cond, e_null) = 1.000` at every
  timestep, with the conditional/unconditional difference at ~1-2% of signal.
  **The model had learned to ignore the text almost entirely.** Classifier-free
  guidance had nothing to amplify, which is why raising its scale only produced
  colour garbage and why raising conditioning dropout 5% → 20% changed nothing.
- **v3 discrete edit types** — conditioning became a type id with its own learned
  token sequence feeding the same cross-attention. Free-form prompts are out of
  scope for now, by Jurek's call ("make it actually do something, prompts later").
- **v4/v5 current** — same idea, 64 types, AnyEdit data, 70.5M parameters.

### Why v2 could never have worked

`kaggle/04-probe-types.py` counted the corpus we were training on. "make it black
and white" — the instruction used to judge the model for five rounds — had **588
examples in 60000**. Sepia had 182. "brighter" had 3. Meanwhile 23% of prompts
fell into a `replace_with` bucket ("make her a panda", "make it the Taj Mahal")
that names no single transformation.

The model was being graded on an edit worth 1% of its data while being fed a third
of a corpus of mutually contradictory targets. It learned the rational thing:
ignore the conditioning and copy the input, which with `before` concatenated on
the channel axis is always the cheapest way to lower the loss.

---

## Current state

**Training in progress.** 70.5M model, from scratch, target 40000 steps, running
in ~15000-step sessions. Session 1 (0 → 15000) and session 2 (15000 → 30000) are
done. One session remains.

- Repo HEAD: see `git log`; everything is pushed.
- Checkpoints live as Kaggle dataset `jerzysukiennik/gedit-ckpt` (versioned).
- Data: five prep notebook outputs, `gedit-anyedit-0` … `-4`, 400000 pairs total,
  390000 in the training split.

### What works

**Synthetic filters.** 13 types that are exact functions of the input —
grayscale, sepia, brightness, saturation, blur, sharpen, tint, invert, contrast —
generated on the fly by `data/synthetic_edits.py` from the `before` images. These
work: `black_and_white` desaturates correctly, verified against ground truth in
the same grid. This was the project's first real success, and it arrived after
3600 steps where the previous architecture never managed it in 94200.

The reason is worth internalizing: **for edits that are exact functions of the
input, synthetic supervision is strictly better than scraped data**, not a
compromise. Unlimited, perfectly consistent, and free — while the corpus's own
"black and white" targets are Stable Diffusion outputs rather than true
desaturations.

### What does not work yet

**Object addition.** `add_hat` and friends place *something* localized near the
right region, and different types produce different textures — so conditioning
and localization both work — but they are blobs, not recognizable objects.

**`time_night`** goes the wrong way (warmer, not darker) despite 6382 examples.

### The open question

At 22.4M, going from step 41200 to 59200 did **not** sharpen objects and visibly
**degraded** `black_and_white`, which had been clean earlier. That is capacity
contention across 53 sampled types, not undertraining — which is why the model is
now 3x larger.

**If objects are still blobs at step 40000, the answer stops being "a bigger
pixel-space model" and becomes latent-space diffusion** — train an autoencoder,
run diffusion on its codes. A 64×64 latent is 4x fewer points than 128×128 pixels
at the same output resolution, which is precisely how Stable Diffusion affords to
synthesize objects.

---

## Landmines

These all cost real time. Each is now guarded in code, but the guards only help if
you know why they exist.

**1. Kaggle serves output only for the LATEST kernel version.** Push a new version
before harvesting the previous run and that checkpoint becomes unreachable via the
CLI. **Always harvest, then push.**

**2. `kaggle datasets status` returning `ready` does not mean your upload landed.**
It describes the previous version. After uploading a large checkpoint, poll the
dataset's **size** until it is non-zero and matches what you sent — a 1.13 GB
checkpoint reported size 0 for several minutes while Kaggle processed it, and
pushing during that window resumes training from the *old* checkpoint. Verify
content, never status.

**3. `git add -A` silently skips gitignored files.** `data/*.json` in `.gitignore`
swallowed `data/anyedit_shardmap.json`, a source file the prep kernels need.
Commit and push both reported success; three kernels then died on
`FileNotFoundError`. Check `git ls-tree HEAD <path>` for anything load-bearing.

**4. Parallel jobs are not automatically disjoint.** `pick_shards` started at the
first shard every time, so three prep notebooks downloaded **identical rows** —
150000 `add` pairs of which 60000 were unique, all three reporting `COMPLETE` with
correct file sizes. Skipping is now counted in **shards, not rows**, because a
row-based skip does not land on shard boundaries. If you parallelize anything,
compare the outputs against each other.

**5. Growing the model means growing the optimizer too.** Expanding the edit-type
embedding table 24 → 64 loaded fine and printed a happy message, then `opt.step()`
died inside `_multi_tensor_adam`: AdamW keeps two momentum tensors per parameter
and the restored ones still had the old shape. They are now zero-padded.

**6. Parameter count is a poor proxy for training cost.** A 64.8M config measured
**2.00 s/step** while a 70.5M one measured **1.69** — UNet compute is dominated by
the high-resolution levels, so `(1,2,3,4)` buys capacity more cheaply than
`(1,2,4,4)`. Always measure on the real machine (`kaggle/08-size-probe.py`), and
**include an anchor config whose throughput you already know** — the probe read
0.64 s/step for a model that really runs at 0.78-0.84, because it used synthetic
batches and skipped data loading. That gap shrinks as the model grows (only +8% at
70.5M), so don't apply a flat correction.

**7. Never judge output below ~50 DDIM steps.** 20 steps once hid real structure
behind a washed-out result that looked identical regardless of conditioning, and
cost a full round of wrong conclusions. Default is 100.

**8. Data file names are load-bearing.** The prep outputs are `gedit_images.bin`,
`gedit_prompts.json`, `gedit_types.json`, `gedit_meta.json`, and Kaggle slugs are
still `gedit-*`. The project was renamed to G-Images but **these were deliberately
not renamed** — `kaggle/02-train.py` finds its input by globbing
`**/gedit_images.bin`, and five existing prep outputs carry those names.

**9. MPS returns ZEROS instead of erroring when the Mac is out of memory.** This
one is vicious: no exception, no warning, just silently wrong numbers that look
plausible. It was caught only because `torch.randn(...).norm()` printed 0.0, which
is physically impossible. It had already invalidated a per-timestep measurement
AND made a working `black_and_white` render as pure colour noise — the model was
fine, the machine was not (76 MB free RAM, two headless Blender processes at 240%
CPU, load average 37 on an 8-core i9).

**Before trusting any local measurement, assert that your first random tensor has
non-zero norm**, and if the machine is loaded, run evaluation on CPU. It is far
slower and completely trustworthy. Do not kill other people's Blender/render
processes to free memory — another session is probably using them.

**10. macOS/MPS is a poor proxy and lacks float64.** The cosine schedule builds its
cumulative product on CPU in float64 and then moves — CUDA accepted the device
version, MPS raised outright, so the Kaggle run was fine while every local
evaluation would have died.

---

## How to run things

Kaggle is driven entirely through the API — no browser needed. The CLI lives in
the project venv and is invoked as a **module**:

```bash
cd ~/Downloads/Claude/Projects/AIe/G-Images
.venv/bin/python -m kaggle kernels status jerzysukiennik/gedit-train
```

**Continue training** (harvest → upload → push, in that order):

```bash
.venv/bin/python -m kaggle kernels output jerzysukiennik/gedit-train -p kaggle-run/out --file-pattern 'ckpt\.pt$'
cp kaggle-run/out/run/ckpt.pt kaggle-run/ckpt-dataset/ckpt.pt
.venv/bin/python -m kaggle datasets version -p kaggle-run/ckpt-dataset -m "step N" --dir-mode skip
# then WAIT for the dataset size to be non-zero, then:
.venv/bin/python -m kaggle kernels push -p kaggle-run/kernel-train
```

**Evaluate a checkpoint** — this is the acceptance test, and for the 13 synthetic
types it shows the model's output next to the **exact ground truth**:

```bash
.venv/bin/python runtime/type_grid.py --ckpt <ckpt> --image <photo> --types black_and_white add_hat add_cat season_winter --guidance 3.0
```

**Fetch sidecars without downloading gigabytes** — `file_pattern` is a **regex,
not a glob**:

```python
api.kernels_output("jerzysukiennik/gedit-anyedit-0", path=d, file_pattern=r".*\.json$")
```

`kaggle-run/` is gitignored and holds checkpoints, kernel metadata and harvested
outputs. Kernel scripts there are **thin bootstraps** that clone the repo and run
`kaggle/*.py` from it, so the repo is the single source of truth — deliberately
unlike G-Micro, where a full copy of the training script lives beside the kernel
and a real bug once survived in the gap between the two.

---

## Working agreements

- **GPU quota is 30h/week and shared** with G-Micro and G-Mini. Check
  `kernels status` before launching anything. A full 40000-step run is ~20h.
- **Do not set up a second Kaggle account to reset the quota.** Jurek has asked
  three times, in different framings; it violates Kaggle's terms. The honest
  options are waiting for the weekly reset or renting a GPU (tens of złoty for
  this model). One of those asks came with "I accidentally deleted the account" —
  the API showed the account alive and holding every artifact, and it turned out
  to be a browser-side problem.
- **Jurek does not touch code or the terminal.** Claude does files, git, Kaggle,
  deploys.
- **Sessions end cleanly on purpose.** `SESSION_STEPS` caps each run below the
  point where the job used to thrash and get OOM-killed — two runs died the same
  way, each burning ~2.3h of quota computing nothing.
- Code and commits in English, conversation in Polish.

---

## The habit that actually moved this project

Every real advance here came from **measuring something cheap before spending GPU
on an assumption**, and every long detour came from skipping that:

- Five GPU sessions went into "more steps" before anyone spent four local minutes
  checking whether sampling was done correctly. It wasn't — guidance was at 1.0.
- Two more went into conditioning dropout before anyone counted how many "black
  and white" examples the corpus contained. 588.
- Object addition was declared impossible from InstructPix2Pix on the basis of a
  two-minute count (109 hats), which turned out to be right and sent us to
  AnyEdit, where `add` alone is larger than the entire IP2P corpus.

Loss is a poor guide here specifically: it flattened around step 300 while visual
quality kept climbing for thousands of steps, and it looked healthy at step 94200
while the model was ignoring its conditioning completely. **Judge checkpoints with
`type_grid.py`, against the ground-truth column.**
