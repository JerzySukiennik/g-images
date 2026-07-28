# G-Images

Drugi model w rodzinie [MicroG](https://github.com/JerzySukiennik/microg) — a local, from-scratch diffusion model for photo editing, trained by [Jurek](https://github.com/JerzySukiennik). "G" (Gzowo, same root as MicroG) + "edit" (what it does).

Voice-triggered from [Gzowo AI](../Gzowo%20AI): "take a photo and make it look like X" → local diffusion, no external API.

Full spec: [`SPEC.md`](SPEC.md). Kaggle training instructions: [`kaggle/README.md`](kaggle/README.md). Logo: [`Design/`](Design/).

## Layout

- `model/` — the U-Net (trained from scratch) + frozen CLIP text encoder + diffusion schedule
- `data/fetch_dataset.py` — builds the training set from `timbrooks/instructpix2pix-clip-filtered`
- `train/train.py` — the training loop
- `kaggle/` — notebook cells + instructions to run the above on Kaggle's free T4×2
- `runtime/` — ONNX export + local inference test, for integration into Gzowo AI's Node bridge

Status: spec approved, data/training pipeline written, not yet run on Kaggle.
