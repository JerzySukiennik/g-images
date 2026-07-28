"""Frozen pretrained CLIP text encoder — the one part of G-Images NOT trained
from scratch (see SPEC.md #2: hand-training a text encoder good enough to
understand arbitrary edit instructions is a separate project the size of
MicroG itself).

Returns the full per-token sequence (last_hidden_state), not the pooled
projection — cross-attention (model/unet.py) needs one vector PER WORD to
know which token refers to which part of the image ("hat" -> the head
region). A single pooled vector (what CLIPTextModelWithProjection gives)
carries no per-word/spatial information, which is exactly why the earlier
FiLM-only design (see git history) couldn't localize object edits like
"add a hat".
"""

import torch
from transformers import CLIPTokenizer, CLIPTextModel

MODEL_NAME = "openai/clip-vit-base-patch32"
SEQ_LEN = 32  # fixed padded length; edit instructions are short, 77 (CLIP's
              # max) would be mostly wasted padding in the packed dataset.


class ClipTextEncoder:
    def __init__(self, device="cpu"):
        self.device = device
        self.tokenizer = CLIPTokenizer.from_pretrained(MODEL_NAME)
        self.model = CLIPTextModel.from_pretrained(MODEL_NAME).to(device).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.embed_dim = self.model.config.hidden_size  # 512 for ViT-B/32
        self.seq_len = SEQ_LEN

    @torch.no_grad()
    def encode(self, prompts, batch_size=64):
        """prompts: list[str] -> float32 tensor [N, SEQ_LEN, embed_dim]."""
        out = []
        for i in range(0, len(prompts), batch_size):
            batch = prompts[i:i + batch_size]
            tok = self.tokenizer(batch, padding="max_length", truncation=True,
                                  max_length=self.seq_len, return_tensors="pt").to(self.device)
            seq = self.model(**tok).last_hidden_state
            out.append(seq.cpu())
        return torch.cat(out, dim=0).float()
