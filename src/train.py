"""Training loops for the teacher (Stage 2) and the student (Stage 3).

Both use the same loop; the only differences are
  * the loss  : teacher -> hard loss (focal / CE)
                student -> hard loss + distillation loss (losses.student_loss)
  * the input : teacher soft labels are pre-computed ONCE on the clean
                (un-masked) training sequences - the teacher is frozen and the
                paper forbids Algorithm 2 on the teacher's input during
                distillation - while the student sees Round-2-masked input.
Model selection: the epoch with the best validation macro-F1 is kept.
"""
import copy
import math
import random
import time

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup

from .augment import round_two
from .losses import class_weights, hard_loss, student_loss
from .metrics import classification_metrics
from .preprocess import drop_marked


def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def encode(tokenizer, texts, max_len):
    return tokenizer(list(texts), truncation=True, max_length=max_len, padding=True,
                     return_tensors="pt")


def _forward(model, batch):
    inputs = {k: v for k, v in batch.items() if k in ("input_ids", "attention_mask", "token_type_ids")}
    return model(**inputs).logits.squeeze(-1).float()


@torch.no_grad()
def predict_logits(model, tokenizer, texts, max_len, bs=128):
    """Logits for a list of texts (no augmentation)."""
    dev = device()
    model.eval().to(dev)
    out = []
    for i in range(0, len(texts), bs):
        batch = {k: v.to(dev) for k, v in encode(tokenizer, texts[i:i + bs], max_len).items()}
        with torch.autocast(dev.type, dtype=torch.float16, enabled=dev.type == "cuda"):
            out.append(_forward(model, batch).cpu())
    return torch.cat(out).numpy() if out else np.zeros(0)


def train_model(cfg, model, tokenizer, train, val, batch_size, teacher_logits=None, log=print):
    """Fine-tune `model` on train = {"text","label"}; returns (best_model, history)."""
    dev = device()
    model.to(dev)
    texts, labels = train["text"], torch.tensor(train["label"])
    tl =torch.as_tensor(teacher_logits, dtype=torch.float) if teacher_logits is not None else None
    cw = class_weights(labels)

    loader = DataLoader(list(range(len(texts))), batch_size=batch_size, shuffle=True,
                        generator=torch.Generator().manual_seed(cfg.seed))
    no_decay = ("bias", "LayerNorm.weight", "layer_norm.weight")
    params = [{"params": [p for n, p in model.named_parameters() if not any(x in n for x in no_decay)],
               "weight_decay": cfg.weight_decay},
              {"params": [p for n, p in model.named_parameters() if any(x in n for x in no_decay)],
               "weight_decay": 0.0}]
    opt = torch.optim.AdamW(params, lr=cfg.lr)
    total = cfg.epochs * len(loader)
    sched = get_linear_schedule_with_warmup(opt, math.ceil(cfg.warmup_ratio * total), total)
    use_amp = cfg.fp16 and dev.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    special = [i for i in tokenizer.all_special_ids if i != tokenizer.unk_token_id]
    gen = torch.Generator().manual_seed(cfg.seed)
    text_rng = random.Random(cfg.seed)

    best_f1, best_state, history = -1.0, None, []
    for epoch in range(cfg.epochs):
        model.train()
        t0, running = time.time(), []
        for idx in loader:
            # hashtag dropout (ours): marked hashtags are removed w.p. p, re-sampled every batch
            batch_texts = [texts[i] for i in idx]
            if cfg.hashtag_dropout > 0:
                batch_texts = [drop_marked(t, cfg.hashtag_dropout, text_rng) for t in batch_texts]
            batch = encode(tokenizer, batch_texts, cfg.max_len)
            if cfg.use_r2:                                       # Round-2 augmentation (online)
                batch["input_ids"] = round_two(batch["input_ids"], batch["attention_mask"],
                                               tokenizer.unk_token_id, special, cfg.pi, gen)
            batch = {k: v.to(dev) for k, v in batch.items()}
            y = labels[idx].to(dev)
            with torch.autocast(dev.type, dtype=torch.float16, enabled=use_amp):
                logits = _forward(model, batch)
            if tl is not None:
                loss, parts = student_loss(cfg, logits, tl[idx].to(dev), y, cw)
            else:
                loss, parts = hard_loss(cfg, logits, y), {}
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            scaler.step(opt)
            scaler.update()
            sched.step()
            running.append(loss.item())

        val_logits = predict_logits(model, tokenizer, val["text"], cfg.max_len)
        m = classification_metrics(val["label"], val_logits)
        history.append({"epoch": epoch + 1, "train_loss": float(np.mean(running)), **{f"val_{k}": v for k, v in m.items()}})
        log(f"  epoch {epoch + 1}/{cfg.epochs} loss={np.mean(running):.4f} "
            f"val_acc={m['acc']:.4f} val_macro_f1={m['macro_f1']:.4f} ({time.time() - t0:.0f}s)")
        if m["macro_f1"] > best_f1:
            best_f1 = m["macro_f1"]
            best_state = copy.deepcopy({k: v.detach().cpu() for k, v in model.state_dict().items()})

    model.load_state_dict(best_state)
    return model, history
