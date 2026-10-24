"""Full evaluation suite run after every training stage.

  clean        : official test split
  slices       : test samples WITH emojis/hashtags vs. WITHOUT (in the raw text)
                 -> shows where semantic-preserving normalisation matters
  perturbed    : test split after Round-1 typos + Round-2 [UNK] masking
                 (paper Sec. V-C robustness protocol, applied to the whole test set)
  cross        : the other dataset (SE model -> DV, DV model -> SE test)
  efficiency   : parameters, size on disk, latency per sample, peak memory
"""
import os
import random
import tempfile
import time

import numpy as np
import torch

from .augment import char_perturb, round_two
from .metrics import classification_metrics
from .preprocess import has_emoji_or_hashtag
from .train import _forward, device, encode, predict_logits


@torch.no_grad()
def predict_logits_perturbed(model, tokenizer, texts, max_len, pi, seed=0, bs=128):
    rng = random.Random(seed)
    texts = [char_perturb(t, pi, rng) for t in texts]
    gen = torch.Generator().manual_seed(seed)
    special = [i for i in tokenizer.all_special_ids if i != tokenizer.unk_token_id]
    dev = device()
    model.eval().to(dev)
    out = []
    for i in range(0, len(texts), bs):
        b = encode(tokenizer, texts[i:i + bs], max_len)
        b["input_ids"] = round_two(b["input_ids"], b["attention_mask"], tokenizer.unk_token_id,
                                   special, pi, gen)
        b = {k: v.to(dev) for k, v in b.items()}
        out.append(_forward(model, b).cpu())
    return torch.cat(out).numpy()


@torch.no_grad()
def efficiency(model, tokenizer, texts, max_len, n=200):
    dev = device()
    model.eval().to(dev)
    n_params = sum(p.numel() for p in model.parameters())
    with tempfile.TemporaryDirectory() as d:
        torch.save(model.state_dict(), os.path.join(d, "m.pt"))
        size_mb = os.path.getsize(os.path.join(d, "m.pt")) / 2 ** 20
    texts = (texts * (n // max(len(texts), 1) + 1))[:n]
    if dev.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    for t in texts[:10]:                                   # warm-up
        _forward(model, {k: v.to(dev) for k, v in encode(tokenizer, [t], max_len).items()})
    if dev.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for t in texts:                                        # batch size 1, as in the paper's IT
        _forward(model, {k: v.to(dev) for k, v in encode(tokenizer, [t], max_len).items()})
    if dev.type == "cuda":
        torch.cuda.synchronize()
    ms = (time.perf_counter() - t0) * 1000 / len(texts)
    out = {"params_M": n_params / 1e6, "size_MB": size_mb, "latency_ms": ms, "device": dev.type}
    if dev.type == "cuda":
        out["peak_mem_MB"] = torch.cuda.max_memory_allocated() / 2 ** 20
    return out


def evaluate_all(cfg, model, tokenizer, raw_test, prep, cross=None, log=print):
    """raw_test: {"text","label"} before preprocessing; prep: preprocessing function."""
    texts = [prep(t) for t in raw_test["text"]]
    labels = np.asarray(raw_test["label"])
    logits = predict_logits(model, tokenizer, texts, cfg.max_len)
    res = {"test": classification_metrics(labels, logits), "test_logits": logits.tolist(),
           "test_labels": labels.tolist()}

    mask = np.array([has_emoji_or_hashtag(t) for t in raw_test["text"]])
    for name, m in (("slice_emoji_hashtag", mask), ("slice_plain", ~mask)):
        if m.sum() > 0 and len(set(labels[m])) > 1:
            res[name] = classification_metrics(labels[m], logits[m])

    if cfg.robust_eval:
        pl = predict_logits_perturbed(model, tokenizer, texts, cfg.max_len, cfg.pi, seed=cfg.seed)
        res["perturbed"] = classification_metrics(labels, pl)

    if cfg.cross_eval and cross is not None:
        name, data = cross
        ct = [prep(t) for t in data["text"]]
        res[f"cross_{name}"] = classification_metrics(data["label"],
                                                      predict_logits(model, tokenizer, ct, cfg.max_len))

    if cfg.efficiency_eval:
        res["efficiency"] = efficiency(model, tokenizer, texts, cfg.max_len)

    t = res["test"]
    log(f"  TEST acc={t['acc']:.4f} macroF1={t['macro_f1']:.4f} wF1={t['weighted_f1']:.4f} "
        f"hateF1={t['hate_f1']:.4f} ECE={t['ece']:.4f}")
    return res
