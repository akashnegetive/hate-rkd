"""Context-privileged knowledge distillation (CP-KD).

Problem
-------
Whether a reply is hateful often depends on the comment it answers ("So true, 100% agree"
is hateful under a hateful parent and harmless under a kind one). Context-aware models need
the conversation thread at inference time, which is slower (longer input) and often not
available. The base paper (Gui et al. 2026) classifies every post in isolation and names
"content lacking sufficient contextual information" as an open problem.

Idea
----
* The **teacher** (BERT) reads "parent [SEP] reply". During training the parent is hidden
  with probability `context_dropout`, so the teacher can also judge a reply on its own.
* For every training reply the teacher is scored twice:
      p_ctx   = sigmoid(teacher(parent, reply))      what the label is, given the thread
      p_alone = sigmoid(teacher("", reply))          what it looks like in isolation
      delta   = |p_ctx - p_alone|                     how much the thread changes the verdict
* The **student** (DistilBERT) reads only the reply and has two outputs:
      hate logit z   learns the context-aware verdict p_ctx (reliability-gated KD, + focal on
                     the gold labels, which annotators gave *with* the parent visible)
      ctx  logit c   learns delta, i.e. "does this reply need its context?"
* At inference the student is as fast as a plain DistilBERT on the reply, and it can say
  "hate", "not hate", or "this one depends on the conversation - check the parent".

Data (downloaded from the authors' GitHub repositories)
------------------------------------------------------
rc   Reddit hate / counter-speech / neutral replies with their parent comment
     (Yu, Blanco & Hong, NAACL 2022). Binary here: hate vs. not hate.
ccc  Civil Comments in Context (Xenos et al. 2021/2022): every reply has a toxicity label given
     WITH the parent (used for training/testing) and one given WITHOUT it. Replies whose label
     flips are human ground truth for "needs context".
smoke_ctx  synthetic conversations to test the code offline.
"""
import copy
import json
import math
import os
import random
import re
import time
import urllib.request

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup

from .augment import char_perturb, round_two
from .losses import class_weights, hard_loss, student_loss
from .metrics import classification_metrics, sigmoid
from .train import device

RC_URL = "https://raw.githubusercontent.com/xinchenyu/counter_context/main/data/gold/{}.jsonl"
# The repository does not document the label ids. Reading examples, "0" are hateful replies,
# "1" neutral and "2" counter-speech. VERIFY against Yu et al. (2022) before reporting.
RC_HATE_LABEL = "0"
CCC_URL = "https://raw.githubusercontent.com/ipavlopoulos/context_toxicity/master/data/CCC.csv"

# Steps of the context ladder (run-name -> settings). Each adds one thing to the previous.
CONTEXT_METHODS = {
    # C0: DistilBERT on the reply alone, gold labels only
    "student_only": dict(use_kd=False, teacher_context=False, kd_gate=False, ctx_head=False),
    # C1: the paper's KD, teacher also sees only the reply
    "kd_reply_teacher": dict(use_kd=True, teacher_context=False, kd_gate=False, ctx_head=False),
    # C2: KD from a teacher that reads the parent (context-privileged teacher)
    "kd_ctx_teacher": dict(use_kd=True, teacher_context=True, kd_gate=False, ctx_head=False),
    # C3: + reliability gate (our earlier contribution)
    "kd_ctx_gate": dict(use_kd=True, teacher_context=True, kd_gate=True, ctx_head=False),
    # C4: + "needs context" head = CP-KD (final)
    "cpkd": dict(use_kd=True, teacher_context=True, kd_gate=True, ctx_head=True),
    # U1: upper bound - the student also reads the parent (needs the thread at inference)
    "student_with_context": dict(use_kd=True, teacher_context=True, student_context=True,
                                 kd_gate=True, ctx_head=False),
}
BASE = dict(preprocess="remove", use_r1=True, use_r2=True, hard_loss="focal",
            kd_revise=False, kd_cb=False, student_context=False)


# ===========================================================================
# data
# ===========================================================================
def _download(url, path):
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        urllib.request.urlretrieve(url, path)
    return path


def _codes_mean(s):
    nums = [float(x) for x in re.findall(r"-?\d+(?:\.\d+)?", str(s))]
    return float(np.mean(nums)) if nums else np.nan


def load_rc(data_dir="data"):
    out = {}
    for split, fname in (("train", "train"), ("val", "val"), ("test", "test")):
        path = _download(RC_URL.format(fname), os.path.join(data_dir, "reddit_context", f"{fname}.jsonl"))
        rows = [json.loads(line) for line in open(path, encoding="utf-8")]
        out[split] = {"text": [r["target"] for r in rows], "context": [r["context"] for r in rows],
                      "label": [int(str(r["label"]) == RC_HATE_LABEL) for r in rows]}
    return out


def load_ccc(data_dir="data", seed=42):
    df = pd.read_csv(_download(CCC_URL, os.path.join(data_dir, "ccc", "CCC.csv")))
    ic, oc = df["tox_codes_ic"].map(_codes_mean), df["tox_codes_oc"].map(_codes_mean)
    df = df.assign(label=(ic > 0.5).astype(int), label_oc=(oc > 0.5).astype(int)).dropna(subset=["text", "parent"])
    df["flip"] = (df["label"] != df["label_oc"]).astype(int)
    strat = df["label"] * 2 + df["flip"]
    tr, rest = train_test_split(df, test_size=0.2, stratify=strat, random_state=seed)
    va, te = train_test_split(rest, test_size=0.5, stratify=rest["label"] * 2 + rest["flip"], random_state=seed)
    pack = lambda d: {"text": d["text"].astype(str).tolist(), "context": d["parent"].astype(str).tolist(),  # noqa: E731
                      "label": d["label"].tolist(), "label_oc": d["label_oc"].tolist(), "flip": d["flip"].tolist()}
    return {"train": pack(tr), "val": pack(va), "test": pack(te)}


_HATE_PARENTS = ["immigrants are vermin and should all be deported", "women are too stupid to vote",
                 "muslims are a disease on this country", "refugees are animals"]
_KIND_PARENTS = ["my immigrant neighbours helped me move today", "women scientists won the prize",
                 "the muslim charity fed hundreds of people", "refugees opened a lovely bakery"]
_ENDORSE = ["so true, 100% agree", "exactly what i was thinking", "finally someone said it", "this. so much this"]
_COUNTER = ["that is a horrible thing to say", "no, this is just wrong and hateful", "please stop spreading hate"]
_HATE_REPLY = ["they are disgusting parasites", "send them all back now", "subhuman trash all of them"]
_NEUTRAL = ["what time is the game tonight", "i had pasta for lunch", "nice weather today"]


def load_smoke_ctx(n=2000, seed=0):
    """Synthetic threads: endorsing replies are hateful ONLY under a hateful parent."""
    rng = random.Random(seed)
    rows = []
    for _ in range(n):
        hateful_parent = rng.random() < 0.5
        parent = rng.choice(_HATE_PARENTS if hateful_parent else _KIND_PARENTS)
        kind = rng.choices(["endorse", "counter", "hate", "neutral"], [0.35, 0.2, 0.2, 0.25])[0]
        reply = rng.choice({"endorse": _ENDORSE, "counter": _COUNTER, "hate": _HATE_REPLY, "neutral": _NEUTRAL}[kind])
        label = int(kind == "hate" or (kind == "endorse" and hateful_parent))
        label_oc = int(kind == "hate")                     # judged alone, "so true" is not hateful
        rows.append((reply, parent, label, label_oc, int(label != label_oc)))
    a, b = int(0.7 * n), int(0.8 * n)
    pack = lambda r: {"text": [x[0] for x in r], "context": [x[1] for x in r], "label": [x[2] for x in r],  # noqa: E731
                      "label_oc": [x[3] for x in r], "flip": [x[4] for x in r]}
    return {"train": pack(rows[:a]), "val": pack(rows[a:b]), "test": pack(rows[b:])}


def load_context_splits(cfg):
    if cfg.dataset == "rc":
        return load_rc(cfg.data_dir)
    if cfg.dataset == "ccc":
        return load_ccc(cfg.data_dir)
    if cfg.dataset == "smoke_ctx":
        return load_smoke_ctx()
    raise ValueError(cfg.dataset)


def preprocess_split(split, prep):
    out = dict(split)
    out["text"] = [prep(t) for t in split["text"]]
    out["context"] = [prep(c) for c in split["context"]]
    return out


def round_one_pairs(split, rho, pi, seed):
    """Algorithm 1 for pairs: typo copies of hateful replies; the parent is copied unchanged."""
    rng = random.Random(seed)
    pool = [i for i, y in enumerate(split["label"]) if y == 1] or list(range(len(split["label"])))
    out = {k: list(v) for k, v in split.items()}
    for _ in range(int(len(split["text"]) * rho)):
        i = rng.choice(pool)
        for k in out:
            out[k].append(char_perturb(split["text"][i], pi, rng) if k == "text" else split[k][i])
    return out


# ===========================================================================
# encoding / prediction
# ===========================================================================
def encode(tokenizer, texts, max_len, contexts=None):
    """Reply only, or the pair "[CLS] parent [SEP] reply [SEP]" when contexts are given."""
    if contexts is None:
        return tokenizer(list(texts), truncation=True, max_length=max_len, padding=True, return_tensors="pt")
    return tokenizer(list(contexts), list(texts), truncation="longest_first", max_length=max_len,
                     padding=True, return_tensors="pt")


def _forward(model, batch):
    inputs = {k: v for k, v in batch.items() if k in ("input_ids", "attention_mask", "token_type_ids")}
    return model(**inputs).logits.float()


@torch.no_grad()
def predict(model, tokenizer, texts, max_len, contexts=None, bs=64):
    """Raw logits, shape (n, num_outputs)."""
    dev = device()
    model.eval().to(dev)
    out = []
    for i in range(0, len(texts), bs):
        ctx = None if contexts is None else contexts[i:i + bs]
        b = {k: v.to(dev) for k, v in encode(tokenizer, texts[i:i + bs], max_len, ctx).items()}
        with torch.autocast(dev.type, dtype=torch.float16, enabled=dev.type == "cuda"):
            out.append(_forward(model, b).cpu())
    return torch.cat(out).numpy()


# ===========================================================================
# training
# ===========================================================================
def _fit(cfg, model, tokenizer, n_train, batch_size, make_batch, loss_fn, val_score, log):
    """Generic loop: AdamW + linear warm-up, fp16 on GPU, Round-2 masking, best-val checkpoint."""
    dev = device()
    model.to(dev)
    loader = DataLoader(list(range(n_train)), batch_size=batch_size, shuffle=True,
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
    best, best_state, history = -1.0, None, []
    for epoch in range(cfg.epochs):
        model.train()
        t0, losses = time.time(), []
        for idx in loader:
            idx = idx.tolist()
            batch = make_batch(idx)
            if cfg.use_r2:
                batch["input_ids"] = round_two(batch["input_ids"], batch["attention_mask"],
                                               tokenizer.unk_token_id, special, cfg.pi, gen)
            batch = {k: v.to(dev) for k, v in batch.items()}
            with torch.autocast(dev.type, dtype=torch.float16, enabled=use_amp):
                logits = _forward(model, batch)
            loss = loss_fn(logits, idx)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            scaler.step(opt)
            scaler.update()
            sched.step()
            losses.append(loss.item())
        score = val_score(model)
        history.append({"epoch": epoch + 1, "train_loss": float(np.mean(losses)), "val_macro_f1": score})
        log(f"  epoch {epoch + 1}/{cfg.epochs} loss={np.mean(losses):.4f} val_macro_f1={score:.4f} "
            f"({time.time() - t0:.0f}s)")
        if score > best:
            best, best_state = score, copy.deepcopy({k: v.detach().cpu() for k, v in model.state_dict().items()})
    model.load_state_dict(best_state)
    return model, history


def train_teacher(cfg, model, tok, train, val, log=print):
    """Teacher: hard labels (focal); reads the parent unless teacher_context is False."""
    y = torch.tensor(train["label"])
    rng = random.Random(cfg.seed)
    use_ctx = cfg.teacher_context
    max_len = cfg.ctx_max_len if use_ctx else cfg.max_len

    def make_batch(idx):
        texts = [train["text"][i] for i in idx]
        if not use_ctx:
            return encode(tok, texts, max_len)
        # context dropout: sometimes hide the parent, so the teacher also learns to judge alone
        ctx = ["" if rng.random() < cfg.context_dropout else train["context"][i] for i in idx]
        return encode(tok, texts, max_len, ctx)

    def loss_fn(logits, idx):
        return hard_loss(cfg, logits[:, 0], y[idx].to(logits.device))

    def val_score(model):
        z = predict(model, tok, val["text"], max_len, val["context"] if use_ctx else None)[:, 0]
        return classification_metrics(val["label"], z)["macro_f1"]

    return _fit(cfg, model, tok, len(train["text"]), cfg.teacher_bs, make_batch, loss_fn, val_score, log)


def teacher_signals(cfg, teacher, tok, split):
    """Teacher logits with the parent, without it, and delta = |p_ctx - p_alone|."""
    if not cfg.teacher_context:
        z = predict(teacher, tok, split["text"], cfg.max_len)[:, 0]
        return z, z, np.zeros_like(z)
    z_ctx = predict(teacher, tok, split["text"], cfg.ctx_max_len, split["context"])[:, 0]
    z_alone = predict(teacher, tok, split["text"], cfg.ctx_max_len, [""] * len(split["text"]))[:, 0]
    return z_ctx, z_alone, np.abs(sigmoid(z_ctx) - sigmoid(z_alone))


def train_student(cfg, model, tok, train, val, t_logits=None, delta=None, log=print):
    """Student: reads the reply only (unless student_context). Output 0 = hate, output 1 = needs-context."""
    y = torch.tensor(train["label"])
    cw = class_weights(y)
    tl = torch.as_tensor(t_logits, dtype=torch.float) if t_logits is not None else None
    dl = torch.as_tensor(delta, dtype=torch.float) if delta is not None else None
    use_ctx = cfg.student_context
    max_len = cfg.ctx_max_len if use_ctx else cfg.max_len

    def make_batch(idx):
        texts = [train["text"][i] for i in idx]
        return encode(tok, texts, max_len, [train["context"][i] for i in idx] if use_ctx else None)

    def loss_fn(logits, idx):
        yy = y[idx].to(logits.device)
        if tl is not None and cfg.use_kd:
            loss, _ = student_loss(cfg, logits[:, 0], tl[idx].to(logits.device), yy, cw)
        else:
            loss = hard_loss(cfg, logits[:, 0], yy)
        if cfg.ctx_head and dl is not None:
            loss = loss + cfg.ctx_weight * F.binary_cross_entropy_with_logits(logits[:, 1], dl[idx].to(logits.device))
        return loss

    def val_score(model):
        z = predict(model, tok, val["text"], max_len, val["context"] if use_ctx else None)[:, 0]
        return classification_metrics(val["label"], z)["macro_f1"]

    return _fit(cfg, model, tok, len(train["text"]), cfg.student_bs, make_batch, loss_fn, val_score, log)


# ===========================================================================
# evaluation
# ===========================================================================
def _flag_quality(flip, score):
    flip = np.asarray(flip)
    if flip.min() == flip.max():
        return None
    return {"auroc": float(roc_auc_score(flip, score)), "ap": float(average_precision_score(flip, score)),
            "prevalence": float(flip.mean())}


def _deferral(labels, logits, score, fracs=(0.05, 0.10, 0.20)):
    """Send the top-`frac` most flagged replies to a human; metrics on the rest."""
    labels, logits, score = np.asarray(labels), np.asarray(logits), np.asarray(score)
    out = {}
    for f in fracs:
        keep = np.argsort(score)[: int(round(len(score) * (1 - f)))]
        m = classification_metrics(labels[keep], logits[keep])
        out[f"{int(f * 100)}%"] = {"acc": m["acc"], "macro_f1": m["macro_f1"]}
    return out


@torch.no_grad()
def _latency(model, tok, texts, max_len, contexts=None, n=200):
    dev = device()
    model.eval().to(dev)
    texts = (texts * (n // max(len(texts), 1) + 1))[:n]
    contexts = None if contexts is None else (contexts * (n // max(len(contexts), 1) + 1))[:n]
    run = lambda i: _forward(model, {k: v.to(dev) for k, v in encode(  # noqa: E731
        tok, [texts[i]], max_len, None if contexts is None else [contexts[i]]).items()})
    for i in range(10):
        run(i)
    if dev.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for i in range(len(texts)):
        run(i)
    if dev.type == "cuda":
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) * 1000 / len(texts)


def evaluate_student(cfg, model, tok, test, teacher=None, t_tok=None):
    use_ctx = cfg.student_context
    max_len = cfg.ctx_max_len if use_ctx else cfg.max_len
    out = predict(model, tok, test["text"], max_len, test["context"] if use_ctx else None)
    z = out[:, 0]
    labels = np.asarray(test["label"])
    p = sigmoid(z)
    res = {"test": classification_metrics(labels, z), "test_logits": z.tolist(), "test_labels": labels.tolist()}
    uncertainty = 1 - np.abs(2 * p - 1)                       # baseline flag: "the model is unsure"
    scores = {"uncertainty": uncertainty}
    if cfg.ctx_head and out.shape[1] > 1:
        scores["needs_context_head"] = sigmoid(out[:, 1])

    if teacher is not None and cfg.teacher_context:
        z_ctx, z_alone, delta = teacher_signals(cfg, teacher, t_tok, test)
        res["teacher_with_context"] = classification_metrics(labels, z_ctx)
        res["teacher_reply_only"] = classification_metrics(labels, z_alone)
        scores["teacher_delta_oracle"] = delta               # needs the parent at test time
        if "needs_context_head" in scores and np.std(delta) > 0:
            res["head_vs_teacher_delta_spearman"] = float(spearmanr(scores["needs_context_head"], delta)[0])

    if "flip" in test:
        flip = np.asarray(test["flip"])
        for name, m in (("slice_context_dependent", flip == 1), ("slice_context_free", flip == 0)):
            if m.sum() and len(set(labels[m])) > 1:
                res[name] = classification_metrics(labels[m], z[m])
            elif m.sum():
                res[name] = {"acc": float(((p[m] >= 0.5).astype(int) == labels[m]).mean()), "n": int(m.sum())}
        res["flag"] = {k: _flag_quality(flip, v) for k, v in scores.items()}
    res["deferral"] = {k: _deferral(labels, z, v) for k, v in scores.items()}

    if cfg.efficiency_eval:
        ctx = test["context"] if use_ctx else None
        enc = encode(tok, test["text"][:500], max_len, None if ctx is None else ctx[:500])
        res["efficiency"] = {
            "params_M": sum(q.numel() for q in model.parameters()) / 1e6,
            "latency_ms": _latency(model, tok, test["text"], max_len, ctx),
            "mean_input_tokens": float(enc["attention_mask"].sum(1).float().mean()),
            "device": device().type,
        }
    return res
