"""Run one context-privileged KD experiment (see src/context_kd.py).

Examples
--------
python scripts/run_context.py --dataset rc  --method cpkd                 # final method, Reddit
python scripts/run_context.py --dataset ccc --method kd_reply_teacher     # paper-style KD baseline
python scripts/run_context.py --dataset smoke_ctx --method cpkd           # offline test (~1 min CPU)

Methods (each adds one thing): student_only -> kd_reply_teacher -> kd_ctx_teacher -> kd_ctx_gate
-> cpkd; student_with_context is the upper bound that needs the thread at inference.
"""
import argparse
import json
import os
import random
import sys
import time
from dataclasses import fields

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from src.config import PILOT, PRESETS, Config                                # noqa: E402
from src.context_kd import (BASE, CONTEXT_METHODS, evaluate_student, load_context_splits,  # noqa: E402
                            normalise_delta, predict, preprocess_split, round_one_pairs, teacher_signals,
                            train_student, train_teacher, _latency)
from src.metrics import classification_metrics                               # noqa: E402
from src.models import build_model, build_tokenizer, load_finetuned          # noqa: E402
from src.preprocess import get_preprocessor                                  # noqa: E402


def str2bool(v):
    return str(v).lower() in ("1", "true", "yes", "y")


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", default="cpkd", choices=list(CONTEXT_METHODS))
    ap.add_argument("--skip_done", action="store_true")
    ap.add_argument("--retrain_teacher", action="store_true")
    for f in fields(Config):
        if f.name != "extra":
            ap.add_argument(f"--{f.name}", type=str2bool if f.type is bool else f.type, default=None)
    return ap.parse_args()


def make_config(args):
    cfg = Config()
    dataset = args.dataset or "rc"
    pilot = dict(PILOT, max_len=64, ctx_max_len=128) if args.pilot else {}
    for k, v in {**PRESETS[dataset], **BASE, **CONTEXT_METHODS[args.method], **pilot}.items():
        setattr(cfg, k, v)
    for f in fields(Config):
        v = getattr(args, f.name, None)
        if v is not None and f.name != "extra":
            setattr(cfg, f.name, v)
    cfg.cross_eval = False
    if not cfg.run_name:
        cfg.run_name = f"{cfg.dataset}_{args.method}_f{cfg.fold}_s{cfg.seed}"
    return cfg


def teacher_key(cfg):
    tname = os.path.basename(cfg.teacher_name.rstrip("/"))
    ctx = f"ctx_cd{cfg.context_dropout}" if cfg.teacher_context else "reply"
    return (f"{cfg.dataset}_f{cfg.fold}_{ctx}_{cfg.preprocess}_r1{int(cfg.use_r1)}_r2{int(cfg.use_r2)}"
            f"_{cfg.hard_loss}_a{cfg.alpha}_s{cfg.seed}_{tname}")


def main():
    args = parse_args()
    cfg = make_config(args)
    res_dir = os.path.join(cfg.out_dir, "results")
    os.makedirs(res_dir, exist_ok=True)
    if args.skip_done and os.path.exists(os.path.join(res_dir, f"{cfg.run_name}__student.json")):
        print(f"[skip] {cfg.run_name} already finished", flush=True)
        return
    log_path = os.path.join(res_dir, f"{cfg.run_name}.log")

    def log(msg):
        print(msg, flush=True)
        with open(log_path, "a") as fh:
            fh.write(msg + "\n")

    def save_result(role, history, ev):
        path = os.path.join(res_dir, f"{cfg.run_name}__{role}.json")
        with open(path, "w") as fh:
            json.dump({"run_name": cfg.run_name, "method": args.method, "role": role,
                       "config": json.loads(cfg.to_json()), "history": history, "eval": ev}, fh, indent=1)
        log(f"saved {path}")

    log(f"=== {cfg.run_name} ({time.strftime('%Y-%m-%d %H:%M')}) ===\n{cfg.to_json()}")
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    # ---- data: replies + parents ------------------------------------------
    data = load_context_splits(cfg)
    prep = get_preprocessor(cfg.preprocess)
    train, val, test = (preprocess_split(data[s], prep) for s in ("train", "val", "test"))
    if cfg.use_r1:
        train = round_one_pairs(train, cfg.rho, cfg.pi, cfg.seed)
    log(f"train={len(train['text'])} (hate={sum(train['label'])})  val={len(val['text'])}  "
        f"test={len(test['text'])} (hate={sum(test['label'])}"
        + (f", context-dependent={sum(test['flip'])})" if "flip" in test else ")"))
    corpus = data["train"]["text"] + data["train"]["context"]
    tok_key = f"{cfg.dataset}_f{cfg.fold}"

    # ---- teacher ------------------------------------------------------------
    teacher = t_tok = None
    if cfg.use_kd:
        t_dir = os.path.join(cfg.teacher_dir or os.path.join(cfg.out_dir, "teachers"), teacher_key(cfg))
        t_eval = os.path.join(t_dir, "teacher_eval.json")
        if os.path.exists(os.path.join(t_dir, "config.json")) and not args.retrain_teacher:
            log(f"[teacher] reusing {t_dir}")
            teacher, t_tok = load_finetuned(t_dir)
            hist = json.load(open(t_eval))["history"] if os.path.exists(t_eval) else []
        else:
            log(f"[teacher] training {cfg.teacher_name}  reads parent: {cfg.teacher_context}  "
                f"context dropout: {cfg.context_dropout}")
            t_tok = build_tokenizer(cfg.teacher_name, corpus, tok_key)
            teacher = build_model(cfg.teacher_name, t_tok)
            teacher, hist = train_teacher(cfg, teacher, t_tok, train, val, log)
            teacher.save_pretrained(t_dir)
            t_tok.save_pretrained(t_dir)
        max_len = cfg.ctx_max_len if cfg.teacher_context else cfg.max_len
        z = predict(teacher, t_tok, test["text"], max_len, test["context"] if cfg.teacher_context else None)[:, 0]
        t_ev = {"test": classification_metrics(test["label"], z), "test_logits": z.tolist(),
                "test_labels": list(test["label"]),
                "efficiency": {"params_M": sum(q.numel() for q in teacher.parameters()) / 1e6,
                               "latency_ms": _latency(teacher, t_tok, test["text"], max_len,
                                                      test["context"] if cfg.teacher_context else None)}}
        if cfg.teacher_context:
            za = predict(teacher, t_tok, test["text"], max_len, [""] * len(test["text"]))[:, 0]
            t_ev["test_reply_only"] = classification_metrics(test["label"], za)
        with open(t_eval, "w") as fh:
            json.dump({"history": hist, "eval": t_ev}, fh)
        save_result("teacher", hist, t_ev)
        log(f"  TEACHER macroF1={t_ev['test']['macro_f1']:.4f}"
            + (f"  (same teacher without the parent: {t_ev['test_reply_only']['macro_f1']:.4f})"
               if "test_reply_only" in t_ev else ""))

    # ---- teacher signals on the training replies ---------------------------
    t_logits = delta = None
    if teacher is not None:
        z_ctx, z_alone, delta = teacher_signals(cfg, teacher, t_tok, train)
        t_logits = z_ctx
        acc = float(((z_ctx > 0).astype(int) == np.asarray(train["label"])).mean())
        log(f"[student] teacher train-acc={acc:.4f}  mean delta={delta.mean():.4f}  "
            f"share delta>0.3={np.mean(delta > 0.3):.3f}")
        if cfg.ctx_head and cfg.ctx_target_norm:
            delta = normalise_delta(delta)
            log(f"[student] needs-context target rescaled: mean={delta.mean():.4f}")

    # ---- student ------------------------------------------------------------
    log(f"[student] {cfg.student_name}  reads parent: {cfg.student_context}  kd={cfg.use_kd}  "
        f"gate={cfg.kd_gate}  needs-context head={cfg.ctx_head}")
    s_tok = build_tokenizer(cfg.student_name, corpus, tok_key)
    student = build_model(cfg.student_name, s_tok, num_labels=2 if cfg.ctx_head else 1)
    student, hist = train_student(cfg, student, s_tok, train, val, t_logits, delta if cfg.ctx_head else None, log)
    ev = evaluate_student(cfg, student, s_tok, test, teacher, t_tok)
    t = ev["test"]
    log(f"  TEST acc={t['acc']:.4f} macroF1={t['macro_f1']:.4f} hateF1={t['hate_f1']:.4f} ECE={t['ece']:.4f}")
    if "flag" in ev:
        log("  needs-context flag (AUROC vs human label flips): "
            + ", ".join(f"{k}={v['auroc']:.3f}" for k, v in ev["flag"].items() if v))
    save_result("student", hist, ev)


if __name__ == "__main__":
    main()
