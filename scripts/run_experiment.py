"""Run one experiment (teacher and/or student) end to end.

Examples
--------
# paper reproduction on SemEval (trains RKD-T, then distils RKD-S)
python scripts/run_experiment.py --dataset se --method rkd

# our full method (reuses nothing - its teacher is trained on SPN text)
python scripts/run_experiment.py --dataset se --method ours

# ablation: our method without the reliability gate
python scripts/run_experiment.py --dataset se --method ours --kd_gate false --run_name se_ours_nogate

# offline test of the whole code path on CPU (about 1 minute)
python scripts/run_experiment.py --dataset smoke --method ours
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

from src.augment import round_one                                   # noqa: E402
from src.config import METHODS, PILOT, PRESETS, Config                     # noqa: E402
from src.data import load_cross_eval_set, load_dataset_splits       # noqa: E402
from src.evaluate import evaluate_all                               # noqa: E402
from src.models import build_model, build_tokenizer, load_finetuned  # noqa: E402
from src.preprocess import get_preprocessor                         # noqa: E402
from src.train import predict_logits, train_model                   # noqa: E402


def str2bool(v):
    return str(v).lower() in ("1", "true", "yes", "y")


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", default="ours", choices=list(METHODS))
    ap.add_argument("--stage", default="both", choices=["teacher", "student", "both"])
    ap.add_argument("--retrain_teacher", action="store_true")
    ap.add_argument("--save_student", action="store_true")
    # every Config field can be overridden from the command line
    for f in fields(Config):
        if f.name == "extra":
            continue
        typ = str2bool if f.type is bool else f.type
        ap.add_argument(f"--{f.name}", type=typ, default=None)
    return ap.parse_args()


def make_config(args):
    cfg = Config()
    dataset = args.dataset or cfg.dataset
    pilot = PILOT if args.pilot else {}
    for k, v in {**PRESETS[dataset], **METHODS[args.method], **pilot}.items():
        setattr(cfg, k, v)
    for f in fields(Config):
        v = getattr(args, f.name, None)
        if v is not None and f.name != "extra":
            setattr(cfg, f.name, v)
    if not cfg.run_name:
        cfg.run_name = f"{cfg.dataset}_f{cfg.fold}_{args.method}_s{cfg.seed}"
    return cfg


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def teacher_key(cfg):
    """Teachers that saw identical data + loss are shared between student variants."""
    tname = os.path.basename(cfg.teacher_name.rstrip("/"))
    return (f"{cfg.dataset}_f{cfg.fold}_{cfg.preprocess}_r1{int(cfg.use_r1)}_r2{int(cfg.use_r2)}"
            f"_{cfg.hard_loss}_a{cfg.alpha}_s{cfg.seed}_{tname}")


def main():
    args = parse_args()
    cfg = make_config(args)
    os.makedirs(os.path.join(cfg.out_dir, "results"), exist_ok=True)
    log_path = os.path.join(cfg.out_dir, "results", f"{cfg.run_name}.log")

    def log(msg):
        print(msg, flush=True)
        with open(log_path, "a") as fh:
            fh.write(msg + "\n")

    log(f"=== {cfg.run_name} ({time.strftime('%Y-%m-%d %H:%M')}) ===\n{cfg.to_json()}")
    set_seed(cfg.seed)

    # ---------------- Stage 1: denoising + Round-1 augmentation ------------
    data = load_dataset_splits(cfg)
    prep = get_preprocessor(cfg.preprocess)
    train = {"text": [prep(t) for t in data["train"]["text"]], "label": data["train"]["label"]}
    val = {"text": [prep(t) for t in data["val"]["text"]], "label": data["val"]["label"]}
    if cfg.use_r1:
        n0 = len(train["text"])
        train["text"], train["label"] = round_one(train["text"], train["label"], cfg.rho, cfg.pi,
                                                  cfg.seed, cfg.r1_minority_only)
        log(f"Round-1 augmentation: {n0} -> {len(train['text'])} training samples")
    log(f"train={len(train['text'])} (hate={sum(train['label'])})  val={len(val['text'])}  "
        f"test={len(data['test']['text'])}")
    cross = load_cross_eval_set(cfg) if cfg.cross_eval else None

    def save_result(role, history, ev):
        path = os.path.join(cfg.out_dir, "results", f"{cfg.run_name}__{role}.json")
        with open(path, "w") as fh:
            json.dump({"run_name": cfg.run_name, "method": args.method, "role": role,
                       "config": json.loads(cfg.to_json()), "history": history, "eval": ev}, fh, indent=1)
        log(f"saved {path}")

    # ---------------- Stage 2: teacher ------------------------------------
    t_dir = os.path.join(cfg.out_dir, "teachers", teacher_key(cfg))
    need_teacher = args.stage in ("teacher", "both") or cfg.use_kd
    if need_teacher:
        if os.path.exists(os.path.join(t_dir, "config.json")) and not args.retrain_teacher:
            log(f"[teacher] reusing {t_dir}")
            teacher, tok = load_finetuned(t_dir)
        else:
            log(f"[teacher] training {cfg.teacher_name} with {cfg.hard_loss} loss")
            tok = build_tokenizer(cfg.teacher_name, data["train"]["text"], f"{cfg.dataset}_f{cfg.fold}")
            teacher = build_model(cfg.teacher_name, tok)
            teacher, hist = train_model(cfg, teacher, tok, train, val, cfg.teacher_bs, log=log)
            teacher.save_pretrained(t_dir)
            tok.save_pretrained(t_dir)
            if args.stage in ("teacher", "both"):
                save_result("teacher", hist, evaluate_all(cfg, teacher, tok, data["test"], prep, cross, log))

    if args.stage == "teacher":
        return

    # ---------------- Stage 3: student -------------------------------------
    t_logits = None
    if cfg.use_kd:
        # soft labels from the frozen teacher on CLEAN sequences (no Round-2)
        t_logits = predict_logits(teacher, tok, train["text"], cfg.max_len)
        acc = float(((t_logits > 0).astype(int) == np.asarray(train["label"])).mean())
        log(f"[student] teacher train-acc={acc:.4f} (wrong soft labels: {1 - acc:.2%})")
        del teacher
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    log(f"[student] training {cfg.student_name}  kd={cfg.use_kd} gate={cfg.kd_gate} "
        f"revise={cfg.kd_revise} cb={cfg.kd_cb} lambda={cfg.lam} tau={cfg.tau}")
    s_tok = build_tokenizer(cfg.student_name, data["train"]["text"], f"{cfg.dataset}_f{cfg.fold}")
    student = build_model(cfg.student_name, s_tok)
    student, hist = train_model(cfg, student, s_tok, train, val, cfg.student_bs,
                                teacher_logits=t_logits, log=log)
    save_result("student", hist, evaluate_all(cfg, student, s_tok, data["test"], prep, cross, log))
    if args.save_student:
        student.save_pretrained(os.path.join(cfg.out_dir, "students", cfg.run_name))
        s_tok.save_pretrained(os.path.join(cfg.out_dir, "students", cfg.run_name))


if __name__ == "__main__":
    main()
