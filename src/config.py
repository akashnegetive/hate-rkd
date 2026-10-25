"""Experiment configuration.

One `Config` object carries every hyper-parameter of a run. Dataset presets
reproduce the settings reported in Gui et al. (IEEE TCSS 2026), Sec. IV-D:

    (rho, pi)                       = (0.1, 0.1)    dual-round augmentation
    (alpha, gamma, tau, lambda)     = (0.2, 2, 0.3, 0.8)   SE
                                    = (0.6, 2, 0.3, 0.8)   DV
    lr = 1e-5, batch 32 (teacher) / 64 (student)
"""
from dataclasses import dataclass, field, asdict
import json


@dataclass
class Config:
    # ---- data -------------------------------------------------------------
    dataset: str = "se"              # se | dv | smoke
    fold: int = 0                    # DV uses 5-fold CV (fold 0..4)
    n_folds: int = 5
    data_dir: str = "data"           # optional local CSVs (see data.py)
    max_len: int = 96
    train_limit: int = 0             # >0: subsample train set (quick debugging)

    # ---- preprocessing ----------------------------------------------------
    # none   : raw text (BERT / DistilBERT baselines)
    # remove : paper's 6-step denoising (deletes emojis + hashtags)
    # spn    : OURS - semantic-preserving normalisation (emoji->words, #hashtag->words)
    preprocess: str = "spn"          # none | remove | spn | spn_emoji
    hashtag_dropout: float = 0.0     # OURS: drop each hashtag w.p. p during training (spn only)

    # ---- dual-round augmentation (paper Alg. 1 and Alg. 2) ----------------
    use_r1: bool = True              # offline character-level perturbation
    use_r2: bool = True              # online Bernoulli [UNK] masking
    rho: float = 0.1                 # R1: fraction of |S| to perturb
    pi: float = 0.1                  # perturbation probability (R1 and R2)
    r1_minority_only: bool = True    # R1 samples are drawn from the hate class

    # ---- models -----------------------------------------------------------
    teacher_name: str = "bert-base-uncased"
    student_name: str = "distilbert-base-uncased"
    pilot: bool = False              # True -> from-scratch CPU models (see PILOT below)

    # ---- losses -----------------------------------------------------------
    hard_loss: str = "focal"         # focal | ce
    alpha: float = 0.2               # focal alpha (weight of hate class)
    gamma: float = 2.0               # focal gamma
    tau: float = 0.3                 # distillation temperature
    lam: float = 0.8                 # lambda: weight of distillation loss (Eq. 14)
    use_kd: bool = True              # False -> student trained on hard labels only

    # ---- OUR distillation extensions (all False = paper's RKD) ------------
    kd_gate: bool = False            # reliability-gated per-sample lambda
    kd_revise: bool = False          # revise soft labels the teacher got wrong
    kd_cb: bool = False              # class-balanced distillation loss
    revise_beta: float = 0.6         # mixing weight of ground truth in revision
    gate_floor: float = 0.0          # minimum reliability (keeps some KD everywhere)
    gate_calibrate: bool = False     # OURS: temperature-scale the teacher (on val) before gating
    gate_temp: float = 1.0           # teacher temperature used by the gate (fitted if gate_calibrate)

    # ---- optimisation -----------------------------------------------------
    lr: float = 1e-5
    teacher_bs: int = 32
    student_bs: int = 64
    epochs: int = 5
    warmup_ratio: float = 0.06
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    seed: int = 42
    fp16: bool = True

    # ---- evaluation -------------------------------------------------------
    cross_eval: bool = True          # evaluate on the *other* dataset as well
    robust_eval: bool = True         # evaluate on a perturbed copy of the test set
    efficiency_eval: bool = True

    # ---- bookkeeping ------------------------------------------------------
    out_dir: str = "runs"
    run_name: str = ""
    extra: dict = field(default_factory=dict)

    def to_json(self):
        return json.dumps(asdict(self), indent=2)


PRESETS = {
    "se": dict(alpha=0.2, gamma=2.0, tau=0.3, lam=0.8),
    "dv": dict(alpha=0.6, gamma=2.0, tau=0.3, lam=0.8),
    # tiny, offline configuration used only to test the code end-to-end
    "smoke": dict(alpha=0.5, gamma=2.0, tau=0.3, lam=0.8, epochs=8, max_len=32,
                  teacher_bs=16, student_bs=16, lr=3e-3, fp16=False,
                  teacher_name="smoke-teacher", student_name="smoke-student"),
}


# CPU pilot study: no pretrained weights / no GPU available.
# Same pipeline and losses; small transformers trained from scratch.
PILOT = dict(teacher_name="scratch-teacher", student_name="scratch-student",
             lr=5e-4, epochs=6, max_len=64, fp16=False)


# Named methods = presets of switches. Use them with --method.
METHODS = {
    # plain fine-tuning baselines (paper: "BERT" / "DistilBERT" rows of Table II)
    "baseline": dict(preprocess="none", use_r1=False, use_r2=False,
                     hard_loss="ce", use_kd=False),
    # faithful re-implementation of the paper (RKD-T / RKD-S)
    "rkd": dict(preprocess="remove", use_r1=True, use_r2=True, hard_loss="focal",
                use_kd=True, kd_gate=False, kd_revise=False, kd_cb=False),
    # FINAL method (SR-KD): emoji->text + reliability-gated distillation.
    # Chosen from the pilot ablations: revision and class-balanced KD did not help.
    "srkd": dict(preprocess="spn_emoji", use_r1=True, use_r2=True, hard_loss="focal",
                 use_kd=True, kd_gate=True, kd_revise=False, kd_cb=False),
    # SR-KD with a calibrated gate: the gate reads a temperature-scaled teacher
    "srkd_cal": dict(preprocess="spn_emoji", use_r1=True, use_r2=True, hard_loss="focal",
                     use_kd=True, kd_gate=True, gate_calibrate=True, kd_revise=False, kd_cb=False),
    # first version: all three KD switches + emoji AND hashtag normalisation
    "ours": dict(preprocess="spn", use_r1=True, use_r2=True, hard_loss="focal",
                 use_kd=True, kd_gate=True, kd_revise=True, kd_cb=True),
    # ours with emoji->text only (hashtags removed as in the paper)
    "ours_emoji": dict(preprocess="spn_emoji", use_r1=True, use_r2=True, hard_loss="focal",
                       use_kd=True, kd_gate=True, kd_revise=True, kd_cb=True),
    # ours + hashtag dropout (shortcut regularisation for hashtags)
    "ours_hd": dict(preprocess="spn", hashtag_dropout=0.5, use_r1=True, use_r2=True,
                    hard_loss="focal", use_kd=True, kd_gate=True, kd_revise=True, kd_cb=True),
}
