"""Unit tests for preprocessing, augmentation and losses.  Run:  python -m pytest -q"""
import os
import random
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.augment import char_perturb, round_one, round_two          # noqa: E402
from src.config import Config                                       # noqa: E402
from src.losses import class_weights, distillation_targets, focal_loss, student_loss  # noqa: E402
from src.metrics import expected_calibration_error                  # noqa: E402
from src.preprocess import denoise_remove, denoise_spn, segment_hashtag  # noqa: E402

PAPER_EX4 = ("The elderly folk of Britain grateful they will soon be gone really does signal "
             "the end. #StopImmigration #BanIslam #GetBritainBack")


# ---------------- preprocessing --------------------------------------------
def test_paper_denoiser_matches_table1():
    assert denoise_remove("Team john in this bitch &#128588;&#128588;") == "Team john in this bitch"
    assert denoise_remove("wow im a bitch????? nice to fucking know.") == "wow im a bitch? nice to fucking know."
    assert denoise_remove("@Ashleyjoyner1 fuck you pussy suckin bitch.") == "fuck you pussy suckin bitch."
    assert denoise_remove("Free lil cj bitch the wait is over ! #december8th") == "Free lil cj bitch the wait is over !"
    assert denoise_remove("Introvert level expert htt.ps://t.co/FPiS5O6fNV") == "Introvert level expert"


def test_paper_denoiser_loses_hate_signal_but_spn_keeps_it():
    assert "islam" not in denoise_remove(PAPER_EX4).lower()
    spn = denoise_spn(PAPER_EX4)
    assert "stop immigration" in spn and "ban islam" in spn and "get britain back" in spn


def test_emoji_to_text_and_mojibake():
    assert "enraged face" in denoise_spn("so ABSURD \U0001F621")
    assert "enraged face" in denoise_spn("so ABSURD ðŸ˜¡")           # mojibake repaired first
    assert denoise_spn("lol \U0001F602\U0001F602\U0001F602").count("face with tears of joy") == 1


def test_hashtag_segmentation():
    assert segment_hashtag("StopImmigration") == "stop immigration"
    assert segment_hashtag("buildthewall") == "build the wall"


# ---------------- augmentation ---------------------------------------------
def test_round_one_adds_rho_fraction_of_minority_copies():
    texts = [f"sample number {i} contains several words" for i in range(100)]
    labels = [1 if i < 20 else 0 for i in range(100)]
    t2, y2 = round_one(texts, labels, rho=0.1, pi=0.1, seed=0, minority_only=True)
    assert len(t2) == 110 and all(y == 1 for y in y2[100:])
    assert all(t not in texts for t in t2[100:])                    # copies really are perturbed


def test_char_perturb_keeps_word_count():
    rng = random.Random(0)
    s = "these immigrants should leave immediately"
    assert len(char_perturb(s, 0.5, rng).split()) == len(s.split())


def test_round_two_never_masks_special_or_padding():
    ids = torch.tensor([[101, 5, 6, 7, 8, 102, 0, 0]])
    mask = torch.tensor([[1, 1, 1, 1, 1, 1, 0, 0]])
    out = round_two(ids, mask, unk_id=100, special_ids=[0, 101, 102], pi=1.0)
    assert out.tolist() == [[101, 100, 100, 100, 100, 102, 0, 0]]


# ---------------- losses ---------------------------------------------------
def test_focal_reduces_to_weighted_ce_when_gamma_zero():
    logits, y = torch.tensor([0.3, -1.2, 2.0]), torch.tensor([1, 0, 1])
    fl = focal_loss(logits, y, alpha=0.5, gamma=0.0)
    ce = torch.nn.functional.binary_cross_entropy_with_logits(logits, y.float())
    assert torch.allclose(fl, 0.5 * ce, atol=1e-6)


def test_revision_flips_wrong_teacher_labels_only():
    cfg = Config(kd_revise=True, tau=0.3, revise_beta=0.6)
    t = torch.tensor([3.0, -3.0])          # teacher: hate, non-hate
    y = torch.tensor([0, 0])               # truth: both non-hate -> first is wrong
    tgt = distillation_targets(cfg, t, y)
    assert tgt[0] < 0.5                    # revised towards the truth
    assert torch.isclose(tgt[1], torch.sigmoid(t[1] / 0.3))   # correct one untouched


def test_gate_gives_unreliable_teacher_less_weight():
    cfg = Config(kd_gate=True, lam=0.8)
    s = torch.zeros(2)
    _, parts_good = student_loss(cfg, s, torch.tensor([4.0, 4.0]), torch.tensor([1, 1]))
    _, parts_bad = student_loss(cfg, s, torch.tensor([-4.0, -4.0]), torch.tensor([1, 1]))
    assert parts_good["lam_mean"] > 0.75 and parts_bad["lam_mean"] < 0.05


def test_class_weights_inverse_frequency():
    w = class_weights([0, 0, 0, 1])
    assert torch.allclose(w, torch.tensor([4 / 6, 2.0]))


def test_ece_zero_for_perfectly_calibrated():
    assert expected_calibration_error([1, 0], [1.0, 0.0]) == 0.0


# ---------------- hashtag dropout ------------------------------------------
def test_hashtag_dropout_marks_and_drops_only_hashtags():
    from src.preprocess import drop_marked, strip_marks
    t = "they keep coming #BuildThatWall #MAGA"
    marked = denoise_spn(t, mark=True)
    assert strip_marks(marked) == denoise_spn(t)                    # eval view unchanged
    rng = random.Random(0)
    assert drop_marked(marked, 1.0, rng) == "they keep coming"       # p=1 removes all hashtags
    assert strip_marks(drop_marked(marked, 0.0, rng)) == denoise_spn(t)


# ---------------- calibrated gate ------------------------------------------
def test_fit_temperature_recovers_overconfidence():
    import numpy as np
    from src.metrics import fit_temperature
    rng = np.random.default_rng(0)
    true_logit = rng.normal(0, 2, 20000)
    y = (rng.random(20000) < 1 / (1 + np.exp(-true_logit))).astype(int)
    t = fit_temperature(3.0 * true_logit, y)        # model is 3x over-confident
    assert abs(t - 3.0) < 0.2


def test_calibrated_gate_softens_overconfident_teacher():
    cfg = Config(kd_gate=True, lam=0.8)
    s, t, y = torch.zeros(1), torch.tensor([-2.0]), torch.tensor([1])   # teacher wrong
    _, raw = student_loss(cfg, s, t, y)
    cfg.gate_temp = 4.0
    _, cal = student_loss(cfg, s, t, y)
    assert cal["lam_mean"] > raw["lam_mean"]       # softer teacher -> less extreme gate
