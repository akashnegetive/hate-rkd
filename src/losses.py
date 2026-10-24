"""Loss functions.

All models output ONE logit per sample (v for the teacher, z for the student);
sigmoid turns it into P(hate). Notation follows the paper:

    p      = sigmoid(v)          teacher probability of "hate"
    p_bar  = p if y==1 else 1-p  probability given to the TRUE class  (Eq. 3)

Paper (RKD)
    focal loss          L_tch / L_std = -alpha_i (1-p_bar)^gamma log p_bar        (Eq. 4, 9)
    distillation loss   L_dst = CE( sigmoid(v/tau), sigmoid(z/tau) )             (Eq. 13)
    total               L_ov  = lambda * L_dst + (1-lambda) * L_std               (Eq. 14)

Ours (three switches, each can be turned on separately for ablation)
    kd_revise : if the teacher is WRONG on a training sample, its soft label is
                pulled towards the ground truth:
                    t_i = beta * y_i + (1-beta) * p_tau_i
                (beta >= 0.5 guarantees the revised label points to the true class)
    kd_gate   : per-sample lambda driven by teacher reliability
                    r_i      = p_bar_i  (teacher's prob. of the true class, tau = 1)
                    lambda_i = lambda * max(r_i, floor)
                so an unsure / wrong teacher hands the sample back to the hard label.
    kd_cb     : class-balanced distillation term; minority-class samples get
                weight w_c = N / (2 N_c) (inverse frequency), because the paper
                applies the imbalance weight alpha only to the hard-label loss.

    L_i = lambda_i * w_i * L_dst,i + (1 - lambda_i) * L_std,i
"""
import torch
import torch.nn.functional as F


def focal_loss(logits, labels, alpha: float, gamma: float, reduction="mean"):
    """Binary focal loss (Eq. 4 / Eq. 9)."""
    labels = labels.float()
    log_p = F.logsigmoid(logits)           # log p
    log_1mp = F.logsigmoid(-logits)        # log (1-p)
    log_pbar = labels * log_p + (1 - labels) * log_1mp
    pbar = log_pbar.exp()
    alpha_i = labels * alpha + (1 - labels) * (1 - alpha)          # Eq. 5
    loss = -alpha_i * (1 - pbar).pow(gamma) * log_pbar
    return loss.mean() if reduction == "mean" else loss


def ce_loss(logits, labels, reduction="mean"):
    return F.binary_cross_entropy_with_logits(logits, labels.float(), reduction=reduction)


def hard_loss(cfg, logits, labels, reduction="mean"):
    if cfg.hard_loss == "focal":
        return focal_loss(logits, labels, cfg.alpha, cfg.gamma, reduction)
    return ce_loss(logits, labels, reduction)


def class_weights(labels):
    """Inverse-frequency weights w_c = N / (2 N_c) for c in {0, 1}."""
    labels = torch.as_tensor(labels)
    n = labels.numel()
    n1 = labels.sum().clamp(min=1).item()
    n0 = max(n - n1, 1)
    return torch.tensor([n / (2 * n0), n / (2 * n1)], dtype=torch.float)


def distillation_targets(cfg, teacher_logits, labels):
    """Soft targets p_tau (Eq. 11), optionally revised (ours)."""
    labels = labels.float()
    p_tau = torch.sigmoid(teacher_logits / cfg.tau)
    if cfg.kd_revise:
        wrong = (teacher_logits > 0).float() != labels
        revised = cfg.revise_beta * labels + (1 - cfg.revise_beta) * p_tau
        p_tau = torch.where(wrong, revised, p_tau)
    return p_tau


def student_loss(cfg, student_logits, teacher_logits, labels, cw=None):
    """Total student objective. Returns (loss, dict of logged parts)."""
    labels = labels.float()
    l_std = hard_loss(cfg, student_logits, labels, reduction="none")

    if not cfg.use_kd or teacher_logits is None:
        return l_std.mean(), {"l_std": l_std.mean().item()}

    target = distillation_targets(cfg, teacher_logits, labels)
    l_dst = F.binary_cross_entropy_with_logits(student_logits / cfg.tau, target, reduction="none")

    if cfg.kd_cb and cw is not None:
        cw = cw.to(student_logits.device)
        l_dst = l_dst * cw[labels.long()]

    if cfg.kd_gate:
        p = torch.sigmoid(teacher_logits)
        reliability = labels * p + (1 - labels) * (1 - p)
        lam = cfg.lam * reliability.clamp(min=cfg.gate_floor)
    else:
        lam = torch.full_like(l_std, cfg.lam)

    loss = (lam * l_dst + (1 - lam) * l_std).mean()
    return loss, {"l_std": l_std.mean().item(), "l_dst": l_dst.mean().item(),
                  "lam_mean": lam.mean().item()}
