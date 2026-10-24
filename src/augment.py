"""Dual-round data augmentation (paper Sec. III-C).

Round 1 (Algorithm 1, offline, text level)
    Pick K = rho * |S| speeches, apply a character-level "attacker" that
    randomly swaps / deletes / inserts / substitutes characters inside words
    (simulates typos). Perturbed copies are ADDED to the training set.
    The paper uses TextAttack's character transformations; we implement the
    same four operations directly so no victim model or extra library is
    needed and it runs in milliseconds.

Round 2 (Algorithm 2, online, token level)
    For every batch, each token is replaced by [UNK] with probability pi
    (independent Bernoulli trials) - simulates missing words. Special tokens
    ([CLS], [SEP], [PAD]) are never touched.
"""
import random
import string

import torch

_LETTERS = string.ascii_lowercase


def _perturb_word(word: str, rng: random.Random) -> str:
    if len(word) < 3 or not word.isalpha():
        return word
    op = rng.choice(("swap", "delete", "insert", "substitute"))
    i = rng.randrange(1, len(word) - 1)         # keep first letter (as TextAttack does)
    if op == "swap":
        return word[:i] + word[i + 1] + word[i] + word[i + 2:]
    if op == "delete":
        return word[:i] + word[i + 1:]
    if op == "insert":
        return word[:i] + rng.choice(_LETTERS) + word[i:]
    return word[:i] + rng.choice(_LETTERS) + word[i + 1:]


def char_perturb(text: str, pi: float, rng: random.Random) -> str:
    """Perturb every word independently with probability pi (the attacker A)."""
    out = [_perturb_word(w, rng) if rng.random() < pi else w for w in text.split()]
    # guarantee at least one change so the copy is actually new
    if out == text.split():
        idx = [k for k, w in enumerate(out) if len(w) >= 3 and w.isalpha()]
        if idx:
            k = rng.choice(idx)
            out[k] = _perturb_word(out[k], rng)
    return " ".join(out)


def round_one(texts, labels, rho: float, pi: float, seed: int, minority_only: bool = True):
    """Algorithm 1. Returns the original data + K perturbed copies."""
    rng = random.Random(seed)
    pool = [i for i, y in enumerate(labels) if y == 1] if minority_only else list(range(len(texts)))
    k = int(len(texts) * rho)
    new_t, new_y = [], []
    for _ in range(k):
        i = rng.choice(pool)                    # uniform sampling with replacement
        new_t.append(char_perturb(texts[i], pi, rng))
        new_y.append(labels[i])
    return list(texts) + new_t, list(labels) + new_y


def round_two(input_ids: torch.Tensor, attention_mask: torch.Tensor, unk_id: int,
              special_ids, pi: float, generator: torch.Generator = None) -> torch.Tensor:
    """Algorithm 2 on a whole batch at once (vectorised Bernoulli trials)."""
    drop = torch.rand(input_ids.shape, generator=generator, device="cpu").to(input_ids.device) < pi
    protect = attention_mask == 0
    for sid in special_ids:
        protect |= input_ids == sid
    drop &= ~protect
    return input_ids.masked_fill(drop, unk_id)
