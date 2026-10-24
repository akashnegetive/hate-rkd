"""Model + tokenizer construction.

Real runs : HuggingFace checkpoints with a 1-unit classification head
            (num_labels=1 -> one logit, sigmoid in the loss), as in the paper.
Smoke runs: tiny randomly initialised BERT (teacher) / DistilBERT (student)
            with a WordPiece vocabulary trained on the synthetic corpus, so the
            whole pipeline can be tested without internet or GPU.
"""
import os

from transformers import (AutoModelForSequenceClassification, AutoTokenizer, BertConfig,
                          BertForSequenceClassification, BertTokenizerFast, DistilBertConfig,
                          DistilBertForSequenceClassification)

SMOKE_DIR = os.path.join(os.path.dirname(__file__), "..", ".smoke_cache")


def _smoke_tokenizer():
    path = os.path.join(SMOKE_DIR, "tokenizer")
    if not os.path.exists(os.path.join(path, "vocab.txt")):
        from tokenizers import BertWordPieceTokenizer
        from .data import load_smoke
        from .preprocess import denoise_remove, denoise_spn
        d = load_smoke(n=2000, seed=123)["train"]["text"]
        corpus = d + [denoise_spn(t) for t in d] + [denoise_remove(t) for t in d]
        tok = BertWordPieceTokenizer(lowercase=True)
        tok.train_from_iterator(corpus, vocab_size=600, min_frequency=1)
        os.makedirs(path, exist_ok=True)
        tok.save_model(path)
    return BertTokenizerFast.from_pretrained(path, do_lower_case=True)


def build_tokenizer(name: str):
    if name.startswith("smoke"):
        return _smoke_tokenizer()
    return AutoTokenizer.from_pretrained(name)


def build_model(name: str, tokenizer=None):
    if name == "smoke-teacher":
        cfg = BertConfig(vocab_size=tokenizer.vocab_size, hidden_size=64, num_hidden_layers=2,
                         num_attention_heads=2, intermediate_size=128, num_labels=1,
                         max_position_embeddings=64)
        return BertForSequenceClassification(cfg)
    if name == "smoke-student":
        cfg = DistilBertConfig(vocab_size=tokenizer.vocab_size, dim=32, n_layers=1, n_heads=2,
                               hidden_dim=64, num_labels=1, max_position_embeddings=64)
        return DistilBertForSequenceClassification(cfg)
    if os.path.isdir(name) or not name.startswith("smoke"):
        return AutoModelForSequenceClassification.from_pretrained(name, num_labels=1)
    raise ValueError(name)


def load_finetuned(path: str):
    """Reload a model + tokenizer saved by `save_pretrained`."""
    return (AutoModelForSequenceClassification.from_pretrained(path),
            AutoTokenizer.from_pretrained(path))
