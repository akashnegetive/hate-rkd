"""Model + tokenizer construction.

Real runs  : HuggingFace checkpoints with a 1-unit classification head
             (num_labels=1 -> one logit, sigmoid in the loss), as in the paper.
Pilot runs : `scratch-teacher` (BERT, 4 layers, 256 dim) and `scratch-student`
             (DistilBERT, 2 layers, 128 dim) trained FROM SCRATCH on the real
             dataset with a WordPiece vocabulary learnt from its training split.
             Used when no GPU / no pretrained weights are available: same
             pipeline and losses, lower absolute scores than pretrained BERT.
Smoke runs : tiny models + synthetic corpus, to test the code offline.
"""
import os

from transformers import (AutoModelForSequenceClassification, AutoTokenizer, BertConfig,
                          BertForSequenceClassification, BertTokenizerFast, DistilBertConfig,
                          DistilBertForSequenceClassification)

CACHE_DIR = os.path.join(os.path.dirname(__file__), "..", ".smoke_cache")

ARCH = {
    "smoke-teacher": ("bert", dict(hidden_size=64, num_hidden_layers=2, num_attention_heads=2,
                                   intermediate_size=128)),
    "smoke-student": ("distil", dict(dim=32, n_layers=1, n_heads=2, hidden_dim=64)),
    "scratch-teacher": ("bert", dict(hidden_size=256, num_hidden_layers=4, num_attention_heads=4,
                                     intermediate_size=1024)),
    "scratch-student": ("distil", dict(dim=128, n_layers=2, n_heads=2, hidden_dim=512)),
}


def _train_wordpiece(corpus, path, vocab_size):
    if not os.path.exists(os.path.join(path, "vocab.txt")):
        from tokenizers import BertWordPieceTokenizer
        tok = BertWordPieceTokenizer(lowercase=True)
        tok.train_from_iterator(corpus, vocab_size=vocab_size, min_frequency=2)
        os.makedirs(path, exist_ok=True)
        tok.save_model(path)
    return BertTokenizerFast.from_pretrained(path, do_lower_case=True)


def build_tokenizer(name: str, corpus=None, key: str = ""):
    """`corpus`/`key` are only used by the from-scratch pilot models."""
    from .preprocess import denoise_remove, denoise_spn
    if name.startswith("smoke"):
        from .data import load_smoke
        d = load_smoke(n=2000, seed=123)["train"]["text"]
        corpus = d + [denoise_spn(t) for t in d] + [denoise_remove(t) for t in d]
        return _train_wordpiece(corpus, os.path.join(CACHE_DIR, "tokenizer"), 600)
    if name.startswith("scratch"):
        # one shared vocabulary per training split, learnt from all three text views
        corpus = list(corpus) + [denoise_spn(t) for t in corpus] + [denoise_remove(t) for t in corpus]
        return _train_wordpiece(corpus, os.path.join(CACHE_DIR, f"tokenizer_{key}"), 8000)
    return AutoTokenizer.from_pretrained(name)


def build_model(name: str, tokenizer=None):
    if name in ARCH:
        kind, kw = ARCH[name]
        if kind == "bert":
            return BertForSequenceClassification(BertConfig(vocab_size=tokenizer.vocab_size, num_labels=1,
                                                            max_position_embeddings=128, **kw))
        return DistilBertForSequenceClassification(DistilBertConfig(vocab_size=tokenizer.vocab_size,
                                                                    num_labels=1,
                                                                    max_position_embeddings=128, **kw))
    return AutoModelForSequenceClassification.from_pretrained(name, num_labels=1)


def load_finetuned(path: str):
    """Reload a model + tokenizer saved by `save_pretrained`."""
    return (AutoModelForSequenceClassification.from_pretrained(path),
            AutoTokenizer.from_pretrained(path))
