"""Dataset loading.

SE  = SemEval-2019 Task 5 (HatEval), English, sub-task A (hate vs non-hate).
      9000 train / 1000 dev / 2970 test - the official split.
      Source priority: local CSVs in data/hateval/ (original release, raw text)
      -> TweetEval `hate` files (same split; user names anonymised to "@user"),
         read from data/tweeteval_hate/ or downloaded from GitHub.

DV  = Davidson et al. 2017 (24,783 tweets, class 0 = hate, 1 = offensive,
      2 = neither). Following the paper, hate (1,430) vs non-hate (23,353).
      No official split -> stratified 5-fold CV; 10 % of each training fold
      is held out as validation for model selection.
      Source priority: data/davidson/labeled_data.csv, downloaded from the
      authors' GitHub repository if missing.

smoke = small synthetic corpus used to test the code offline.

Every loader returns {"train": {"text": [...], "label": [...]}, "val": ..., "test": ...}
with label 1 = hate.
"""
import os
import random
import urllib.request

import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split


TWEETEVAL_URL = "https://raw.githubusercontent.com/cardiffnlp/tweeteval/main/datasets/hate/{}"
DAVIDSON_URL = ("https://raw.githubusercontent.com/t-davidson/hate-speech-and-offensive-language/"
                "master/data/labeled_data.csv")


def _download(url, path):
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        urllib.request.urlretrieve(url, path)
    return path


def _split(texts, labels):
    return {"text": list(map(str, texts)), "label": [int(y) for y in labels]}


# --------------------------------------------------------------------------
# SemEval-2019 (HatEval)
# --------------------------------------------------------------------------
def load_se(data_dir="data"):
    local = os.path.join(data_dir, "hateval")
    if os.path.exists(os.path.join(local, "train.csv")):
        out = {}
        for split, fname in (("train", "train.csv"), ("val", "dev.csv"), ("test", "test.csv")):
            df = pd.read_csv(os.path.join(local, fname))
            out[split] = _split(df["text"], df["HS"])
        return out
    out = {}
    for split in ("train", "val", "test"):
        files = []
        for kind in ("text", "labels"):
            name = f"{split}_{kind}.txt"
            files.append(_download(TWEETEVAL_URL.format(name), os.path.join(data_dir, "tweeteval_hate", name)))
        with open(files[0], encoding="utf-8") as ft, open(files[1]) as fl:
            texts = [line.rstrip("\n") for line in ft]
            labels = [int(line) for line in fl]
        out[split] = _split(texts, labels)
    return out


# --------------------------------------------------------------------------
# Davidson
# --------------------------------------------------------------------------
def _dv_frame(data_dir="data"):
    df = pd.read_csv(_download(DAVIDSON_URL, os.path.join(data_dir, "davidson", "labeled_data.csv")))
    df = df[["tweet", "class"]].rename(columns={"tweet": "text"})
    df["label"] = (df["class"] == 0).astype(int)
    return df.reset_index(drop=True)


def load_dv(fold=0, n_folds=5, seed=42, data_dir="data"):
    df = _dv_frame(data_dir)
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    train_idx, test_idx = list(skf.split(df["text"], df["label"]))[fold]
    train_df, test_df = df.iloc[train_idx], df.iloc[test_idx]
    train_df, val_df = train_test_split(train_df, test_size=0.1, stratify=train_df["label"],
                                        random_state=seed)
    return {"train": _split(train_df["text"], train_df["label"]),
            "val": _split(val_df["text"], val_df["label"]),
            "test": _split(test_df["text"], test_df["label"])}


def load_dv_full(data_dir="data"):
    df = _dv_frame(data_dir)
    return _split(df["text"], df["label"])


# --------------------------------------------------------------------------
# synthetic smoke-test corpus
# --------------------------------------------------------------------------
_GROUPS = ["immigrants", "refugees", "women", "muslims", "migrants"]
_HATE = ["{g} are vermin and should be deported", "send all {g} back now",
         "{g} are a disease on this country", "i hate {g} so much"]
_NEUTRAL = ["{g} deserve respect and safety", "great talk about {g} rights today",
            "my neighbour helped some {g} today", "reading a report about {g} and jobs"]
_HATE_TAGS = ["#BanIslam", "#SendThemBack", "#StopImmigration", "#buildthewall"]
_OK_TAGS = ["#refugeeswelcome", "#LoveWins", "#Monday", "#news"]
_HATE_EMO = ["\U0001F621", "\U0001F92E", "\U0001F595"]
_OK_EMO = ["\U0001F60A", "❤️", "\U0001F602"]


_AMBIGUOUS = ["the elderly folk will soon be gone", "look at this news about {g}",
              "so many {g} in my town now", "they keep coming every year"]


def _smoke_sample(rng, label_noise=0.08):
    y = int(rng.random() < 0.35)
    g = rng.choice(_GROUPS)
    if rng.random() < 0.4:
        # the SAME neutral sentence is hateful or not depending ONLY on tags/emojis,
        # so a denoiser that deletes them cannot separate the classes
        body = rng.choice(_AMBIGUOUS).format(g=g) + " "
        body += " ".join(rng.sample(_HATE_TAGS if y else _OK_TAGS, 2)) + " "
        body += rng.choice(_HATE_EMO if y else _OK_EMO)
    else:
        body = rng.choice(_HATE if y else _NEUTRAL).format(g=g)
        if rng.random() < 0.5:
            body += " " + rng.choice(_HATE_TAGS if y else _OK_TAGS)
        if rng.random() < 0.5:
            body += " " + rng.choice(_HATE_EMO if y else _OK_EMO)
    if rng.random() < 0.3:
        body = "@user" + str(rng.randint(1, 99)) + " " + body + " https://t.co/xyz" + str(rng.randint(0, 9))
    if rng.random() < 0.2:
        body += " !!!!!"
    if rng.random() < label_noise:          # annotation noise -> the teacher is never perfect
        y = 1 - y
    return body, y


def load_smoke(n=2000, seed=0):
    rng = random.Random(seed)
    rows = [_smoke_sample(rng) for _ in range(n)]
    t, y = zip(*rows)
    a, b = int(0.7 * n), int(0.8 * n)
    return {"train": _split(t[:a], y[:a]), "val": _split(t[a:b], y[a:b]), "test": _split(t[b:], y[b:])}


# --------------------------------------------------------------------------
def load_dataset_splits(cfg):
    if cfg.dataset == "se":
        data = load_se(cfg.data_dir)
    elif cfg.dataset == "dv":
        data = load_dv(cfg.fold, cfg.n_folds, 42, cfg.data_dir)
    elif cfg.dataset == "smoke":
        data = load_smoke()
    else:
        raise ValueError(cfg.dataset)
    if cfg.train_limit:
        rng = random.Random(cfg.seed)
        idx = rng.sample(range(len(data["train"]["text"])), min(cfg.train_limit, len(data["train"]["text"])))
        data["train"] = {k: [v[i] for i in idx] for k, v in data["train"].items()}
    return data


def load_cross_eval_set(cfg):
    """The OTHER dataset, used to measure cross-dataset generalisation."""
    if cfg.dataset == "se":
        return "dv_full", load_dv_full(cfg.data_dir)
    if cfg.dataset == "dv":
        return "se_test", load_se(cfg.data_dir)["test"]
    return "smoke_other", load_smoke(n=200, seed=1)["test"]
