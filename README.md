# SR-KD: Semantics-preserving, Reliability-aware Knowledge Distillation for Hate Speech Detection

M.Tech thesis project. It extends **Gui, Deng & Zhao, "Improving Hate Speech Detection via Robust
Knowledge Distillation", IEEE TCSS 13(4), 2026** (called *RKD* here). It contains a faithful
re-implementation of RKD and three easy-to-medium, parameter-free improvements:

| | Problem in RKD | Our fix | Where |
|---|---|---|---|
| **N1** | Denoising *deletes* emojis and hashtags, which removes hate cues (`#BanIslam`) | **Semantic-preserving normalisation**: `😡` → "enraged face", `#BanIslam` → "ban islam" | `src/preprocess.py` |
| **N2** | The student trusts the teacher equally on every sample, even wrong ones | **Reliability-gated λ** per sample + **revision** of wrong soft labels | `src/losses.py` |
| **N3** | Class imbalance handled only in the hard-label loss | **Class-balanced distillation loss** | `src/losses.py` |
| **E** | Only accuracy and F1 reported | + **ECE / Brier calibration**, **cross-dataset**, **emoji/hashtag slice**, **McNemar** | `src/evaluate.py` |

With all three switches off, the code is exactly the paper's method, so every comparison is
like for like.

**Read [`docs/THESIS_GUIDE.md`](docs/THESIS_GUIDE.md)** for the research gap, equations, a
file-by-file code explanation, the 6-day plan and viva questions.

## Quick start (Colab / Kaggle, T4 GPU)

```bash
git clone https://github.com/akashnegetive/hate-rkd.git && cd hate-rkd
pip install -r requirements.txt
python -m pytest -q tests                 # 12 unit tests
bash scripts/run_all.sh smoke             # whole pipeline on synthetic data, ~2 min
bash scripts/run_all.sh se_main           # SemEval: baseline / RKD / ours × 3 seeds
bash scripts/run_all.sh se_ablation
FOLDS="0 1 2 3 4" bash scripts/run_all.sh dv_main   # Davidson, 5-fold CV
bash scripts/run_all.sh tables            # runs/tables.md (+ McNemar tests)
python scripts/make_figures.py --runs runs --dataset se
```

Or open `notebooks/colab_run.ipynb`. Set `OUT=/content/drive/MyDrive/hate-rkd-runs` to keep
results on Google Drive between sessions. Trained teachers are cached and reused.

### Single experiments

```bash
python scripts/run_experiment.py --dataset se --method rkd            # paper (RKD-T + RKD-S)
python scripts/run_experiment.py --dataset se --method ours           # SR-KD
python scripts/run_experiment.py --dataset se --method ours --kd_gate false --run_name se_no_gate
python scripts/run_experiment.py --dataset dv --method ours --fold 3
```

Any field of `src/config.py:Config` can be set from the command line (`--epochs 3 --lam 0.5 ...`).

### Data
Downloaded automatically from HuggingFace (`cardiffnlp/tweet_eval` config `hate` for SemEval-2019,
`tdavidson/hate_speech_offensive` for Davidson). To use the original raw HatEval files instead,
put `train.csv`, `dev.csv` and `test.csv` (columns `text`, `HS`) in `data/hateval/`.

## Output
`runs/results/<run>__{teacher,student}.json` holds the config, training history and every
metric (clean, perturbed, slices, cross-dataset, efficiency), together with the test
logits used for significance tests and reliability diagrams.
