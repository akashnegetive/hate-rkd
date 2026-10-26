# SR-KD: Semantics-preserving, Reliability-aware Knowledge Distillation for Hate Speech Detection

M.Tech thesis project. It extends **Gui, Deng & Zhao, "Improving Hate Speech Detection via Robust
Knowledge Distillation", IEEE TCSS 13(4), 2026** (called *RKD* here). It contains a faithful
re-implementation of RKD and easy-to-medium, parameter-free improvements, each tested on real data:

| | Problem in RKD | Our change | Pilot result (real data) |
|---|---|---|---|
| **N1** | Denoising *deletes* emojis | **Emoji → text**: `😡` → "enraged face" | better teacher on every seed ✅ |
| **N2** | Student trusts the teacher equally on every sample | **Reliability-gated λ** per sample | +2.5 on top of N1 ✅ |
| (tested) | Hashtags deleted | keep `#BanIslam` → "ban islam" (± hashtag dropout) | hurts: SemEval test reverses hashtag meaning ❌ |
| (tested) | Wrong soft labels; imbalance only in hard loss | label revision; class-balanced KD | no gain ❌ (negative ablations) |
| **E** | Only accuracy/F1 reported | + **ECE calibration**, **cross-dataset**, **slices**, **McNemar** | — |

**GPU results with real BERT / DistilBERT ([`docs/GPU_RESULTS.md`](docs/GPU_RESULTS.md)).** The
reliability gate added to the paper's method (`--method rkd_gate`, ladder step X6) improves the
student on every SemEval seed (64.3 ± 1.0 vs. 63.5 ± 0.6 macro-F1; significant on 1 of 3 seeds)
and on Davidson fold 0 (71.2 vs. 70.3; p = 0.0007), at zero inference cost. Emoji → text did
**not** help with pretrained models, although it did in the from-scratch CPU pilot
([`docs/PILOT_RESULTS.md`](docs/PILOT_RESULTS.md)). Keeping hashtags costs −14 macro-F1 on
SemEval, which confirms the HatEval hashtag shift. Our reproduction of RKD is 5–8 points below the
paper's reported SemEval numbers (baselines too); see the gap analysis.

## New contribution: context-privileged distillation (CP-KD)

A reply like "so true, 100% agree" is hateful or harmless depending on the comment it answers.
**CP-KD** trains a BERT teacher that reads *parent + reply* and distils it into a DistilBERT
student that reads **only the reply**. The student also outputs a **"needs context"** score,
learnt for free from how much the teacher's verdict changes when the parent is hidden. It stays
as fast as a reply-only model and flags the replies that need the thread. Datasets: Reddit
replies with parents (Yu et al., NAACL 2022) and Civil Comments in Context (Xenos et al.),
whose human label flips test the flag. Details, novelty check and expected results:
**[`docs/CONTEXT_KD.md`](docs/CONTEXT_KD.md)**. Run: **`notebooks/gpu_context.ipynb`**.

**GPU result: negative ([`docs/CONTEXT_RESULTS.md`](docs/CONTEXT_RESULTS.md)).** BERT gains at most
+0.9 to +1.7 macro-F1 from the parent (Reddit) and nothing on CCC, so the reply-only student does
not improve (+0.1 / +0.2, not significant), and the needs-context head finds human
context-dependent replies worse than the model's own uncertainty (AUROC 0.75 vs. 0.82).
Giving the student the parent makes it worse (−2.5 / −2.9).

With all switches off, the code is exactly the paper's method, so every comparison is
like for like.

Each choice is checked against prior work in
**[`docs/LITERATURE_VALIDATION.md`](docs/LITERATURE_VALIDATION.md)**. Emoji-to-text follows
OffensEval practice (NULI 2019), the gate adapts confidence-conditioned KD (CCKD 2021, Hard-Gate
KD 2022), and the hashtag shift confirms MITRE (SemEval 2019). The contribution is the validated
combination for hate-speech KD, plus the negative results.

**Read [`docs/THESIS_GUIDE.md`](docs/THESIS_GUIDE.md)** for the research gap, equations, a
file-by-file code explanation, the 6-day plan and viva questions.

## Run on GPU: step-by-step ladder (recommended)

Open **[`notebooks/gpu_ladder.ipynb`](notebooks/gpu_ladder.ipynb)** in Colab (T4 GPU) or Kaggle
(GPU + Internet on) and run the cells from the top. It trains real BERT / DistilBERT and adds one
change per step:

| Step | Change | Purpose |
|---|---|---|
| L0 | BERT / DistilBERT, raw text | paper's baseline rows |
| L1 | paper's RKD | **before** (+ reproduction check vs. paper Table II) |
| L2 | + emoji → text | effect of N1 |
| L3 | + reliability gate = SR-KD | effect of N2, **after** |
| X1–X7 | L3 (or L1) + one extra change | hashtags, hashtag dropout, revision, class balance, calibrated gate, gate without emoji, smaller λ |

Results are saved to Google Drive (Colab) and finished steps are skipped on re-run, so a
disconnect costs at most one step. `python scripts/make_ladder.py --runs <OUT>` builds the table:
mean ± std per step, the change versus the previous step, and the per-seed McNemar test.

Command-line equivalent:

```bash
DATASET=se SEEDS="42 1 2" OUT=runs bash scripts/run_ladder.sh     # SemEval, 3 seeds
DATASET=dv FOLDS="0" OUT=runs bash scripts/run_ladder.sh          # Davidson fold 0
python scripts/make_ladder.py --runs runs                         # runs/ladder.md
```

## Quick start (other grids)

```bash
git clone https://github.com/akashnegetive/hate-rkd.git && cd hate-rkd
pip install -r requirements.txt
python -m pytest -q tests                 # 13 unit tests
bash scripts/run_all.sh smoke             # whole pipeline on synthetic data, ~2 min
bash scripts/run_all.sh se_main           # SemEval: baseline / RKD / SR-KD × 3 seeds
bash scripts/run_all.sh se_ablation
FOLDS="0 1 2 3 4" bash scripts/run_all.sh dv_main   # Davidson, 5-fold CV
bash scripts/run_all.sh tables            # runs/tables.md (+ McNemar tests)
python scripts/make_figures.py --runs runs --dataset se
```

Trained teachers are cached and reused (`--teacher_dir`), and `--skip_done` skips finished runs.

### Single experiments

```bash
python scripts/run_experiment.py --dataset se --method rkd            # paper (RKD-T + RKD-S)
python scripts/run_experiment.py --dataset se --method srkd           # SR-KD (final)
python scripts/run_experiment.py --dataset se --method srkd --kd_gate false --run_name se_no_gate
python scripts/run_experiment.py --dataset se --method srkd --pilot true   # CPU, no pretrained weights
python scripts/run_experiment.py --dataset dv --method srkd --fold 3
```

Any field of `src/config.py:Config` can be set from the command line (`--epochs 3 --lam 0.5 ...`).

### Data
Downloaded automatically from the authors' GitHub repositories (TweetEval `hate` split for
SemEval-2019, `t-davidson/hate-speech-and-offensive-language` for Davidson). To use the original raw HatEval files instead,
put `train.csv`, `dev.csv` and `test.csv` (columns `text`, `HS`) in `data/hateval/`.

## Output
`runs/results/<run>__{teacher,student}.json` holds the config, training history and every
metric (clean, perturbed, slices, cross-dataset, efficiency), together with the test
logits used for significance tests and reliability diagrams.
