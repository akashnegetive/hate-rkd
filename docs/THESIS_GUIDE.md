# SR-KD: Semantics-preserving, Reliability-aware Knowledge Distillation for Hate Speech Detection

**Thesis guide: research gap, method, code explanation, 6-day plan, viva preparation**

Base paper: Y. Gui, H. Deng, S. Zhao, *"Improving Hate Speech Detection via Robust Knowledge
Distillation"*, IEEE Trans. Computational Social Systems, vol. 13, no. 4, pp. 5207–5220, Aug. 2026.
In this guide the paper's method is called **RKD** (RKD-T = teacher, RKD-S = student).

---

## 1. One-paragraph summary (say this first in the viva)

RKD compresses a BERT hate-speech detector into DistilBERT using knowledge distillation, with
three supports: text denoising, two rounds of data augmentation, and focal loss. We found three
weaknesses. (1) The denoiser **deletes emojis and hashtags**, yet the paper's own error analysis
shows hateful intent hidden in hashtags such as `#BanIslam`. (2) The student **trusts the
teacher equally on every sample**, including samples the teacher gets wrong. (3) Class imbalance
is handled **only in the hard-label loss**, not in the distillation loss. We propose **SR-KD**,
which (a) turns emojis and hashtags into words instead of deleting them, (b) sets the
distillation weight per sample from how reliable the teacher is on that sample, and corrects
wrong soft labels, and (c) makes the distillation loss class-balanced. We also add two
evaluations the paper lacks: **calibration (ECE)** and **cross-dataset generalisation**. Each
change is a few lines of code, adds no parameters, and leaves inference cost unchanged, so the
student stays exactly as small and fast as RKD-S.

---

## 2. The base paper in one page

Pipeline (paper Fig. 2):

```
Stage 1  raw tweets ──6-step denoising──► denoised ──Round-1 (char typos, offline)──► augmented
                                                       │ tokenise
                                                       ▼
                                          token ids ──Round-2 ([UNK] masking, online, per batch)
Stage 2  BERT teacher  trained with focal loss on hard labels y
Stage 3  teacher frozen → soft labels p_τ ; DistilBERT student trained with
         L_ov = λ·L_dst + (1-λ)·L_std
```

Key equations (one logit per tweet, sigmoid gives P(hate)):

| Eq. | Meaning |
|---|---|
| (2) `p = σ(v)` | teacher probability of hate |
| (3) `p̄ = p if y=1 else 1−p` | probability given to the true class |
| (4)/(9) `L = −α_i (1−p̄)^γ log p̄` | focal loss: α balances classes, γ focuses on hard samples |
| (11)/(12) `p_τ = σ(v/τ)`, `q_τ = σ(z/τ)` | temperature-scaled teacher and student probabilities |
| (13) `L_dst = CE(p_τ, q_τ)` | distillation loss, **no class weighting** |
| (14) `L_ov = λ L_dst + (1−λ) L_std` | **one λ for all samples** |

Reported hyper-parameters: ρ = π = 0.1; (α, γ, τ, λ) = (0.2, 2, 0.3, 0.8) for SE and
(0.6, 2, 0.3, 0.8) for DV; Adam with lr 1e-5; batch 32 (teacher) and 64 (student).
Results (Table II): RKD-S reaches 71.8 Acc / 71.5 M-F1 on SE and 96.2 Acc / 95.5 W-F1 on DV,
with about 40% lower latency than RKD-T.

---

## 3. Research gaps (with evidence)

| # | Gap | Evidence in the paper | Status in the literature (Oct 2026 search) |
|---|---|---|---|
| G1 | **Denoising deletes meaning.** `sps-rem` removes emojis and `htg-rem` removes hashtags. | Table I deletes `#december8th` and the emojis. Table IX, Exp. 4 ("…signal the end. #StopImmigration #BanIslam #GetBritainBack") is hate that **both** RKD models miss; after denoising the hateful hashtags are gone. The paper's own future work asks for "non-textual signals". | Emoji-to-text helps hate detection in non-KD BiLSTM studies, and hashtag segmentation is a common but un-ablated preprocessing step. **No work studies semantic-preserving normalisation inside a hate-speech KD pipeline.** |
| G2 | **The teacher is trusted blindly.** λ is a constant, so wrong or uncertain soft labels get 80% of the loss weight. | Sec. III-E admits ambiguous samples give near-0.5 teacher confidences that "hinder the student". Table IX Exp. 1: RKD-S copies an ambiguous decision. | Confidence-aware KD (CA-MKD, multi-teacher, vision) and label revision (Lan et al. 2024, vision) exist. In hate speech, BAT-KD adapts the *temperature* by bias type and DDML uses multiple students. A GitHub HASOC project explicitly lists "confidence-weighted distillation" as **future work**. **Per-sample reliability gating plus soft-label revision is not applied to hate-speech KD.** |
| G3 | **Imbalance ignored in L_dst.** α appears only in L_std, Eq. 9. Eq. 13 is plain CE. | DV is 5.8% hate. With λ = 0.8, 80% of the gradient comes from the unweighted term. | Balanced KD (Zhang et al. 2021) does this for long-tailed **vision**. No text or hate-speech KD work found. |
| G4 | **No reliability or generalisation evaluation.** Only accuracy and F1 on in-domain test sets. | Table II/VIII. | Calibration of distilled students has been studied for emotion tasks (EMNLP 2022), not for hate-speech KD. Cross-dataset evaluation of hate-speech KD students was not found. |

**Why this is a good M.Tech scope:** every gap is fixed by changing the *data* or the *loss*.
There are no new networks, no extra parameters and no change to inference cost, so it is
**easy to medium** to implement. Results are measurable on the same two public datasets, and each
fix can be switched on or off separately for a clean ablation.

---

## 4. Proposed method: SR-KD

### N1. Semantic-Preserving Normalisation (SPN), `src/preprocess.py`

Paper steps 1, 3, 4 and 5 are kept (mojibake repair, URL removal, repeated-character
elimination, username removal). Two steps are replaced:

| Paper step | Our step | Example |
|---|---|---|
| sps-rem: delete emojis | **emo-txt**: emoji → English CLDR name, repeated emojis collapsed to one | `😡` → `enraged face`; `😂😂😂` → `face with tears of joy` |
| htg-rem: delete hashtag | **htg-seg**: hashtag → words (camel-case split, else `wordsegment` unigram LM) | `#BanIslam` → `ban islam`; `#buildthewall` → `build the wall` |

Why it works: BERT's WordPiece tokenizer splits an emoji or a long hashtag into meaningless
sub-word pieces (the paper's Fig. 1 problem). Converting them to *ordinary English words*
removes that noise **without** deleting the meaning. The words "enraged", "ban" and "islam"
already have good pre-trained embeddings.

### N2. Reliability-aware distillation, `src/losses.py`

For training sample i, with teacher logit v_i and label y_i:

```
reliability   r_i = σ(v_i)       if y_i = 1
                    1 − σ(v_i)   if y_i = 0          (teacher's probability of the TRUE class)

gate          λ_i = λ · max(r_i, floor)              (floor = 0 by default)

revision      t_i = β·y_i + (1−β)·p_τ,i   if the teacher is wrong (sign of v_i ≠ y_i)
              t_i = p_τ,i                 otherwise          (β = 0.6)
```

* **Gate.** When the teacher is confident *and* right (r ≈ 1), the student listens to it with
  λ_i ≈ λ, as in the paper. When the teacher is wrong or unsure (r small), λ_i shrinks and the
  sample is handed back to the focal hard-label loss.
* **Revision.** Even for a wrong teacher, the soft label still holds useful "dark knowledge"
  (how close to the boundary the sample is). We keep 40% of it but move the label to the
  correct side. β ≥ 0.5 guarantees this: for y = 1, t = 0.6 + 0.4·p ≥ 0.6.

### N3. Class-balanced distillation, `src/losses.py`

```
w_c = N / (2·N_c)                  inverse class frequency, from the training set
L_dst,i ← w_{y_i} · L_dst,i
```

On DV the hate weight is about 8.7 and the non-hate weight about 0.53, so the minority class is
no longer drowned out in the 80% of the loss that comes from distillation.

### Final objective

```
L_i = λ_i · w_{y_i} · BCE(q_τ,i , t_i)  +  (1 − λ_i) · Focal(z_i , y_i)
```

With `kd_gate = kd_revise = kd_cb = False` this reduces **exactly** to the paper's Eq. 14. That
is why the ablation is clean: the same code runs both methods.

### E. New evaluation protocol, `src/evaluate.py` and `src/metrics.py`

* **ECE (Expected Calibration Error)**: are 80%-confident predictions right 80% of the time?
  This matters for moderation, where a threshold decides what a human reviews.
* **Brier score**: mean squared error of P(hate).
* **Cross-dataset**: SE-trained model on all of DV, and DV-trained model on the SE test set.
* **Slice analysis**: test tweets with and without emojis/hashtags. This is direct evidence for N1.
* **Robustness**: whole test set with Round-1 typos and Round-2 masking (paper Sec. V-C).
* **McNemar test**: tells you whether "ours vs. RKD" is statistically significant.

---

## 5. Code walkthrough

```
hate-rkd/
├── src/
│   ├── config.py      all hyper-parameters (Config), dataset presets, method presets
│   ├── data.py        load SE / DV / synthetic smoke data → {"train","val","test"}
│   ├── preprocess.py  paper denoiser (denoise_remove) and ours (denoise_spn)
│   ├── augment.py     Round-1 char perturbation, Round-2 [UNK] masking
│   ├── losses.py      focal loss, paper KD loss, our gate / revise / class-balance
│   ├── models.py      build BERT / DistilBERT with a 1-logit head (or tiny smoke models)
│   ├── train.py       one training loop for teacher and student, best-epoch selection
│   ├── metrics.py     Acc, macro/weighted F1, hate P/R/F1, ECE, Brier
│   └── evaluate.py    clean / slices / perturbed / cross-dataset / efficiency
├── scripts/
│   ├── run_experiment.py  one experiment: Stage 1 → Stage 2 (teacher) → Stage 3 (student)
│   ├── run_all.sh         the full experiment grid, in resumable sections
│   ├── make_tables.py     results JSON → Markdown/CSV tables + McNemar test
│   └── make_figures.py    reliability diagram, setting comparison, ablation chart
├── tests/test_core.py     12 unit tests (paper Table I examples, loss properties…)
└── notebooks/colab_run.ipynb  click-through notebook for Colab/Kaggle
```

### 5.1 `config.py`
One dataclass, `Config`, holds every knob. `PRESETS` copies the paper's per-dataset values
(α = 0.2 for SE, 0.6 for DV). `METHODS` names three switch combinations:

* `baseline`: raw text, no augmentation, CE loss, no KD. These are the paper's "BERT" and
  "DistilBERT" rows.
* `rkd`: paper denoising, R1 + R2, focal loss, plain KD. This is the faithful re-implementation.
* `ours`: SPN, R1 + R2, focal loss, KD with gate + revise + class-balance.

Every field can be overridden from the command line (`--kd_gate false`, `--lam 1.0`, …).
That is how the ablations are run.

### 5.2 `data.py`
* `load_se`: HatEval 2019 official split. It uses local raw CSVs if present, otherwise
  HuggingFace `cardiffnlp/tweet_eval` (config `hate`), which is the same split.
* `load_dv`: Davidson, where class 0 = hate is label 1 and everything else is label 0. It uses
  stratified 5-fold CV, as in the paper, and holds out 10% of each training fold as validation.
* `load_cross_eval_set`: returns the *other* dataset for cross-dataset testing.
* `load_smoke`: a synthetic corpus in which 40% of tweets share one neutral sentence and
  differ only in their hashtags and emojis, with 8% of labels flipped. It exists to test the
  code offline, and it shows the mechanism of N1 in a controlled setting.

### 5.3 `preprocess.py`
Small single-purpose functions (`repair_mojibake`, `remove_urls`, `emoji_to_text`,
`segment_hashtag`, …) composed into two pipelines. `denoise_remove` reproduces the paper's
Table I outputs exactly; `tests/test_core.py::test_paper_denoiser_matches_table1` checks this.
`has_emoji_or_hashtag` defines the evaluation slice.

### 5.4 `augment.py`
* `char_perturb`: for each word, with probability π, apply swap / delete / insert / substitute
  of one inner character. These are the four TextAttack operations the paper describes. They are
  implemented directly, so augmentation takes milliseconds instead of the paper's 218–3132 ms per
  sample (Table IV).
* `round_one` (Algorithm 1): add K = ρ·|S| perturbed copies, sampled from the hate class.
* `round_two` (Algorithm 2): vectorised Bernoulli masking of a whole batch to `[UNK]`. It never
  touches `[CLS]`, `[SEP]` or padding.

### 5.5 `losses.py`
* `focal_loss`: Eq. 4/9 written with `logsigmoid` for numerical stability.
* `distillation_targets`: Eq. 11 plus our optional revision.
* `student_loss`: Eq. 14 plus our optional gate and class weights. It returns the loss and the
  logged parts (`l_std`, `l_dst`, `lam_mean`).

### 5.6 `train.py`
`train_model` is used for both networks:
1. Shuffle indices, tokenise the batch, apply Round-2 masking (if enabled).
2. Forward pass with mixed precision on GPU, then compute the loss (teacher: focal; student:
   `student_loss` with that batch's pre-computed teacher logits).
3. AdamW with a linear warm-up/decay schedule and gradient clipping.
4. After every epoch, compute validation macro-F1 and keep the best weights.

Teacher logits are computed **once** on clean (un-masked) inputs (`run_experiment.py`).
The paper also freezes the teacher and does not apply Algorithm 2 to its input, so this is
exact and saves one teacher forward pass per student step.

### 5.7 `run_experiment.py`: the three stages
1. **Stage 1**: load the data, preprocess with the chosen denoiser, run Round-1 augmentation.
2. **Stage 2**: train the teacher, or **reuse** a cached one. The cache key covers dataset,
   fold, preprocessing, augmentation, loss, α and seed, so all KD-loss ablations share one
   teacher. Then evaluate and save `runs/results/<run>__teacher.json`.
3. **Stage 3**: compute teacher soft labels, train the student, evaluate, and save
   `<run>__student.json`.

### 5.8 `evaluate.py`, `make_tables.py`, `make_figures.py`
`evaluate_all` produces every number in one JSON. Test logits and labels are stored, so the
tables, McNemar test and reliability diagrams can be redrawn later without retraining.

---

## 6. Experiments to run (GPU: Colab T4 or Kaggle T4/P100)

| Section | Command | What it gives the thesis | Approx. T4 time* |
|---|---|---|---|
| sanity | `bash scripts/run_all.sh smoke` | proves the code runs | 2 min |
| SE main | `bash scripts/run_all.sh se_main` | Table: baseline vs RKD vs SR-KD, 3 seeds, mean ± std | 2–3 h |
| SE ablation | `bash scripts/run_all.sh se_ablation` | Table: contribution of N1, N2 (gate, revise), N3 | 1 h |
| DV main | `FOLDS="0 1 2 3 4" bash scripts/run_all.sh dv_main` | Table: DV results, 5-fold CV | 5–6 h (≈1.2 h with `FOLDS="0"`) |
| tables | `bash scripts/run_all.sh tables` | all tables + McNemar p-values | seconds |
| figures | `python scripts/make_figures.py --runs runs --dataset se` | 3 thesis figures | seconds |

\*These are estimates (5 epochs, max_len 96, fp16), not measurements. Check the time of the first
epoch in the log and adjust. If you are short on time, use `--epochs 3` and/or `FOLDS="0"`.

Thesis tables this produces:
1. Main comparison (Acc, M-F1, W-F1, hate-F1) on SE and DV, as in paper Table II.
2. Ablation, as in paper Tables V–VII, plus our component rows.
3. Robustness on the perturbed test set (paper Table VIII).
4. **New:** calibration (ECE, Brier) with a reliability diagram.
5. **New:** cross-dataset generalisation.
6. **New:** emoji/hashtag slice versus plain slice.
7. Efficiency (params, size, latency). The student is identical for RKD and SR-KD, so this table
   shows SR-KD keeps RKD's ~40% speed-up.

**What to expect, as hypotheses to test, not promised results:** the largest gain from N1 should
appear on the emoji/hashtag slice. N2 and N3 should help most on DV (heavy imbalance) and in hate
recall/F1, and should improve ECE. If a component does not help, report that honestly. A
negative ablation result, explained, is still a valid thesis finding.

---

## 7. Six-day plan

| Day | Work | Output |
|---|---|---|
| 1 | Read this guide and the code; run `pytest` and `run_all.sh smoke` on Colab; check that the SE/DV loaders download; run one SE `rkd` experiment and compare it to paper Table II | reproduction number for RKD |
| 2 | `se_main` (seeds 42, 1, 2); save `runs/` to Google Drive | SE main table |
| 3 | `se_ablation`; start `dv_main` with `FOLDS="0"` | ablation table, first DV numbers |
| 4 | Remaining DV folds (or keep fold 0 only if GPU time is short); `dv_ablation` | DV table |
| 5 | `tables`, `make_figures.py`; pick 5 qualitative examples where SR-KD fixes RKD (search the test logits); write the results chapter | figures, case-study table |
| 6 | Write method, discussion and limitations chapters; prepare slides and viva answers (Sec. 9) | thesis draft |

---

## 8. Deviations from the paper (state these honestly in the thesis)

1. **Round-1 attacker.** The paper uses TextAttack; we implement the same four character
   operations directly. This is functionally equivalent and much faster.
2. **SE source.** If the original raw HatEval CSVs are not available, `tweet_eval/hate` is used.
   It has the same split, but user names are already replaced by `@user`, which our denoisers
   remove anyway.
3. **Distillation CE.** The paper's Eq. 13 writes only the true-class term `p̄ log q̄`. We use the
   full Bernoulli cross-entropy `−[p_τ log q_τ + (1−p_τ) log(1−q_τ)]`, which is the standard
   KD loss for a sigmoid output. It is the same for both RKD and SR-KD, so the comparison is fair.
4. **Epochs and model selection.** The paper does not report epochs. We use 5 epochs and keep
   the best validation macro-F1 epoch.
5. **Robustness protocol.** The paper perturbs only samples that all models classify correctly.
   We perturb the whole test set, which is simpler and the same for every model.
6. **Algorithm 2 wording.** As written, the paper keeps a token when `r > 0`, which with
   probability π would *keep* only 10% of tokens. We follow the clearly intended meaning:
   *mask* with probability π.

Because the paper's code is not public, your RKD reproduction may differ by 1–3 points from
Table II. Always compare SR-KD against **your own** RKD run, with the same code, seeds and
hardware. This is the fair comparison, and examiners accept it.

---

## 9. Likely viva questions

**Q: What is novel if gating and label revision exist in vision KD?**
A: The contribution is (i) finding *why* the hate-speech KD pipeline fails (deleted hashtag and
emoji semantics, blind trust in the teacher, imbalance only in L_std), with evidence from the
base paper's own error cases; (ii) a combined, parameter-free fix for binary sigmoid KD in this
domain; and (iii) the first calibration and cross-dataset evaluation of hate-speech KD students.
Combining existing tools for a new, well-motivated problem is a recognised contribution, and
the ablations show what each part adds.

**Q: Why not simply lower λ?**
A: A lower λ reduces the teacher's influence on *all* samples, including the many where it is
right. The gate lowers it only where the teacher is unreliable. The `se_abl_no_gate` row versus
a tuned λ shows the difference.

**Q: Does the gate use labels at test time?**
A: No. Reliability uses training labels only. At inference the student is plain DistilBERT with
zero extra cost.

**Q: Could emoji names introduce bias (e.g., the 🍑 emoji)?**
A: Yes. CLDR names are literal, and slang meanings are lost. Future work: learned emoji
embeddings (emoji2vec) or slang lexicons.

**Q: Why ECE?**
A: Moderation systems use thresholds and route uncertain posts to humans. A model that is 95%
confident but right only 70% of the time sends the wrong posts to review.

**Q: Why is cross-dataset performance low?**
A: The datasets define hate differently. SE targets immigrants and women, while DV separates
"hate" from "offensive", so DV's offensive tweets are labelled non-hate. A drop is expected. We
compare *relative* drops between methods.

**Q: Statistical significance?**
A: Report mean ± std over 3 seeds or 5 folds, plus McNemar's test on the paired test predictions
(`make_tables.py --compare`).

---

## 10. Limitations and future work
* English only, and only two datasets.
* CLDR emoji names are literal. Hashtag segmentation can fail on rare slang (`#libtards`).
* The reliability gate depends on the teacher's own calibration. Temperature-calibrating the
  teacher first is a natural extension.
* Implicit hate (sarcasm, coded language) remains hard; see the paper's Exp. 3–5.

---

## 11. References used for the gap analysis
* Gui, Deng, Zhao, IEEE TCSS 13(4), 2026 (base paper).
* Zhang et al., *Balanced Knowledge Distillation for Long-tailed Learning*, arXiv:2104.10510.
* Zhang et al., *Confidence-Aware Multi-Teacher Knowledge Distillation*, arXiv:2201.00007.
* Lan et al., *Improve Knowledge Distillation via Label Revision and Data Selection*, arXiv:2404.03693.
* BAT-KD (bias-aware adaptive temperature KD for hate speech), in arXiv:2603.18786 (proceedings volume).
* DDML: Multi-Student Knowledge Distillation for Hate Speech, Entropy 27(4), 2025 (doi:10.3390/e27040417).
* Calibrating Student Models for Emotion-related Tasks, EMNLP 2022.
* Basile et al., SemEval-2019 Task 5; Davidson et al., ICWSM 2017; Lin et al., Focal Loss, ICCV 2017;
  Hinton et al., Distilling the Knowledge in a Neural Network, 2015.
