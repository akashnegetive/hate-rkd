# Context-privileged knowledge distillation (CP-KD)

**New thesis contribution.** It follows the supervisor's request after the GPU ladder showed
only small gains (+0.8 macro-F1) for the loss-level changes.

## 1. Limitation addressed

A reply's meaning often depends on the comment it answers:

| Parent comment | Reply | Hateful? |
|---|---|---|
| "refugees are animals" | "so true, 100% agree" | **yes** (endorses hate) |
| "refugees opened a lovely bakery" | "so true, 100% agree" | no |

* The base paper (Gui et al., IEEE TCSS 2026) classifies every post in isolation. Its own
  discussion lists "content lacking sufficient contextual information" as unsolved.
* Prior work shows context changes human labels:
  * Pavlopoulos et al. (ACL 2020): about 5 % of toxicity labels flip.
  * Yu, Blanco & Hong (NAACL 2022): hate vs. counter-speech judgements change with the
    parent, and models improve with context.
  * Bourgeade et al. (LREC-COLING 2024): context is "necessary but difficult" for models.
* Context-aware models need the whole thread at inference. That doubles the input length, and
  the thread is often unavailable (deleted parents, API limits, streaming moderation).

## 2. Method

```
TRAINING                                                  INFERENCE
parent + reply ──► BERT teacher (parent hidden 30 %) ─┐   reply ──► DistilBERT student ─► P(hate)
                         │ scored twice                │                        └────────► P(needs context)
                         ├─ p_ctx   (with parent)  ────┤──► KD target for the hate output (gated)
                         └─ p_alone (parent hidden)    │
                    delta = |p_ctx − p_alone|  ────────┘──► target for the needs-context output
```

1. **Context-privileged teacher.** BERT reads `[CLS] parent [SEP] reply [SEP]`. *Context
   dropout:* the parent is replaced by an empty string with probability 0.3, so the same teacher
   can also judge a reply alone.
2. **Free context-sensitivity signal.** For each training reply,
   δ = |σ(teacher(parent, reply)) − σ(teacher("", reply))|. No extra human annotation is needed.
3. **Student with two outputs, reading only the reply:**
   * hate output: gated KD from p_ctx (our reliability gate) + focal loss on the gold labels,
     which were annotated with the parent visible;
   * needs-context output: binary cross-entropy against δ (weight 0.5). δ is close to 0 for most
     replies, so it is rescaled so its 95th percentile maps to 1 (`ctx_target_norm`). The ranking is
     unchanged. On the synthetic check with 3 epochs this raised the head's AUROC from 0.66 to 0.90.
4. **Use:** the student answers hate / not hate and gives a needs-context score. Replies with a
   high score can be routed to a human or to a context-aware model. The rest are handled by the
   cheap student, which never needs the thread.

Code: `src/context_kd.py` (data, training, evaluation), `scripts/run_context.py` (one run),
`scripts/run_context_ladder.sh` (steps C0–C4, U1), `scripts/make_context_tables.py`,
`notebooks/gpu_context.ipynb`.

## 3. Ladder (each step adds one thing)

| Step | Change |
|---|---|
| C0 | DistilBERT, reply only, gold labels |
| C1 | + KD from a teacher that also reads only the reply (paper-style KD) |
| C2 | teacher reads parent + reply (context-privileged) |
| C3 | + reliability gate |
| C4 | + needs-context head = **CP-KD** |
| U1 | upper bound: the student also reads the parent (needs the thread at inference) |

## 4. Datasets (downloaded automatically from the authors' GitHub)

| Name | Source | Content | Size |
|---|---|---|---|
| `rc` | Yu, Blanco & Hong, NAACL 2022 (`xinchenyu/counter_context`) | Reddit reply + parent; hate / counter / neutral, labelled with the parent visible. Used as hate vs. not hate. | 3,325 / 713 / 713 |
| `ccc` | Xenos et al. 2021/2022 (`ipavlopoulos/context_toxicity`, CC0) | Civil Comments reply + parent; toxicity labelled **with** (~5 annotators) and **without** the parent | 9,929 posts; toxic = ≥ 2 of 5 annotators (score ≥ 0.4): 467 toxic with context, 914 label flips |

**Verify before reporting:** the `rc` label ids are not documented in the repository. `"0"`
was taken as *hate* from reading examples (`src/context_kd.py: RC_HATE_LABEL`). Check this
against the paper or its code. `ccc` has no official split; we use a stratified **60/10/30** split
(seed 42, stratified by label and flip), so the test set holds about 140 toxic and 274
context-dependent replies. With the usual "> 0.5" rule (≥ 3 of 5 annotators) only 119 replies
(1.2 %) are toxic, too few to train on (a CPU pilot teacher predicted "not toxic" for
everything). Report the ≥ 0.4 rule as a stated design decision.

## 5. Evaluation

* Hate detection on the test set: macro-F1, hate-F1, ECE.
* **Context-dependent slice (ccc):** accuracy on replies whose human label flips with context.
* **Needs-context flag (ccc):** AUROC / AP of the student's needs-context score for finding the
  human label flips, compared with the student's own uncertainty 1 − |2p − 1| (the obvious
  baseline) and with the teacher's δ on the test set (an oracle that needs the parent).
* **Deferral:** macro-F1 on the remaining replies after sending the 5 / 10 / 20 % most-flagged
  ones to a human.
* **Cost:** latency and input tokens of the reply-only student vs. the pair-reading U1 / teacher.

## 6. What to expect (honest)

* A reply-only student **cannot fully recover** a label that depends on a parent it never
  sees. The synthetic check confirms this: the teacher with the parent scores 100 % macro-F1,
  the same teacher without it 78 %, and the reply-only students 80–84 %. Large accuracy gains
  for C2–C4 over C1 are therefore not guaranteed. Pavlopoulos et al. (2020) also found that
  context barely improved classifiers.
* The contribution is in **knowing when context is needed** (flag + deferral) at no
  inference cost, and in measuring how much of the context-aware teacher's knowledge a reply-only
  student can absorb (C2/C4 vs. C1, gap to U1).
* On the synthetic data the needs-context head matched, but did not beat, the uncertainty
  baseline (AUROC 0.900 vs. 0.896), because there every ambiguous reply is also a
  context-dependent one. Whether the head beats uncertainty on **real** data (ccc) is the key
  experimental question.

## 7. Novelty check (October 2026 searches)

| Prior work | What it does | Difference |
|---|---|---|
| Pavlopoulos et al., ACL 2020 | context effect on toxicity labels; context-aware classifiers | no distillation; context needed at inference |
| Xenos et al., WOAH 2021 / First Monday 2022 | context-sensitivity estimation **supervised by human double annotations** (CCC); KD used to **augment training data** | we obtain the context-sensitivity signal from a context-dropout teacher without double annotation, and distil it into a reply-only hate classifier. **Read the full paper to confirm** (full text could not be opened here). |
| Yu et al., NAACL 2022; Bourgeade et al., LREC-COLING 2024; CoSyn 2023 | context-aware hate / abuse detection | the thread is required at inference; no distillation |
| Snell et al., "Learning by Distilling Context", 2022 | distils instructions/prompts into LLM weights | generation tasks, fixed context; not per-sample conversational context or hate |
| Piot & Parapar, ECIR 2025; SHIELD 2024; HateCOT 2024 | distil LLM *explanations* for hate speech | explanations, not conversational context |
| Gui et al., IEEE TCSS 2026 (base) | robust KD for hate speech | no context at all |

In these searches we found no work that distils a context-privileged teacher into a reply-only
hate-speech student, or that learns a "needs context" signal from a teacher's with/without-parent
gap. Phrase this as "to the best of our knowledge".

## 8. How to run (Kaggle / Colab)

Open `notebooks/gpu_context.ipynb` (GPU + Internet on) and run it from the top. Or:

```bash
DATASETS="rc ccc" SEEDS="42 1 2" OUT=runs bash scripts/run_context_ladder.sh
python scripts/make_context_tables.py --runs runs          # runs/context_ladder.md
```
