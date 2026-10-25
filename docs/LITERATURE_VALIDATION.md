# Literature validation of each design choice

**Purpose.** The supervisor asked that every individual choice (emoji handling, hashtag
handling, confidence gating, loss changes, class weighting) be justified from prior work before
any combination is claimed. This document lists, for each choice, the prior work that
**supports** it, the prior work that **contradicts or qualifies** it, what **our own experiments**
showed, and the resulting **verdict and novelty claim**.

**How the sources were checked (October 2026).** Every paper below was confirmed to exist
(title, authors, venue) through web search. The ACL Anthology and Semantic Scholar could not be
opened from the build environment, so for several papers only the abstract or quoted passages
were read. These are marked *(abstract/quotes only)*. Read the full text of those before quoting
numbers from them in the thesis.

---

## Summary table

| # | Choice | Prior work individually? | Literature verdict | Our pilot result | Final decision |
|---|---|---|---|---|---|
| 1 | Emoji → text instead of deletion | **Yes**: OffensEval winners, sentiment ablations; Hatemoji shows emoji-based hate exists | Mostly supports, one counter-example | Better teacher on 3/3 seeds | **Keep (N1)** |
| 2a | Hashtag → segmented words | **Yes**: NULI, ekphrasis, many SemEval systems | Supported generally, but on HatEval hashtags are a known label shift (MITRE 2019) | Hurts on SemEval (−7 M-F1) | **Drop**; report as confirmation of MITRE |
| 2b | Hashtag dropout | Related: token/feature dropout, shortcut-mitigation work; not found for hashtags specifically | Untested idea | Does not fix the shift | **Drop** (negative result) |
| 3 | Per-sample confidence gating of KD | **Yes**: CCKD (2021), Hard-Gate KD (EMNLP 2022), CA-MKD (2022), Weighted Soft Labels (ICLR 2021) | Strongly supported (vision/NLG) | +2.5 on top of N1 | **Keep (N2)**, cited as an adaptation of CCKD |
| 4 | Revising wrong teacher labels | **Yes**: Knowledge/Logits Adjustment (Wen et al. 2021), Label Revision (Lan et al. 2024) | Supported in vision | Hurts (−1.1) | **Drop**; explain via label noise |
| 5 | Class-balanced distillation loss | **Yes**: Balanced KD (Zhang et al. 2021), class-balanced reweighting in long-tail KD | Supported in long-tailed vision | Raises hate recall, lowers F1 | **Drop**; explain by interaction with focal α |
| 6 | Focal loss (kept from the paper) | **Yes**: WOAH 2024 imbalance study; DCL (Lu et al. 2023) | Supported | (paper's choice, unchanged) | Keep |
| 7 | Calibration (ECE) for KD students | **Yes**: EMNLP 2022 (emotion), Hard-Gate KD; hate-speech calibration studies | Supported as an evaluation | ECE 5.6 → 3.9 | Keep (evaluation) |
| 8 | Cross-dataset testing | **Yes**: Swamy et al. 2019, Fortuna et al., Yin & Zubiaga 2021 review | Standard practice | +4.5 cross-dataset | Keep (evaluation) |

**On the combination.** We found no prior work that combines emoji normalisation with
reliability-gated distillation for compressing hate-speech detectors (searches covered hate,
offensive and toxic language together with distillation). The closest hate-speech KD works are
RKD (the base paper), Deng et al. 2023 (multi-teacher KD for Chinese offensive language), DDML
2025 (multi-student KD) and BAT-KD (bias-adaptive temperature). None of them uses per-sample
teacher-reliability gating or emoji-preserving normalisation. A search cannot prove absence, so
the thesis should phrase this as "to the best of our knowledge".

---

## 1. Emoji → text (N1)

**What we do.** Replace each emoji with its CLDR name (`😡` → "enraged face") and collapse
repeats, instead of deleting it (the paper's `sps-rem` step).

**Supporting prior work**
* **NULI at SemEval-2019 Task 6** (Liu, Li & Zou, 2019; ACL S19-2011), the **winner of
  OffensEval 2019 sub-task A**, fine-tuned BERT after preprocessing that included **emoji
  substitution** and **hashtag segmentation** (as reported in the OffensEval overview,
  Zampieri et al. 2019). *(abstract/quotes only)*
* Several **SemEval-2020 Task 12 (OffensEval 2020)** systems replaced emojis with their text
  description using the `emoji` Python package before BERT's WordPiece tokenizer, e.g.
  **Kungfupanda** ("convert all emojis to words with corresponding semantic meanings") and **UPB**.
  *(abstract/quotes only)*
* **Hatemoji** (Kirk, Vidgen, Röttger, Thrush & Hale, NAACL 2022) shows that hate expressed
  *through emojis* is common and that existing detectors fail on it, so emojis carry
  hate-relevant signal that deletion throws away.
* Sentiment studies: **"An Effective BERT-Based Pipeline for Twitter Sentiment" (Sensors)**
  converts emojis and emoticons to text, with an ablation of that step. **Scientific Reports 2025,
  "Sentiment analysis of emoji fused reviews"** compares removal against dictionary-based text
  replacement and argues that removal discards signal. *(abstract/quotes only)*

**Contradicting / qualifying prior work**
* **BNU-HKBU UIC NLP Team 2 at SemEval-2019 Task 6** reported that "translating emoji into
  words … and removing the hashtag gave worse results than using the original data" with BERT.
  *(quote only)*
* **Reimann & Dakota, KONVENS 2021** ("Examining the Effects of Preprocessing on the Detection
  of Offensive Language in German Tweets") found BERT models **far more sensitive** to each
  preprocessing step than classical classifiers, and inconsistent between models. Preprocessing
  effects must therefore be measured, not assumed. *(abstract only)*

**Our evidence.** The emoji→text teacher beat the paper's teacher on all three seeds (+1.0,
+0.8, +2.7 macro-F1) with lower ECE (7.0 vs. 14.1).

**Verdict.** Supported by the literature and by our data. **Not novel on its own.** It is
standard practice in OffensEval systems, but the base paper deletes emojis, and we show that
restoring them helps inside its KD pipeline. Cite NULI, Kungfupanda/UPB and Hatemoji; also cite
BNU-HKBU and KONVENS 2021 to show the effect is setup-dependent, which is why we ran ablations.

---

## 2. Hashtags: segmentation (and our hashtag dropout)

**What we tried.** `#BanIslam` → "ban islam" (camel-case split, else `wordsegment`), optionally
dropping each hashtag with probability 0.5 during training.

**Supporting prior work for segmentation**
* **ekphrasis** (Baziotis, Pelekis & Doulkeridis, DataStories at SemEval-2017 Task 4) was built
  for Twitter preprocessing, with hashtag word segmentation as a core feature. It is widely
  reused, e.g. in SemEval-2018 Task 1 systems and in 2020 sarcasm detection.
* **NULI** (OffensEval 2019 winner) segmented hashtags. OffensEval 2020 systems split
  camel-cased hashtags (`#KeithEllisonAbuse` → "keith ellison abuse"). *(quotes only)*

**Contradicting prior work, and the explanation of our result**
* **MITRE at SemEval-2019 Task 5** (Gertner, Henderson, Merkhofer, Marsh, Wellner & Zarrella,
  2019; ACL S19-2080) reported that on **HatEval** tweets with **#BuildThatWall/#BuildTheWall
  were labelled hate 98 % of the time in training but only 35 % in the test set**, and that the
  word "bitch" shifted from 78 % to 43 %. A second HatEval system paper (arXiv 2011.13238)
  reports the same figures. *(quotes only)*
* **LT3 at SemEval-2019 Task 5** observed good training performance but a considerable recall
  drop on the HatEval test set. *(abstract only)*
* General shortcut-learning work in hate speech (keyword-sampled corpora, identity-term bias,
  causal debiasing, e.g. "Mitigating Biases in Hate Speech Detection from a Causal Perspective",
  Findings of EMNLP 2023; "Dynamically Refined Regularization for Improving Cross-corpora Hate
  Speech Detection", 2022) shows that classifiers latch onto corpus-specific tokens that do not
  transfer.

**Our evidence.** Keeping segmented hashtags cost −7.2 macro-F1 (seed 42), and hashtag dropout
did not fix it (−11.1). Our counts on the TweetEval version of the split agree with MITRE:
#buildthatwall is 100 % hate in train (331 tweets) and 24 % in test (464 tweets). The same
pattern holds for #maga (89 % → 30 %), #nodaca (100 % → 32 %), #walkaway (91 % → 16 %) and
#americafirst (97 % → 22 %).

**Verdict.** Hashtag segmentation is well supported in general, but on HatEval the hashtags are
a **known** shortcut, so the paper's deletion is the right choice *for this benchmark*.
**Correction to our earlier docs:** the hashtag shift is **not a new finding**. It confirms
MITRE (2019). What we add is (i) the measurement over more hashtags, (ii) evidence that the
shift explains why RKD's `htg-rem` step works, which the base paper attributes only to "noise",
and (iii) the negative result that hashtag dropout does not remove the shortcut. Report hashtag
segmentation as a tested and rejected alternative.

---

## 3. Per-sample confidence gating of distillation (N2)

**What we do.** λ_i = λ · p_true,i: the distillation weight of sample i is scaled by the
teacher's probability of the *true* class, so samples where the teacher is unsure or wrong rely
more on the hard-label focal loss.

**Supporting prior work (the mechanism exists already)**
* **Confidence Conditioned Knowledge Distillation (CCKD)**, Mishra & Sundaram, arXiv
  2107.06993 (2021): "uses the teacher's confidence in the correct class to build
  sample-specific losses and targets". **This is the closest prior work to our gate.** It was
  evaluated on MNIST, Fashion-MNIST and CIFAR-10. *(abstract only)*
* **Hard Gate Knowledge Distillation**, Lee, Tian, Zhao, Cheung & Zhang, **EMNLP 2022**
  (2022.emnlp-main.665): an *instance-specific hard gate* that switches between learning from the
  teacher and from the training data, using the teacher's calibrated score. It improves
  generalisation **and lowers calibration error** in NLG. This is NLP evidence for gating and
  for our calibration observation. *(abstract only)*
* **CA-MKD, Confidence-Aware Multi-Teacher KD** (arXiv 2201.00007; authors/venue to verify,
  believed to be Zhang, Chen & Wang, ICASSP 2022): sample-wise teacher reliability from the cross-entropy between teacher prediction
  and ground truth; large weights for teacher predictions close to the one-hot label.
* **Weighted Soft Labels**, Zhou et al., **ICLR 2021** ("Rethinking Soft Labels for Knowledge
  Distillation: A Bias-Variance Tradeoff Perspective"): the bias-variance trade-off of soft
  labels varies **per sample**, and weighting soft labels per sample beats both uniform KD and
  filtering samples out.

**Our evidence.** Adding the gate to emoji normalisation gave +2.5 macro-F1 (seed 42). Over
three seeds, SR-KD reached 60.8 vs. 55.7 for RKD (significant on each seed), with ECE 3.9 vs. 5.6.

**Verdict.** Strongly supported. **Not a new mechanism.** Present it as *adapting CCKD-style
reliability weighting to binary, sigmoid-based hate-speech KD with a focal hard-label loss*.
In our searches, no hate-speech KD work used it (RKD, Deng et al. 2023, DDML 2025 and BAT-KD use
fixed λ, multiple teachers or students, or adaptive temperature). Also add a related-work
paragraph contrasting it with BAT-KD's per-sample *temperature*.

---

## 4. Revising wrong teacher soft labels (tested, dropped)

**Supporting prior work**
* **Knowledge Adjustment / Logits Adjustment** in "Preparing Lessons: Improve Knowledge
  Distillation with Better Supervision", Wen, Lai & Qian, *Neurocomputing* 454 (2021): rectifies
  incorrect teacher logits using the ground-truth label (CIFAR, Tiny-ImageNet).
* **Label Revision**, Lan et al., arXiv 2404.03693 (2024): mixes the one-hot label into wrong
  soft labels, plus data selection.

**Qualifying evidence.** Zhou et al. (ICLR 2021) show that treating "bad" samples harshly, by
filtering, *hurts* distillation. On noisy-labelled hate-speech data, a "wrong" teacher is often
right and the label wrong (HatEval's annotation shift above), so pushing towards the label can
hurt.

**Our evidence.** Removing revision *improved* the student (+1.1 at seed 42). The teacher was
wrong on only 2–3 % of training tweets.

**Verdict.** Supported in vision with clean labels; **not supported here**. Report it as a
negative ablation with the label-noise explanation. The gate already handles the useful part:
it lowers the teacher's weight on exactly those samples without forcing a target.

---

## 5. Class-balanced distillation loss (tested, dropped)

**Supporting prior work**
* **Balanced Knowledge Distillation for Long-tailed Learning** (Zhang et al., arXiv
  2104.10510; *Neurocomputing* 2023): instance-balanced classification loss plus a
  **class-balanced distillation loss**.
* Long-tail KD work notes that a teacher trained on imbalanced data passes its majority bias to
  the student and adds class-balanced reweighting of the distillation term (summarised in
  "Balancing Knowledge Distillation for Imbalance Learning with Bilevel Optimization", arXiv 2605.17839).
* In NLP, focal loss combined with KL distillation improved minority-class F1 for DistilBERT and
  ALBERT emotion recognition (*Scientific Reports* 2025). *(abstract only)*

**Qualifying evidence.** The **WOAH 2024** study of class imbalance in abusive-language
detection ("A Study of the Class Imbalance Problem in Abusive Language Detection") found most
imbalance methods **sensitive to hyper-parameters** and that the best method depends on the
dataset. *(abstract only)*

**Our evidence.** On Davidson, class-balanced KD raised hate recall by about 16 points but
reduced precision more (F1 −0.7 to −2.3). On SemEval it hurt (−2.4). The reason: the paper's
focal α already re-weights the classes, and inverse-frequency weights (≈ 8.7× for hate on
Davidson) on top of it over-correct.

**Verdict.** Supported in long-tailed vision; **not supported in this pipeline**, because
imbalance is already handled by focal loss. Report as a negative ablation. A softer weighting
(e.g. square-root inverse frequency or "effective number" weights, Cui et al., CVPR 2019) is a
possible future-work variant.

---

## 6. Focal loss (inherited from the base paper)

* **WOAH 2024** imbalance study: random oversampling, **focal loss** and lexicon-based
  augmentation were the most versatile, and focal loss often gave peak performance on binary
  abusive-language datasets. *(abstract only)*
* **DCL** (Lu et al., IEEE/ACM TASLP 2023), a baseline in the base paper, also uses focal loss
  for hate-speech imbalance.

No change by us, so no novelty claim.

---

## 7. Calibration (ECE) as an evaluation of distilled students

* **Hosseini & Caragea, EMNLP 2022**, "Calibrating Student Models for Emotion-related Tasks":
  the distillation process affects student calibration, and they measure ECE in- and
  out-of-domain.
* **Hard Gate KD** (EMNLP 2022) reports ECE and finds gating lowers it, consistent with our
  ECE 5.6 → 3.9.
* "Rethinking the Knowledge Distillation From the Perspective of Model Calibration" (arXiv
  2111.01684) and "A Statistical Perspective on Distillation" (ICML 2021) link teacher
  probability quality to student quality.
* Hate-speech calibration: "To BAN or not to BAN: Bayesian Attention Networks for Reliable Hate
  Speech Detection" (arXiv 2007.05304) uses ECE; "Don't Go To Extremes" (arXiv 2402.11406)
  studies calibration of LLMs on implicit hate.

**Verdict.** Well-established metric. Applying it to hate-speech KD students is an evaluation
contribution, not a method contribution.

---

## 8. Cross-dataset generalisation

* **Swamy et al. 2019** (cross-dataset testing; title/venue to verify, believed to be "Studying
  Generalisability across Abusive Language Detection Datasets", CoNLL 2019): macro-F1 drops of 2–30 points across the Waseem, Davidson, Founta and
  Zampieri datasets (as summarised in the review below).
* **Fortuna et al.** trained 1,698 classifiers across datasets and found that generalisation
  depends on both dataset and model.
* **Yin & Zubiaga, "Towards generalisable hate speech detection: a review on obstacles and
  solutions"** (arXiv 2102.08886; journal version believed to be PeerJ CS 2021, to verify).

**Verdict.** Standard evaluation; the base paper omits it. Our +4.5 (SE → DV) and +2.2 (DV → SE)
are an evaluation contribution.

---

## What to claim in the thesis (revised wording)

> We re-implement RKD and examine its design choices against prior work. Following
> OffensEval practice (NULI, OffensEval-2020 systems) and evidence that emojis carry hateful
> meaning (Hatemoji), we translate emojis into text instead of deleting them. Following
> confidence-conditioned distillation (CCKD; Hard-Gate KD; CA-MKD), we weight the distillation
> term per sample by the teacher's probability of the true class. To the best of our knowledge
> this combination has not been applied to hate-speech model compression. It improves the
> distilled student by 5.1 macro-F1 on SemEval-2019 (3 seeds, significant on each) with better
> calibration and cross-dataset transfer, at no inference cost. We also test three further
> choices supported by prior work in other settings (hashtag segmentation, soft-label revision,
> class-balanced distillation) and show why each fails here. For hashtags, this confirms the
> HatEval train/test shift first reported by MITRE (2019).

## Reference list (with links found during verification)

* Gui, Deng & Zhao. Improving Hate Speech Detection via Robust Knowledge Distillation. IEEE TCSS 13(4), 2026.
* Liu, Li & Zou. NULI at SemEval-2019 Task 6. SemEval 2019, pp. 87–91. https://aclanthology.org/S19-2011/
* Zampieri et al. SemEval-2019 Task 6 (OffensEval) overview. https://arxiv.org/abs/1903.08983
* Kungfupanda at SemEval-2020 Task 12. https://arxiv.org/abs/2004.13432
* UPB at SemEval-2020 Task 12. https://arxiv.org/abs/2010.13609
* BNU-HKBU UIC NLP Team 2 at SemEval-2019 Task 6. https://www.researchgate.net/publication/333718000
* Kirk, Vidgen, Röttger, Thrush & Hale. Hatemoji. NAACL 2022. https://aclanthology.org/2022.naacl-main.97/
* Reimann & Dakota. Examining the Effects of Preprocessing on the Detection of Offensive Language in German Tweets. KONVENS 2021. https://aclanthology.org/2021.konvens-1.14
* An Effective BERT-Based Pipeline for Twitter Sentiment. Sensors. https://pdfs.semanticscholar.org/f5f9/677f548ef3a2fe8cd86053cd6bea803f3668.pdf
* Sentiment analysis of emoji fused reviews using machine learning and BERT. Sci. Rep. 2025. https://www.nature.com/articles/s41598-025-92286-0
* Baziotis, Pelekis & Doulkeridis. DataStories at SemEval-2017 Task 4 (ekphrasis). https://github.com/cbaziotis/ekphrasis
* Gertner et al. MITRE at SemEval-2019 Task 5. SemEval 2019, pp. 453–459. https://aclanthology.org/S19-2080/
* Towards Interpretable Multilingual Detection of Hate Speech against Immigrants and Women in Twitter at SemEval-2019 Task 5. https://arxiv.org/abs/2011.13238
* LT3 at SemEval-2019 Task 5. https://aclanthology.org/S19-2077/
* Basile et al. SemEval-2019 Task 5 overview. https://aclanthology.org/S19-2007/
* Mitigating Biases in Hate Speech Detection from a Causal Perspective. Findings of EMNLP 2023. https://aclanthology.org/2023.findings-emnlp.440.pdf
* Dynamically Refined Regularization for Improving Cross-corpora Hate Speech Detection. https://arxiv.org/abs/2203.12536
* Mishra & Sundaram. Confidence Conditioned Knowledge Distillation. https://arxiv.org/abs/2107.06993
* Lee et al. Hard Gate Knowledge Distillation. EMNLP 2022. https://arxiv.org/abs/2210.12427
* Confidence-Aware Multi-Teacher Knowledge Distillation (CA-MKD). https://arxiv.org/abs/2201.00007
* Zhou et al. Rethinking Soft Labels for Knowledge Distillation: A Bias-Variance Tradeoff Perspective. ICLR 2021. https://arxiv.org/abs/2102.00650
* Wen, Lai & Qian. Preparing Lessons: Improve Knowledge Distillation with Better Supervision. Neurocomputing 454, 2021. https://arxiv.org/abs/1911.07471
* Lan et al. Improve Knowledge Distillation via Label Revision and Data Selection. https://arxiv.org/abs/2404.03693
* Zhang et al. Balanced Knowledge Distillation for Long-tailed Learning. https://arxiv.org/abs/2104.10510
* Balancing Knowledge Distillation for Imbalance Learning with Bilevel Optimization. https://arxiv.org/abs/2605.17839
* Optimised knowledge distillation for efficient social media emotion recognition using DistilBERT and ALBERT. Sci. Rep. 2025. https://www.nature.com/articles/s41598-025-16001-9
* A Study of the Class Imbalance Problem in Abusive Language Detection. WOAH 2024. https://aclanthology.org/2024.woah-1.4
* Hosseini & Caragea. Calibrating Student Models for Emotion-related Tasks. EMNLP 2022. https://aclanthology.org/2022.emnlp-main.629
* Rethinking the Knowledge Distillation From the Perspective of Model Calibration. https://arxiv.org/abs/2111.01684
* To BAN or not to BAN: Bayesian Attention Networks for Reliable Hate Speech Detection. https://arxiv.org/abs/2007.05304
* Don't Go To Extremes: Calibration Limitations of LLMs in Implicit Hate Speech Detection. https://arxiv.org/abs/2402.11406
* Yin & Zubiaga. Towards generalisable hate speech detection: a review on obstacles and solutions. https://arxiv.org/abs/2102.08886
* Deng et al. Enhancing Offensive Language Detection with Data Augmentation and Knowledge Distillation. Research 2023. https://pmc.ncbi.nlm.nih.gov/articles/PMC10506735/
* DDML: Multi-Student Knowledge Distillation for Hate Speech. Entropy 27(4), 2025. https://www.ncbi.nlm.nih.gov/pmc/articles/PMC12025758/
* BAT-KD (bias-aware adaptive temperature KD for hate speech), in workshop proceedings arXiv 2603.18786.
* Cui et al. Class-Balanced Loss Based on Effective Number of Samples. CVPR 2019 (for future work).
