#!/usr/bin/env bash
# Full experiment grid. Run sections separately so a Colab/Kaggle session
# time-out never loses more than one section:
#
#   bash scripts/run_all.sh smoke      # 2 min, CPU ok  - sanity check
#   bash scripts/run_all.sh se_main    # ~2.5 h on a T4 - Table "main results" SE (3 seeds)
#   bash scripts/run_all.sh se_ablation# ~2-3 h (3 seeds) - ablation table
#   bash scripts/run_all.sh dv_main    # ~6 h (5 folds)  - set FOLDS="0" for ~1.2 h
#   bash scripts/run_all.sh tables     # aggregate everything into runs/tables.md
#   bash scripts/run_all.sh pilot      # CPU-only pilot, from-scratch models (~3-4 h)
#
# Teachers are cached in runs/teachers/ and re-used automatically, so
# ablations that only change the student loss do NOT retrain BERT.
set -euo pipefail
cd "$(dirname "$0")/.."
PY="python scripts/run_experiment.py"
SEEDS=${SEEDS:-"42 1 2"}
FOLDS=${FOLDS:-"0 1 2 3 4"}
OUT=${OUT:-runs}

case "${1:-}" in
smoke)
  for m in baseline rkd srkd; do $PY --dataset smoke --method $m --out_dir $OUT/smoke; done
  python scripts/make_tables.py --runs $OUT/smoke
  ;;

se_main)
  for s in $SEEDS; do
    for m in baseline rkd srkd; do
      $PY --dataset se --method $m --seed $s --run_name se_${m}_s$s --out_dir $OUT
    done
  done
  ;;

se_ablation)   # every row changes ONE thing w.r.t. "srkd" (final) or "rkd" (paper); all seeds
  for s in $SEEDS; do
    # N1 alone: emoji->text with the paper's KD loss (no gate)
    $PY --dataset se --method srkd --kd_gate false --seed $s --run_name se_abl_n1_only_s$s --out_dir $OUT
    # N2 alone: reliability gate with the paper's preprocessing
    $PY --dataset se --method rkd --kd_gate true --seed $s --run_name se_abl_n2_only_s$s --out_dir $OUT
    # components that did NOT help in the pilot (reported as negative results)
    $PY --dataset se --method srkd --kd_revise true --seed $s --run_name se_abl_plus_revise_s$s --out_dir $OUT
    $PY --dataset se --method srkd --kd_cb true     --seed $s --run_name se_abl_plus_cb_s$s     --out_dir $OUT
    # keeping hashtags (shortcut analysis), with and without hashtag dropout
    $PY --dataset se --method srkd --preprocess spn --seed $s --run_name se_abl_keep_hashtags_s$s --out_dir $OUT
    $PY --dataset se --method srkd --preprocess spn --hashtag_dropout 0.5 --seed $s --run_name se_abl_hashtag_dropout_s$s --out_dir $OUT
    # calibrated gate (teacher temperature-scaled on validation data before gating)
    $PY --dataset se --method srkd_cal --seed $s --run_name se_abl_cal_gate_s$s --out_dir $OUT
    # is the gate just "a smaller lambda"?
    $PY --dataset se --method srkd --kd_gate false --lam 0.5 --seed $s --run_name se_abl_lam05_s$s --out_dir $OUT
  done
  ;;

dv_main)
  for f in $FOLDS; do
    for m in baseline rkd srkd; do
      $PY --dataset dv --method $m --fold $f --seed 42 --run_name dv_${m}_f${f}_s42 --out_dir $OUT
    done
  done
  ;;

dv_ablation)
  f=0; s=42
  $PY --dataset dv --method srkd --kd_gate false --fold $f --run_name dv_abl_n1_only_f${f}_s$s --out_dir $OUT
  $PY --dataset dv --method rkd --kd_gate true   --fold $f --run_name dv_abl_n2_only_f${f}_s$s --out_dir $OUT
  $PY --dataset dv --method srkd --kd_cb true    --fold $f --run_name dv_abl_plus_cb_f${f}_s$s --out_dir $OUT
  ;;

pilot)   # CPU-only pilot on the REAL datasets with from-scratch models (no GPU / no BERT weights)
  P="$PY --pilot true --out_dir ${OUT}_pilot"
  for m in rkd ours ours_emoji ours_hd baseline; do $P --dataset se --method $m --seed 42 --run_name se_${m}_s42; done
  # preprocessing ablation with the paper's KD loss (isolates N1 variants)
  $P --dataset se --method rkd --preprocess spn                       --run_name se_abl_spn_paperkd_s42
  $P --dataset se --method rkd --preprocess spn --hashtag_dropout 0.5 --run_name se_abl_spnhd_paperkd_s42
  $P --dataset se --method rkd --preprocess spn_emoji                 --run_name se_abl_emoji_paperkd_s42
  # KD-loss ablation with the paper's preprocessing (isolates N2 + N3)
  $P --dataset se --method rkd --kd_gate true --kd_revise true --kd_cb true --run_name se_abl_kd_only_s42
  # leave-one-out from ours_emoji
  $P --dataset se --method ours_emoji --kd_gate false   --run_name se_abl_emo_no_gate_s42
  $P --dataset se --method ours_emoji --kd_revise false --run_name se_abl_emo_no_revise_s42
  $P --dataset se --method ours_emoji --kd_cb false     --run_name se_abl_emo_no_cb_s42
  for m in rkd ours_emoji ours_hd baseline; do $P --dataset dv --method $m --fold 0 --run_name dv_${m}_f0_s42; done
  for s in 1 2; do for m in rkd ours_emoji ours_hd baseline; do $P --dataset se --method $m --seed $s --run_name se_${m}_s$s; done; done
  # final method (gate only) = what `srkd` does
  G="--method ours_emoji --kd_revise false --kd_cb false"
  for s in 42 1 2; do $P --dataset se $G --seed $s --run_name se_ours_gate_s$s; done
  $P --dataset dv $G --fold 0 --run_name dv_ours_gate_f0_s42
  $P --dataset dv --method ours_emoji --kd_revise false --fold 0 --run_name dv_ours_gate_cb_f0_s42
  # calibrated gate (tested, not adopted)
  for s in 42 1 2; do $P --dataset se --method srkd_cal --seed $s --run_name se_srkd_cal_s$s; done
  $P --dataset dv --method srkd_cal --fold 0 --run_name dv_srkd_cal_f0_s42
  python scripts/make_tables.py --runs ${OUT}_pilot --compare se_rkd se_ours_gate --compare se_rkd se_ours_emoji \
      --compare se_rkd se_ours_hd --compare se_baseline se_ours_gate
  ;;

tables)
  python scripts/make_tables.py --runs $OUT --compare se_rkd se_srkd --compare se_baseline se_srkd --compare dv_rkd dv_srkd
  ;;

*)
  echo "usage: $0 {smoke|se_main|se_ablation|dv_main|dv_ablation|pilot|tables}"; exit 1 ;;
esac
