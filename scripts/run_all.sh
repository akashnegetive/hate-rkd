#!/usr/bin/env bash
# Full experiment grid. Run sections separately so a Colab/Kaggle session
# time-out never loses more than one section:
#
#   bash scripts/run_all.sh smoke      # 2 min, CPU ok  - sanity check
#   bash scripts/run_all.sh se_main    # ~2.5 h on a T4 - Table "main results" SE (3 seeds)
#   bash scripts/run_all.sh se_ablation# ~1 h            - ablation table
#   bash scripts/run_all.sh dv_main    # ~6 h (5 folds)  - set FOLDS="0" for ~1.2 h
#   bash scripts/run_all.sh tables     # aggregate everything into runs/tables.md
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
  for m in baseline rkd ours; do $PY --dataset smoke --method $m --out_dir $OUT/smoke; done
  python scripts/make_tables.py --runs $OUT/smoke
  ;;

se_main)
  for s in $SEEDS; do
    for m in baseline rkd ours; do
      $PY --dataset se --method $m --seed $s --run_name se_${m}_s$s --out_dir $OUT
    done
  done
  ;;

se_ablation)   # seed 42 only; every row changes ONE thing w.r.t. "ours" or "rkd"
  s=42
  # (N1) semantic-preserving normalisation alone, paper's KD loss
  $PY --dataset se --method ours --kd_gate false --kd_revise false --kd_cb false --seed $s --run_name se_abl_spn_only_s$s --out_dir $OUT
  # (N2+N3) our KD loss alone, paper's destructive denoising
  $PY --dataset se --method rkd --kd_gate true --kd_revise true --kd_cb true --seed $s --run_name se_abl_kd_only_s$s --out_dir $OUT
  # leave-one-out from the full method
  $PY --dataset se --method ours --kd_gate false   --seed $s --run_name se_abl_no_gate_s$s   --out_dir $OUT
  $PY --dataset se --method ours --kd_revise false --seed $s --run_name se_abl_no_revise_s$s --out_dir $OUT
  $PY --dataset se --method ours --kd_cb false     --seed $s --run_name se_abl_no_cb_s$s     --out_dir $OUT
  # is the gate just "a smaller lambda"? -> global lambda lowered instead of gating
  $PY --dataset se --method ours --kd_gate false --lam 0.5 --seed $s --run_name se_abl_lam05_s$s --out_dir $OUT
  # loss-term ablations (paper Table VII style) for our method
  $PY --dataset se --method ours --use_kd false    --seed $s --run_name se_abl_no_ldst_s$s   --out_dir $OUT
  $PY --dataset se --method ours --lam 1.0 --kd_gate false --seed $s --run_name se_abl_no_lstd_s$s --out_dir $OUT
  ;;

dv_main)
  for f in $FOLDS; do
    for m in baseline rkd ours; do
      $PY --dataset dv --method $m --fold $f --seed 42 --run_name dv_${m}_f${f}_s42 --out_dir $OUT
    done
  done
  ;;

dv_ablation)
  f=0; s=42
  $PY --dataset dv --method ours --kd_gate false --kd_revise false --kd_cb false --fold $f --run_name dv_abl_spn_only_f${f}_s$s --out_dir $OUT
  $PY --dataset dv --method rkd --kd_gate true --kd_revise true --kd_cb true --fold $f --run_name dv_abl_kd_only_f${f}_s$s --out_dir $OUT
  ;;

tables)
  python scripts/make_tables.py --runs $OUT --compare se_rkd se_ours --compare se_baseline se_ours
  ;;

*)
  echo "usage: $0 {smoke|se_main|se_ablation|dv_main|dv_ablation|tables}"; exit 1 ;;
esac
