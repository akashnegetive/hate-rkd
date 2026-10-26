#!/usr/bin/env bash
# Step-by-step research ladder: each step adds ONE change, so its effect can be measured.
#
#   L0  baseline       BERT / DistilBERT fine-tuned on raw text with CE (paper's baseline rows)
#   L1  rkd            the paper's method (RKD-T / RKD-S)                       = "before"
#   L2  emoji          L1 + emoji -> text                                        (N1)
#   L3  gate           L2 + reliability-gated distillation = SR-KD (final)      (N1 + N2) = "after"
#   X*  extras         L3 (or L1) + one more change, to show what does NOT help
#
# Usage (GPU, Colab or Kaggle):
#   DATASET=se SEEDS="42 1 2" bash scripts/run_ladder.sh
#   DATASET=dv FOLDS="0"      bash scripts/run_ladder.sh      # FOLDS="0 1 2 3 4" for full 5-fold CV
#   EXTRAS=0 ...                                              # main ladder only (L0-L3)
#   python scripts/make_ladder.py --runs $OUT                 # tables
#
# Safe to re-run after a disconnect: finished steps are skipped (--skip_done).
#   OUT          results folder (put it on Google Drive in Colab)        default runs
#   TEACHER_DIR  teacher checkpoints (local disk is fine; ~420 MB each)   default $OUT/teachers
#   CLEAN_TEACHERS=1  delete a seed's/fold's teachers once its steps are done (saves disk)
#   EXTRA_ARGS   passed to every run, e.g. "--pilot true" (CPU) or "--epochs 3"
#   STEPS        run only these steps, e.g. STEPS="L1_rkd X6_gate_without_emoji" (default: all)
set -uo pipefail
cd "$(dirname "$0")/.."

DATASET=${DATASET:-se}
SEEDS=${SEEDS:-"42 1 2"}
FOLDS=${FOLDS:-"0"}
OUT=${OUT:-runs}
TEACHER_DIR=${TEACHER_DIR:-$OUT/teachers}
EXTRAS=${EXTRAS:-1}
CLEAN_TEACHERS=${CLEAN_TEACHERS:-0}
EXTRA_ARGS=${EXTRA_ARGS:-}
STEPS=${STEPS:-}

run() {   # run <name> <args...>
  local name=$1; shift
  local rn="${DATASET}_${name}_f${FOLD}_s${SEED}"
  if [ -n "$STEPS" ] && [[ " $STEPS " != *" $name "* ]]; then return; fi
  if [ -f "${OUT}/results/${rn}__student.json" ]; then echo "[skip] ${rn}"; return; fi
  echo ">>> ${rn}"
  # shellcheck disable=SC2086
  python scripts/run_experiment.py --dataset "$DATASET" --fold "$FOLD" --seed "$SEED" \
      --run_name "$rn" --out_dir "$OUT" --teacher_dir "$TEACHER_DIR" --skip_done $EXTRA_ARGS "$@" \
      || echo "!!! FAILED ${rn} (continuing)"
}

if [ "$DATASET" = "dv" ]; then LOOP_FOLDS=$FOLDS; else LOOP_FOLDS=0; fi

for FOLD in $LOOP_FOLDS; do
  for SEED in $SEEDS; do
    # --- teacher: raw text, CE -----------------------------------------------
    run L0_baseline --method baseline
    # --- teacher: paper's denoising --------------------------------------------
    run L1_rkd      --method rkd
    [ "$EXTRAS" = 1 ] && run X6_gate_without_emoji --method rkd --kd_gate true
    # --- teacher: emoji -> text (shared by L2, L3 and most extras) -------------
    run L2_emoji    --method rkd --preprocess spn_emoji
    run L3_gate     --method srkd
    if [ "$EXTRAS" = 1 ]; then
      run X3_plus_revise        --method srkd --kd_revise true
      run X4_plus_class_balance --method srkd --kd_cb true
      run X5_calibrated_gate    --method srkd_cal
      run X7_lambda05_no_gate   --method srkd --kd_gate false --lam 0.5
      # --- teachers: hashtags kept ---------------------------------------------
      run X1_plus_hashtags        --method srkd --preprocess spn
      run X2_plus_hashtag_dropout --method srkd --preprocess spn --hashtag_dropout 0.5
    fi
    if [ "$CLEAN_TEACHERS" = 1 ]; then
      rm -rf "${TEACHER_DIR}/${DATASET}_f${FOLD}_"*"_s${SEED}_"*
    fi
  done
done
echo "=== LADDER DONE (${DATASET}) ==="
