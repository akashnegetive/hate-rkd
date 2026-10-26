#!/usr/bin/env bash
# Context ladder: each step adds ONE thing (see src/context_kd.py).
#
#   C0  student_only          DistilBERT on the reply alone
#   C1  kd_reply_teacher      + KD from a BERT teacher that also sees only the reply (paper-style)
#   C2  kd_ctx_teacher        teacher reads "parent [SEP] reply" (context-privileged teacher)
#   C3  kd_ctx_gate           + reliability gate
#   C4  cpkd                  + "needs context" head = CP-KD (final)
#   U1  student_with_context  upper bound: student also reads the parent (needs the thread)
#
# Usage:  DATASETS="rc ccc" SEEDS="42 1 2" OUT=runs bash scripts/run_context_ladder.sh
#         python scripts/make_context_tables.py --runs $OUT
# Safe to re-run: finished steps are skipped. Same env options as run_ladder.sh
# (TEACHER_DIR, CLEAN_TEACHERS, EXTRA_ARGS, STEPS).
set -uo pipefail
cd "$(dirname "$0")/.."

DATASETS=${DATASETS:-"rc ccc"}
SEEDS=${SEEDS:-"42 1 2"}
OUT=${OUT:-runs}
TEACHER_DIR=${TEACHER_DIR:-$OUT/teachers}
CLEAN_TEACHERS=${CLEAN_TEACHERS:-0}
EXTRA_ARGS=${EXTRA_ARGS:-}
STEPS=${STEPS:-}

run() {   # run <step> <method>
  local step=$1 method=$2
  local rn="${DS}_${step}_f0_s${SEED}"
  if [ -n "$STEPS" ] && [[ " $STEPS " != *" $step "* ]]; then return; fi
  if [ -f "${OUT}/results/${rn}__student.json" ]; then echo "[skip] ${rn}"; return; fi
  echo ">>> ${rn}"
  # shellcheck disable=SC2086
  python scripts/run_context.py --dataset "$DS" --method "$method" --seed "$SEED" --run_name "$rn" \
      --out_dir "$OUT" --teacher_dir "$TEACHER_DIR" --skip_done $EXTRA_ARGS \
      || echo "!!! FAILED ${rn} (continuing)"
}

for DS in $DATASETS; do
  for SEED in $SEEDS; do
    run C0_student_only      student_only
    run C1_kd_reply_teacher  kd_reply_teacher
    run C2_kd_ctx_teacher    kd_ctx_teacher        # trains the context teacher
    run C3_kd_ctx_gate       kd_ctx_gate           # reuses it
    run C4_cpkd              cpkd
    run U1_student_with_context student_with_context
    if [ "$CLEAN_TEACHERS" = 1 ]; then
      rm -rf "${TEACHER_DIR}/${DS}_f0_"*"_s${SEED}_"*
    fi
  done
done
echo "=== CONTEXT LADDER DONE ==="
