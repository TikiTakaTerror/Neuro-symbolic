#!/bin/zsh
# Reproduce the full experiment, then build the tables and figures.
#   NeurASP (Category A)        : n = 1, 2, 3
#   DeepProbLog exact (Cat B)   : n = 1, 2  (n = 3 does not complete)
#   LNN (Category C)            : reasoning-only, 1 digit
set -u
cd "${0:A:h}"                                   # project root (this script's directory)
NPY=venvs/venv_neurasp/bin/python
DPY=venvs/venv_deepproblog/bin/python
LPY=venvs/venv_lnn/bin/python
TO=$(command -v gtimeout || command -v timeout) # GNU timeout (gtimeout on macOS)
if [ -z "$TO" ]; then
  echo "ERROR: GNU timeout not found. Install it (macOS: brew install coreutils) and re-run."
  echo "It is required so DeepProbLog at 3 digits is stopped at the budget instead of running indefinitely."
  exit 1
fi
mkdir -p logs

for n in 1 2 3; do
  echo "[run] NeurASP n=$n"
  $TO 1400 $NPY shared/run_neurasp.py --n $n > logs/run_neurasp_n$n.log 2>&1
done

for n in 1 2; do
  echo "[run] DeepProbLog exact n=$n"
  $TO 1400 $DPY shared/run_deepproblog.py --method exact --n $n > logs/run_dpl_exact_n$n.log 2>&1
done

# DeepProbLog exact at 3 digits does not complete (the inference circuit does not
# compile within the budget). Run as a documented probe; exit 124 = timed out.
echo "[run] DeepProbLog exact n=3 (expected: does not complete)"
$TO 1300 $DPY shared/run_deepproblog.py --method exact --n 3 > logs/run_dpl_exact_n3.log 2>&1
rm -f results/curve_dpl_exact_n3_n3.csv   # failed probe leaves only an init point; not a result

echo "[run] LNN (Category C, reasoning-only)"
$LPY shared/run_lnn.py --eval_instances 1000 > logs/run_lnn.log 2>&1

echo "[run] building tables and figures"
$NPY shared/make_figures.py
echo "[run] done"
