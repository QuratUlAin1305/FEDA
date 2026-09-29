#!/usr/bin/env bash
# Full pipeline, run from the repository root.
set -e
for lo in 0 1400 2800 4200; do
  python conformers.py $lo $((lo+1400))   # conf_<lo>.pkl
  python features.py $lo                  # feat_<lo>.npz
done
python exp_main.py      # all configurations of Table II + repeated CV -> results/results.json
python stats.py         # Table II values and FEDA-vs-ESPH test (reads results/results.json)
