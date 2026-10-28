#!/usr/bin/env bash
# Regenerates every result, table and figure of the paper (about 1.5 h on an Apple M4).
set -euo pipefail
cd "$(dirname "$0")/.."
python scripts/exp_validate.py
python scripts/make_kat.py --check
python scripts/exp_qp2024.py
python scripts/exp_security.py
python scripts/exp_misuse.py
python scripts/exp_robust.py
python scripts/exp_ablation.py
python scripts/exp_nist.py
python scripts/exp_nist_null.py
python scripts/exp_ml.py           # deep-learning attacks (PyTorch; GPU/MPS recommended, about 1-2 h)
python scripts/exp_speed.py        # run on an otherwise idle machine
python scripts/make_figures.py
python scripts/make_tables.py ../manuscript/build
