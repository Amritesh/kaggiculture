#!/usr/bin/env bash
# Prepare a fresh machine (GPU box included) to train and evaluate the agent.
set -euo pipefail

PY=${PY:-python3}
$PY -m venv .venv
. .venv/bin/activate
pip install --quiet --upgrade pip

# Torch: install the build that matches the machine.
#   CUDA 12.1 -> pip install torch --index-url https://download.pytorch.org/whl/cu121
#   CPU/Apple -> pip install torch
pip install --quiet torch numpy scikit-learn jsonschema

# The official engine is vendored by clone so the simulator is bit-exact.
mkdir -p vendor
if [ ! -d vendor/kaggle-environments ]; then
  git clone --quiet https://github.com/Kaggle/kaggle-environments.git vendor/kaggle-environments
  git -C vendor/kaggle-environments checkout --quiet 9b6bedeaeb478067a335ff5776b214fb436218f6
fi

echo "ready. sanity check:"
.venv/bin/python -m unittest discover -s tests
