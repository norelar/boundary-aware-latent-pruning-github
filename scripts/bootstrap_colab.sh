#!/usr/bin/env bash
set -euo pipefail

python -m pip install --upgrade pip
python -m pip install -e ".[all]"
python -m compileall -q src

echo "Installation complete. Edit a file under configs/ and run latent-pruning validate-config."
