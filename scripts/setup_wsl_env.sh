#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
conda_bin="${CONDA_EXE:-$HOME/miniconda3/bin/conda}"
env_file="$repo_root/configs/environment-wsl.yml"
env_name="quant-competition-wsl"

if [[ ! -x "$conda_bin" ]]; then
  echo "Conda executable not found: $conda_bin" >&2
  echo "Set CONDA_EXE to the WSL Conda executable and rerun." >&2
  exit 1
fi

echo "Using repository: $repo_root"
echo "Using conda: $conda_bin"
if "$conda_bin" env list | awk '{print $1}' | rg -x "$env_name" >/dev/null; then
  "$conda_bin" env update --name "$env_name" --file "$env_file" --prune
else
  "$conda_bin" env create --name "$env_name" --file "$env_file"
fi

"$conda_bin" run -n "$env_name" python -m ipykernel install --user \
  --name "$env_name" --display-name "Python (quant-competition-wsl)"

echo
echo "Environment created/updated: $env_name"
echo "Run the GPU checks with:"
echo "  $conda_bin run -n $env_name python $repo_root/scripts/check_wsl_gpu.py"
