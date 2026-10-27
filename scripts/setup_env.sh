#!/usr/bin/env bash
set -euo pipefail
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
env_dir="${1:-$repo_root/.venv}"
constraints="$repo_root/requirements-lock-cu124.txt"
python3.10 -m venv "$env_dir"
"$env_dir/bin/python" -m pip install --upgrade pip==26.2.1 setuptools wheel packaging ninja -c "$constraints"
"$env_dir/bin/python" -m pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124 -c "$constraints"
"$env_dir/bin/python" -m pip install -r "$repo_root/requirements.txt" -c "$constraints"
MAX_JOBS="${MAX_JOBS:-8}" "$env_dir/bin/python" -m pip install flash-attn==2.7.4.post1 --no-build-isolation -c "$constraints"
# Pin the exact public SageAttention source used by the validated Wan runtime.
MAX_JOBS="${MAX_JOBS:-4}" EXT_PARALLEL="${EXT_PARALLEL:-1}" "$env_dir/bin/python" -m pip install --no-build-isolation --no-deps -c "$constraints" \
  'git+https://github.com/cooper1637/SageAttention.git@d9704247a5139ab4c03bf7fc6b35cc0e2cbb5ea4'
"$env_dir/bin/python" "$repo_root/scripts/check_environment.py"
