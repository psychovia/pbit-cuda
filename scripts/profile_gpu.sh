#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

# Process-local WSL libraries; never replace any system driver/library.
if [[ -d /usr/lib/wsl/lib ]]; then
    wsl_driver_dir="${PBIT_WSL_DRIVER_DIR:-}"
    if [[ -z "$wsl_driver_dir" ]]; then
        for driver in /usr/lib/wsl/drivers/nv*.inf_*; do
            if [[ -f "$driver/libnvidia-ptxjitcompiler.so.1" &&
                  -f "$driver/libcuda.so.1.1" &&
                   -f "$driver/libcuda_loader.so" ]] &&
               cmp -s "$driver/libcuda_loader.so" /usr/lib/wsl/lib/libcuda.so.1.1; then
                wsl_driver_dir="$driver"
                break
            fi
        done
    fi
    if [[ -z "$wsl_driver_dir" ]]; then
        echo 'Cannot identify the matching WSL CUDA/PTX driver directory; set PBIT_WSL_DRIVER_DIR.' >&2
        exit 1
    fi
    export LD_LIBRARY_PATH="/usr/lib/wsl/lib:$wsl_driver_dir:/usr/local/cuda-13.4/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi

if [[ -n "${PBIT_PYTHON:-}" ]]; then
    profile_python="$PBIT_PYTHON"
elif [[ -x /home/psychovia/miniforge3/envs/pbit-cugraph/bin/python ]]; then
    profile_python=/home/psychovia/miniforge3/envs/pbit-cugraph/bin/python
else
    profile_python=python3
fi
exec "$profile_python" scripts/profile_gpu.py "$@"
