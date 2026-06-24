#!/usr/bin/env bash
set -euo pipefail

# Compare CPU and GPU OEPROP values for the external-grid acetone regression
# cases. This script is intended for a CUDA/HIP host; it does not build QUICK.
#
# Required environment:
#   CPU_QUICK=/path/to/cpu/install/bin/quick
#   GPU_QUICK=/path/to/gpu/install/bin/quick
#   QUICK_BASIS=/path/to/install/basis
#
# Optional:
#   QUICK_ROOT=/path/to/QUICK
#   WORKDIR=/private/tmp/quick_gpu_oeprop_validation

if [[ -z "${CPU_QUICK:-}" || -z "${GPU_QUICK:-}" || -z "${QUICK_BASIS:-}" ]]; then
  echo "Set CPU_QUICK, GPU_QUICK, and QUICK_BASIS before running." >&2
  exit 2
fi

if [[ -n "${QUICK_ROOT:-}" ]]; then
  root="${QUICK_ROOT}"
else
  script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  root="$(cd "${script_dir}/../.." && pwd)"
fi

workdir="${WORKDIR:-/private/tmp/quick_gpu_oeprop_validation_$$}"
mkdir -p "${workdir}/cpu" "${workdir}/gpu"

compare_files() {
  local cpu_file="$1"
  local gpu_file="$2"
  local label="$3"
  python3 - "$cpu_file" "$gpu_file" "$label" <<'PY'
import math
import re
import sys

cpu_path, gpu_path, label = sys.argv[1:4]
num = re.compile(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[Ee][-+]?\d+)?")

def numbers(path):
    vals = []
    with open(path, "r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            vals.extend(float(x) for x in num.findall(line))
    return vals

cpu = numbers(cpu_path)
gpu = numbers(gpu_path)
if len(cpu) != len(gpu):
    raise SystemExit(f"{label}: value-count mismatch CPU={len(cpu)} GPU={len(gpu)}")

diffs = [abs(a - b) for a, b in zip(cpu, gpu)]
max_abs = max(diffs) if diffs else 0.0
rms = math.sqrt(sum(d*d for d in diffs) / len(diffs)) if diffs else 0.0
print(f"{label}: n={len(diffs)} max_abs={max_abs:.6e} rms={rms:.6e}")
if max_abs > 2.0e-7:
    raise SystemExit(f"{label}: max_abs exceeds 2.0e-7")
PY
}

run_case() {
  local input="$1"
  local artifact="$2"
  local label="$3"

  cp "${root}/test/${input}" "${workdir}/cpu/"
  cp "${root}/test/${input}" "${workdir}/gpu/"

  (
    cd "${workdir}/cpu"
    "${CPU_QUICK}" "${input}" > "${input}.stdout"
  )
  (
    cd "${workdir}/gpu"
    "${GPU_QUICK}" "${input}" > "${input}.stdout"
  )

  compare_files "${workdir}/cpu/${artifact}" "${workdir}/gpu/${artifact}" "${label}"
}

run_case "esp_grid_acetone_b3lyp_def2svp.in" "esp_grid_acetone_b3lyp_def2svp.esp" "ESP CPU vs GPU"
run_case "efield_grid_acetone_b3lyp_def2svp.in" "efield_grid_acetone_b3lyp_def2svp.efield" "EFIELD CPU vs GPU"
run_case "efg_grid_acetone_b3lyp_def2svp.in" "efg_grid_acetone_b3lyp_def2svp.efg" "analytic EFG CPU vs GPU"

run_case "efg_grid_numerical_acetone_b3lyp_def2svp.in" \
  "efg_grid_numerical_acetone_b3lyp_def2svp.efg" "numerical EFG CPU vs GPU"

compare_files "${workdir}/gpu/efg_grid_acetone_b3lyp_def2svp.efg" \
  "${workdir}/gpu/efg_grid_numerical_acetone_b3lyp_def2svp.efg" \
  "GPU analytic EFG vs GPU numerical EFG"

echo "GPU OEPROP validation completed in ${workdir}"
