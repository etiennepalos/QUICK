# Independent absolute and field-response references

Run from the QUICK repository root after completing the native external-field
campaign for each desired backend. Select a Python interpreter with NumPy and
PySCF installed; the QUICK build does not depend on either package.

```bash
PYSCF_PYTHON=/path/to/pyscf-environment/bin/python
"$PYSCF_PYTHON" -B validation/external_field/validate_pyscf_field_response.py \
  --quick-reports /path/to/field-cpu/report.json /path/to/field-cuda/report.json \
  --python "$PYSCF_PYTHON" --workdir /path/to/new-independent-reference
```

The default cases cover restricted HF, unrestricted charged HF, a mixed-SP
Cartesian-d basis, and PBE0. Every supplied QUICK report must contain all
selected cases. Use `--cases` to select a smaller explicit set. CPU and CUDA
reports share independent references when their complete inputs match.

The worker reads the exact Gaussian contractions from this checkout's QUICK
`.BAS` files, including the first element before the first `****` delimiter.
It uses Cartesian orbitals and QUICK's Bohr conversion. It computes SCF energy
finite differences at 0.001 and 0.0005 bohr, with PySCF level-5 XC quadrature.
The parser and temporary factory are local to the worker process; installed
PySCF files are unchanged. Source, basis, scientific libraries, report, and
raw-result fingerprints are recorded.

`absolute-field/report.json` preserves the existing independent absolute
comparison and its original gates: HF energy 1e-6 hartree, HF gradient 1e-5
hartree/bohr, and DFT energy/gradient 5e-5 in the corresponding atomic units.
The overall `report.json` adds exact zero-field controls and reports their
absolute discrepancies using those same gates. It separately compares the
zero-to-field energy change (1e-6 hartree) and gradient change (1e-6
hartree/bohr), with thresholds declared before running the references.

`passed` requires every original absolute and new control check. A false
absolute result remains false and produces exit status 2. The distinct
`incremental_field_validation_passed` additionally summarizes the field
response while retaining every HF/UHF absolute gate and every structural,
source, and oracle check; it excludes only explicitly enumerated DFT absolute
comparisons. It does not establish absolute DFT force agreement. The QUICK SG1
and PySCF XC quadratures differ, so absolute PBE0 discrepancies must be
reported separately from the incremental applied-field response. The original
absolute report and all original failed checks remain visible and unchanged.

The script reads existing QUICK outputs and launches independent CPU reference
calculations only. It does not rerun QUICK or use a GPU. Use a new workdir for
each run; do not reuse or overwrite a failed reference directory.
