# Selected-component matched-SG1 diagnostic

This separate diagnostic asks whether an absolute PBE0 force discrepancy
comes from using different XC quadratures. It preserves the independent
higher-grid report, its failed checks, and its raw oracle output. It does
not replace an absolute force-accuracy test or establish convergence of SG1.

Compile the small exporter with the compiler and serial CPU module files
matching the QUICK library being checked:

```sh
FC=/path/to/matching/gfortran
"$FC" -cpp -ffree-line-length-none \
    -I /path/to/QUICK/src/util \
    -I /path/to/build/amber-modules/quick/serial \
    validation/external_field/quick_sg1_export_driver.f90 \
    -L /path/to/build/src -Wl,-rpath,/path/to/build/src -lquick \
    -L /path/to/ddx/lib -Wl,-rpath,/path/to/ddx/lib -lddx \
    -o /path/to/quick-sg1-export
```

Use a Python environment with NumPy and PySCF. The higher-grid report is
the immutable `absolute-field/report.json` produced by
`validate_pyscf_field_response.py`, with its adjacent `oracle-output.json`.

```sh
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  /path/to/pyscf/bin/python -B \
    validation/external_field/diagnose_pyscf_matched_sg1.py \
    --driver /path/to/quick-sg1-export \
    --quick-report /path/to/accepted-native-field/report.json \
    --higher-grid-report /path/to/reference/absolute-field/report.json \
    --workdir /path/to/new-matched-grid-diagnostic --component 1z
```

The bounded default is neutral restricted PBE0/STO-3G, one O-z component,
at displacements 0.001 and 0.0005 bohr, both with and without the applied
field. `--case` selects the case name in the native report; `--component`
selects one one-based atom/axis, and `--steps` selects displacement sizes.
This performs ten native and ten independent SCFs with the default steps.
It rejects a reference basis that differs from the exact native `.BAS`
definition recorded in the accepted report. Basis, input/output, scientific
library, source, and loaded QUICK-library fingerprints are retained.

Each displaced geometry gets its own actual QUICK SG1 point coordinates
and full partitioned weights, exported after the API force call. PySCF runs
an independent Cartesian SCF on that geometry-specific quadrature. Energy
finite differences therefore include motion of the grid and partition
weights. A fixed-grid PySCF analytic gradient is not used.

The diagnostic reproduces QUICK's `weight < DMCutoff` skipping. Independent
PySCF evaluates full AOs; it does not reproduce native bin/primitive AO
screening or the native low-density skip. Record the resulting residual
instead of assuming a bitwise match. The original higher-grid absolute
discrepancies remain visible and failed. Agreement here establishes
consistency for the selected quadrature/component, not physical convergence
of all molecular forces or accuracy of untested components.

The companion `validate_pyscf_field_response.py` validator supplies the
higher-grid report. Its failed checks and adjacent raw oracle are read-only
inputs here. Use matching locally generated accepted native reports and raw
directories; the records retain absolute run paths, so arbitrary archive
relocation is not handled by this diagnostic. No higher-grid failure is
removed, waived or converted into a matched-grid acceptance pass.
