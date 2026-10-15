# XC quadrature and PBE0 force convergence

The two validators separate agreement on the **same quadrature** from
convergence toward a denser quadrature. Neither changes a previous failed
check or changes the default SG0/SG1 arithmetic.

`validate_pbe0_xc_convergence.py` exports the actual QUICK SG1 coordinates
and partitioned weights at every displaced geometry. Independent Cartesian
PySCF PBE0 SCFs use the exact QUICK STO-3G contractions, first on those
exported grids and then on PySCF levels 5, 7, and 9. Every nuclear gradient
component is differentiated from energies at 0.001 and 0.0005 bohr, with
zero and mixed applied uniform fields. Grid motion and partition-weight
response are included by rebuilding the grid at each geometry.

The native calculation tightens `CUTOFF`, `GRADCUTOFF`, `XCCUTOFF`, and
`BASISCUTOFF` to `1e-12`, with `DENSERMS=1e-10`. The reference SCF energy
tolerance is `1e-12` hartree and the orbital-gradient tolerance `1e-9`.
The diagnostic reproduces native weight skipping, but independent full AO
evaluation does not reproduce native bin/primitive or low-density screening.
The remaining energy difference is recorded rather than assumed zero.

The preserved 2026-10-15 baseline contains 296 geometry/quadrature records:
74 QUICK SCFs and 296 independent PySCF SCFs, all on one CPU thread. All
nine matched-SG1 gradient components agree within `4.40e-8` hartree/bohr
at the smaller displacement. The largest native SG1 discrepancy against
level 9 is `2.693e-4` hartree/bohr. Level 7 to 9 changes the complete
gradient by at most `3.19e-8` hartree/bohr and energy by `1.96e-9` hartree.
This demonstrates SG1 quadrature error for this fixture; tightening integral
screening does not make SG1 a denser grid. Sixteen baseline checks remain
failed against the unchanged `5e-5` absolute force/refinement gate. The
companion historical record also retains the eight earlier CPU/CUDA
zero/applied-field absolute failures.

## Explicit native refinement

The paired opt-in controls select an unpruned angular rule on every radial
shell, using the existing SG1 Euler–Maclaurin radial transformation, atomic
radii, molecular partition, grid response, binning, screening, and CPU/GPU
XC evaluators:

```text
DFT PBE0 BASIS=STO-3G GRADIENT XC_RADIAL=100 XC_ANGULAR=434 MATRIXZERO=1e-14
```

Both controls must be present and positive, with radial count at most 400
and an existing supported Lebedev angular count. The controls require DFT
and reject `SG0`. Angular rules 74, 230, and 266 are rejected because they
contain negative weights that the existing XC weight screening would skip.
SG0 and SG1 retain their default grids when the controls are absent. A new
ordinary job resets temporary grid dimensions after a refined job.

`validate_native_xc_convergence.py` compares full native gradients and
energies for explicit radial/angular grids against the preserved highest
independent grid. Its default sequence is `50x194 75x302 100x434 150x590`.
Start with `--central-only` to establish cost and locate a useful protocol;
omit that option to differentiate all nine native force components at both
displacements on the final selected grid. Every candidate retains its own
checks, including failed coarse candidates. A successful refined candidate
does not turn the old SG1 checks into passes.

Grid density must be refined together with the existing `MATRIXZERO`
threshold. Native packing discards individual quadrature weights below
this density/matrix threshold. Splitting a near-nuclear shell into many
small weights can therefore remove core contributions at a fixed default
threshold, even as angular force accuracy improves. Use `--matrixzero` to
study this explicitly; its default remains `1e-10`. Record both controls
with the selected quadrature, and retain any failures of the ordinary
threshold. An apparent force plateau alone is insufficient evidence of
joint energy/force convergence.

At `MATRIXZERO=1e-14`, the bounded STO-3G central-grid campaign passes all
46 checks with the original `5e-5` gates. The maximum errors against level 9
across zero/applied fields and both displacement references are:

| Native grid | Energy error (hartree) | Force error at 0.0005 bohr (hartree/bohr) |
|---|---:|---:|
| 75 × 302 | 1.45e-6 | 1.62e-5 |
| 100 × 434 | 2.92e-7 | 3.23e-6 |
| 150 × 590 | 2.41e-8 | 8.37e-7 |
| 200 × 974 | 1.85e-8 | 1.81e-7 |

The 100 × 434 grid also passes all nine native gradient-component energy
finite differences at both displacements and both fields: maximum error
`2.18e-7` at 0.001 bohr and `5.47e-8` at 0.0005 bohr. Tightening
`MATRIXZERO` from `1e-14` to `1e-16` on the 100 × 434 and 150 × 590 grids
changes energies by at most `9.72e-10` hartree and forces by `3.37e-11`
hartree/bohr. Earlier default-threshold refined-grid failures are preserved.

`validate_xc_basis_convergence.py` separately transfers the protocol to the
WATER27 water and water-dimer geometries with the exact QUICK def2-SVP basis,
25/50 Cartesian AOs, and the same mixed uniform field. It compares three
native grids against PySCF level 7/9 full gradients with grid response. The
independent applied-field gradient includes both AO-center derivatives of
the dipole operator and the classical `-Z F` term; selected-component
two-step energy differences check that reference extension. This is six
native central calculations rather than another full finite-difference
campaign. Its acceptance must be assessed independently of the STO-3G result.

The 2026-10-15 basis transfer passes all 44 checks. Maximum full-gradient
errors against the independently qualified level-9 reference are:

| Native grid | Water, 25 Cartesian AOs | Water dimer, 50 Cartesian AOs |
|---|---:|---:|
| 100 × 434 | 1.190e-5 | 1.020e-5 |
| 150 × 590 | 1.384e-6 | 3.009e-6 |
| 200 × 974 | 3.689e-7 | 7.768e-7 |

Values are in hartree/bohr. Across both molecules, level 7 to 9 changes
energy by at most `2.39e-9` hartree and full gradients by `4.53e-8`
hartree/bohr. The independent field-gradient extension passes its selected
energy finite differences within `1.51e-7` and `3.77e-8` hartree/bohr at
0.001 and 0.0005 bohr. The 150 × 590 grid with `MATRIXZERO=1e-14` provides
more accuracy margin for these polarized-basis fixtures than 100 × 434.
Production WATER20 configurations still need their own assessment.

Compile `quick_sg1_export_driver.f90` against the exact installed CPU library
and corresponding module files as described in `MATCHED_SG1.md`. The same
exporter accepts the new controls; it also exports the resulting grid.

```sh
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  /path/to/pyscf/python -B validate_pbe0_xc_convergence.py \
  --repo /path/to/QUICK --driver /path/to/quick-sg1-export \
  --quick-report /path/to/accepted-native-field/report.json \
  --higher-grid-report /path/to/preserved/absolute-field/report.json \
  --workdir /path/to/new-independent-grid-campaign

OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python3 -B validate_native_xc_convergence.py \
  --repo /path/to/QUICK --driver /path/to/new-quick-sg1-export \
  --reference /path/to/new-independent-grid-campaign/report.json \
  --workdir /path/to/new-native-grid-campaign --central-only
```

Each output directory must be new. Raw inputs, native outputs, actual grids,
independent SCF logs/results, loaded-library hashes, source hashes, and
individual error checks are retained. The accepted native library must
match the inspected source. GPU runs require a separately authorized GPU
window and the explicit `--allow-gpu` argument.

These small water/STO-3G checks do not establish convergence of a different
basis, water cluster, solvent model, timestep, or trajectory observable.
Those remain separate production-protocol checks.
