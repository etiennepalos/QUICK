# Finite ddX discretization convergence

`validate_continuum_convergence.py` runs a predeclared JSON plan with a retained
QUICK solvation API client. It records every input, total energy, full nuclear
gradient, SCF convergence, GPU dispatch counters and runtime hashes. Runs are
sequential and restricted to one CPU thread by the runner. Existing unsuccessful
comparisons are retained; an operationally successful SCF does not qualify a
continuum discretization.

The energy and maximum gradient refinement gates are **1e-5 Eh** and
**5e-5 Eh/bohr**. A selected protocol requires two successive harmonic refinements
with the cavity and density-integration grids held fixed, plus a volume-grid
check. These are finite, molecule-specific checks, rather than a proof of the
infinite-order limit or a change to QUICK's defaults.

## Reproduction

The runner accepts the existing client built from
`quick_solvation_api_driver.f90` and a plan containing `case`, `points` and
`comparisons`. Each point defines `id`, `model` (`COSMO` or `PCM`) and `grid`
(`lmax`, `nlebedev`, `nrad`, `nang`). Each comparison names its left and right
point IDs, model and category. The case uses the same method, basis, charge,
multiplicity and atom format as `validate_solvation.py`.

```bash
python3 validation/solvation/validate_continuum_convergence.py \
  --driver "$SOLVATION_DRIVER" --basis "$QUICK_BASIS" \
  --plan "$PLAN_JSON" --workdir "$NEW_OUTPUT_DIRECTORY" \
  --launcher 'taskset -c 30' --timeout 600
```

The output directory must not exist. `protocol-before-science.json` preserves the
initial controls and runtime identity; `partial-report.json` is updated after
each completed point; `report.json` includes the final comparisons and exceptions.
A failed refinement returns a nonzero status while retaining the completed data.
A plan without refinement pairs is an operational bridge and cannot qualify a
protocol. GPU jobs should run under an exclusive device allocation; the shared
parser verifies native volume and Coulomb-cache counters.

## WATER27 water-dimer study, 15 October 2026

The case is neutral singlet HF/STO-3G with 14 Cartesian AOs. The coordinates are
retained verbatim in the archived plans and inputs. Fixed controls are dielectric
78.3553, SCF limit 250, density RMS 1e-10, integral/gradient cutoffs 1e-12,
ddX tolerance 1e-10, radius scale 1.1, eta 0.1, COSMO shift -1 or PCM shift 0,
switching 0, kappa 0, native incore 0, maximum iterations 200, DIIS 20, native
threads 1, and no applied field. No radii, model definition or host conventions
were adjusted to obtain a smaller error. The COSMO host factor remains
`(epsilon - 1) / epsilon`; the existing all-space regular-solid-harmonic density
projection is retained.

The accepted CPU and CUDA libraries come from the `9d63c228` field-Hessian
checkpoint. The five solvation production modules match the source hashes in
the earlier continuum campaign. Repeating the historical CPU COSMO
L14/N5810/volume100x434 point reproduces the total energy and all 18 gradient
components bit for bit. Detailed source, driver, basis and dynamic-library
identities are preserved in the archived JSON reports.

Initial one-thread CPU results:

| Model and refinement | Fixed grids | Absolute energy change / Eh | Maximum gradient change / Eh/bohr | Gate |
| --- | --- | ---: | ---: | --- |
| COSMO L14 to L16 | N5810; volume100x434 | 6.24889e-5 | 1.73181e-4 | Fail |
| COSMO L16 to L18 | N5810; volume100x434 | 2.71526e-5 | 8.07426e-4 | Fail |
| PCM L10/N974 to L12/N1454 | volume100x434 | 9.24412e-7 | 2.19486e-5 | Pass |
| PCM L12/N1454 to L14/N2030 | volume100x434 | 2.54530e-5 | 6.74092e-5 | Fail |
| PCM volume100x434 to 150x974 | L14/N2030 | 1.12236e-6 | 5.35345e-6 | Pass |
| PCM volume150x974 to 200x1454 | L14/N2030 | 4.78866e-9 | 3.14206e-8 | Pass |

All eight initial SCFs converge in 18 cycles and pass operational checks. The
PCM L10/L12/L14 sequence changes harmonic order and cavity quadrature together;
it does not isolate harmonic truncation. The volume results support using
150x974 for a further fixed-N5810 harmonic sequence, with 200x1454 retained as
its denser check. Neither model is qualified by the initial campaign. A cause
for the high-order sensitivity is not established by these comparisons.

The subsequent GPU sequence isolates harmonic refinement at **N5810 and
volume150x974 throughout**:

| PCM harmonic refinement | Absolute energy change / Eh | Maximum gradient change / Eh/bohr | Gate |
| --- | ---: | ---: | --- |
| L12 to L14 | 2.70818e-5 | 6.12963e-5 | Fail |
| L14 to L16 | 5.66197e-5 | 2.10572e-4 | Fail |
| L12 to L16 | 8.37014e-5 | 2.33766e-4 | Fail |

All three GPU SCFs converge in 18 cycles. Native volume counters are
`[19, 19, 1]`; each run builds one Coulomb cache, reuses it 18 times, performs
19 matrix-vector products and records zero direct calls and zero fallbacks.
Wall times are 34.79, 42.18 and 50.55 seconds respectively, under the stated
one-thread host/device allocation. These are execution diagnostics, without a
performance-benchmark claim.

The CPU/CUDA bridge uses identical PCM L14/N2030/volume200x1454 inputs. The
absolute energy difference is 4.01337e-9 Eh and the maximum gradient difference
is 5.53606e-9 Eh/bohr; all operational checks pass. A separate diagnostic energy
gate of 1e-9 Eh is missed, while its 1e-8 Eh/bohr gradient gate passes. Both
residuals and that diagnostic failure are retained in `cpu-gpu-bridge-v1.json`.
The continuum refinement gates above remain unchanged.

**Outcome:** no PCM or COSMO protocol is qualified for this dimer by the bounded
study. Finite pairwise sensitivity grows at the highest tested orders. The
volume check is specific to L14/N2030; a higher-order/N5810 volume limit has not
been established. The results support preserving the opt-in implementation and
its existing accuracy evidence while withholding a converged-dimer or continuum
BOMD claim. Identifying or changing the density-projection/cavity formulation
would require a separate, explicit model investigation. No model change is
justified by these discretization comparisons alone.

The archived study is
`adaptive-electrostatics-20261015/continuum-convergence` beside the worktrees.
It contains the predeclared `plan-v1.json`, `plan-gpu-bridge-v1.json` and
`plan-gpu-pcm-fixedN-v1.json`; complete `cpu-v1`, `gpu-bridge-v1` and
`gpu-pcm-fixedN-v1` reports; exact validator/client source snapshots; and the
combined `assessment-v1.json`. The historical failed campaigns remain intact.

## Explicit Coulomb continuation, 15 October 2026

The preceding unsuccessful study uses the unchanged default
`DDX_SOURCE=REGULAR`. A separate frozen-density investigation found growing
high-order regular multipoles from density outside its parent sphere. The
installed Psi4/ddX source path also integrates the regular solid harmonics over
all space; this observation does not establish a QUICK-only implementation
defect. The ddX theory assumes compact charge support. Gaussian molecular
density has exterior tails, and removing those tails would change its charge.

The opt-in `DDX_SOURCE=COULOMB` preserves all density and instead continues the
parent-sphere source kernel outside radius R as
`4*pi/(2*l+1)*(R/r)^(l+1)*Y_lm(direction)`. Inside, the regular expression is
unchanged. Values are continuous at R; the interior and exterior radial
derivatives have distinct one-sided limits. The energy, Fock matrix and nuclear
gradient use the same continued source. Parent-following volume nodes keep the
source kernel fixed under a nuclear displacement, while the existing AO,
partition and moving-grid derivatives still contribute. This is an explicit
finite functional, inspired by the exterior Coulomb treatment in PySCF; it is
neither a new-method claim nor a claim of numerical equivalence to every PySCF
ddPCM/ddCOSMO setting. The default arithmetic and historical failures above
are preserved.

The frozen `continuum-source-v6` CPU/CUDA runtimes were used for the following
neutral HF/STO-3G dimer results, with the same geometry, dielectric, radii and
solver controls as above. The original gates remain 1e-5 Eh / 5e-5 Eh/bohr.

| Source COULOMB refinement | Fixed grids | Absolute energy change / Eh | Maximum gradient change / Eh/bohr | Gate |
| --- | --- | ---: | ---: | --- |
| PCM L12 to L14 | N5810; volume150x974 | 2.41965e-6 | 1.94667e-5 | Pass |
| PCM L14 to L16 | N5810; volume150x974 | 3.36731e-6 | 1.35292e-5 | Pass |
| COSMO L12 to L14 | N5810; volume150x974 | 2.02934e-6 | 6.30204e-6 | Pass |
| COSMO L14 to L16 | N5810; volume150x974 | 9.67167e-7 | 4.81695e-6 | Pass |
| PCM volume150x974 to 200x1454 | L16/N5810 | 2.90119e-7 | 9.38521e-8 | Pass |
| COSMO volume150x974 to 200x1454 | L16/N5810 | 2.87127e-7 | 9.54899e-8 | Pass |

A lower-cost search retains the failed PCM L10 to L12/N2030 energy change
1.02221e-5 Eh and L8 to L12 maximum gradient change 5.13978e-5 Eh/bohr.
Increasing harmonic order at the same N2030 cavity grid closes the finite
qualification without weakening either gate:

| Source COULOMB refinement | Fixed grids | Absolute energy change / Eh | Maximum gradient change / Eh/bohr | Gate |
| --- | --- | ---: | ---: | --- |
| PCM L12 to L14 | N2030; volume150x974 | 2.39699e-6 | 2.13035e-5 | Pass |
| PCM L14 to L16 | N2030; volume150x974 | 3.33050e-6 | 1.30142e-5 | Pass |
| PCM L12 to L16 | N2030; volume150x974 | 5.72749e-6 | 2.80771e-5 | Pass |
| PCM N2030 to N5810 | L16; volume150x974 | 7.03432e-8 | 5.62690e-6 | Pass |
| PCM volume150x974 to 200x1454 | L16/N2030 | 2.90141e-7 | 9.34434e-8 | Pass |
| COSMO L8 to L10 | N2030; volume150x974 | 2.15993e-6 | 9.97003e-6 | Pass |
| COSMO L10 to L12 | N2030; volume150x974 | 5.29388e-6 | 1.21890e-5 | Pass |
| COSMO L8 to L12 | N2030; volume150x974 | 3.13396e-6 | 2.21591e-5 | Pass |
| COSMO N2030 to N5810 | L12; volume150x974 | 1.07069e-6 | 1.67210e-5 | Pass |
| COSMO volume150x974 to 200x1454 | L12/N2030 | 2.87299e-7 | 9.45299e-8 | Pass |

For this dimer, PCM L16/N2030 and COSMO L12/N2030 with volume150x974 are
qualified finite working settings for the explicit continued source. The
successive, spanning, cavity and volume comparisons above do not establish an
infinite-order limit, transfer to other geometries/bases, or validate a continuum
BOMD trajectory. They do not qualify the original REGULAR source.

Independent validation also checks the implementation rather than only its
refinement behavior:

- Scalar basis values and derivatives through l=18, inside/outside/at the
  boundary, at the origin, and at extreme finite distance: independent m=0
  normalized value error 6.28e-14, analytic-gradient error 3.20e-15; all-component
  two-step finite-difference error below 9.67e-10. Interior/default preservation
  is exact.
- CPU molecular campaign: 112 retained SCFs and 898 checks pass. Water all nine
  force components and three dimer coordinates pass two-step self-consistent
  finite differences for both models; the smaller-step maximum error is
  8.35e-8 Eh/bohr. RHF/UHF frozen-density energy derivatives agree with their
  Fock contributions to 1.83e-9. Old-v5 versus new omitted/REGULAR CPU total
  energies and complete gradients are identical.
- Matched GPU campaign: 14 SCFs and 176 checks pass. Source2 CPU/GPU total
  energies and full gradients differ by at most 4.02e-9 Eh and 5.55e-9
  Eh/bohr. GPU density variational error is below 4.70e-10. Native volume and
  Coulomb counters prove execution without a fallback. Old/default/REGULAR GPU
  differences remain below the declared 1e-10 Eh / 1e-9 Eh/bohr preservation
  gates.
- The strict parser passes 32 cases with ddX enabled and the same 32 cases
  with ddX disabled. Same-job REGULAR to COULOMB to REGULAR restores energy
  within 8.64e-14 Eh and the matrix within 2.92e-11 while preserving density,
  stored Fock/energy and API step exactly. Two-rank CPU MPI propagates source2
  correctly and matches serial energy/forces within 2.85e-14/9.29e-15. The
  enabled/disabled gas-phase water energies and gradients are identical.

The complete source2 archive is
`adaptive-electrostatics-20261015/continuum-source-validation-v1`. It includes
all plans, full energies/gradients, controls, source/runtime identities, failed
lower-order comparisons, `cross-campaign-comparisons-v1.json`, and an immutable
seal. Reproduction entry points are `validate_ddx_source_convergence.py`,
`validate_ddx_source_molecular_convergence.py`,
`validate_ddx_source_gpu_convergence.py`,
`validate_ddx_source_smoke_convergence.py`, and the finite-plan runner above.
