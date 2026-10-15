# Applied fields and optional ddX: implementation acceptance, 2026-10-15

**Engineering acceptance complete for the documented scope; numerical and publication limits remain.** This report
records the implemented behavior and completed engineering/scientific checks.
It does not establish production trajectory ensembles, universal default-grid
accuracy or a whole-BOMD speedup. The companion JSON pins exact workstation
reports and raw-data provenance. No manuscript text is included.

Author: **etiennepalos <33858083+etiennepalos@users.noreply.github.com>**.
Branch: `feature/gpu-electrostatics-bomd`; parent:
`4a5a14d5d593438a962ab699bc6f19742d945d4a`. The release commit is the
containing Git commit. Its final SHA and verified remote state belong in
the external `publication-receipt.json` artifact after publication; this
report makes no push claim or self-referential commit assertion.

## Implemented scope

Uniform static fields now have analytic Gaussian-moment nuclear derivatives,
including classical nucleus/external-charge contributions. Optional CMake
`QUICK_DDX=ON` couples the continuous QM density to electrostatic equilibrium
ddCOSMO/ddPCM. Both can feed QUICK's energy/gradient API and velocity-Verlet
BOMD, with live-density ESP/EFIELD/EFG sampling and no extra property SCF.
The BOMD force consumer rejects unconverged SCF results before accepting a step.

| Component | Implementation |
|---|---|
| Applied field | Native FP64 CUDA AO matrix and electronic gradient; unchanged CPU moment-integral reference; classical charge terms on CPU |
| Continuum electrostatics | Continuous density, exact nuclear multipoles, full-density reaction operator and variational energy; analytic AO-center, cavity, moving-node and partition gradients |
| CUDA continuum work | Volume AO density/reaction matrix/derivative operators; reverse Coulomb matrix and derivatives; geometry cache with cuBLAS matrix-vector reuse |
| CPU continuum work | ddX forward/adjoint solves, cavity/grid/partition construction, harmonic projection; host OpenMP settings restored after native calls |
| Parallel interface | CPU serial and MPI2 accepted; CUDA serial and CUDA MPI1 accepted on one physical GPU; job-local communicator ownership retained |
| Published methods | Existing gas-phase ESP/MKS/OEPROP methods retained; optional continuum is OFF by default; external Amber RESP/rwRESP fitters were not executed |

The solvent operator uses the current full density, including alpha+beta for
unrestricted calculations. It is added after the ordinary incremental operator
has been saved and MPI-reduced; the variational solvation energy is added once.
The force refreshes the retained final density without another SCF. Independent
frozen-density directions test `dE_solv = Tr(V_solv dP)`, including both spin
channels, while checking that diagnostic calls leave the ordinary Fock/energy
state and API SCF count unchanged.

| Source responsibility | Repository path |
|---|---|
| Keywords and compatibility guards | `src/modules/quick_method_module.f90` |
| Applied-field moments and native CUDA kernels | `src/modules/quick_oei_module.f90`, `src/gpu/gpu_external_field.cu` |
| Native ddX model/forward/adjoint solve | `src/modules/quick_ddx_backend_module.f90` |
| Volume partition, normalized AOs and Coulomb operators | `src/modules/quick_solvation_grid_module.f90`, `quick_solvation_ao_module.f90`, `quick_solvation_coulomb_module.f90` |
| CUDA volume/reverse-Coulomb operators and cache | `src/gpu/gpu_solvation.cu`, `gpu_solvation_coulomb.cu`, `gpu_solvation_atomic.h` |
| SCF/gradient integration and lifecycle | `src/modules/quick_solvation_module.f90`, SCF operator modules, `src/modules/include/gradient.fh`, public API and main teardown |

The new buffers/cache are independent of published ESP/XC/OEI storage. Cache
keys cover active basis data, coordinates, cavity nodes and device; density
and reaction-charge changes reuse integrals at fixed geometry. Geometry moves
invalidate the cache, and cleanup precedes GPU teardown. Its budget is the
smaller of 8 GiB and one-quarter free GPU memory, including persistent workspace;
size/allocation limits select the direct CUDA operator. Physical device OOM
and second-device invalidation were not deliberately induced.

See the [interface guide](applied-fields-and-ddx.md),
[field validation](../../validation/external_field/README.md),
[continuum validation](../../validation/solvation/README.md), and
[BOMD driver](../../validation/oeprop_bomd/README.md) for runnable controls.

## Units, signs and model definition

For field F and origin r0, QUICK adds `+F.<chi_mu|r-r0|chi_nu>` to Hcore and
`-sum(q_A F.(R_A-r0))` to the classical energy. API gradients are **+dE/dR**;
mechanical forces have the opposite sign. Energies are Eh, gradients Eh/bohr,
fields Eh/(e bohr), ESP Eh/e and EFG Eh/(e bohr²). API atomic/point-charge
coordinates use Angstrom; origins and API property probes use bohr. For charge
Q, an origin shift d changes E by +QF.d; translating the complete charged
system at fixed origin changes E by -QF.d. EFG is QUICK's `dE_i/dC_j`, the
negative electrostatic-potential Hessian convention.

Electronic multipoles use **Psi4/ddX-style all-space regular-solid-harmonic
coupling**, including parent-partitioned volume nodes outside their sphere.
Stock PySCF's outside-sphere density/leakage correction is absent. Reference
comparisons must match projection, basis, cavity and volume grids, volume
partition and quadrature weights. A stock PySCF comparison is a distinct
model comparison unless these choices are explicitly matched.

| Continuum control | Value/meaning |
|---|---|
| Default quadratures | Lmax 6, cavity Lebedev 302, volume 75 radial×302 angular per atom; starting grids requiring system-specific convergence |
| Radius/dielectric/tolerance | Bondi scale 1.1, epsilon 78.3553, native solver tolerance 1e-10 |
| COSMO/PCM finite-dielectric scaling | COSMO `(epsilon-1)/epsilon`; PCM factor 1, with dielectric dependence already in its solve |
| Fixed native parameters | eta 0.1; COSMO shift −1 / PCM shift 0; switching 0; kappa 0; incore 0; maxiter 200; DIIS 20; nproc 1 |
| FMM | Enabled; pm=pl=max(8,Lmax) |
| Epsilon 1 | Explicit bypass reproduces gas-phase energy/gradient |
| Field+solvent convention | Requires `DDX_FIELD=SOLUTE`: perturb QM density/nuclei; no imposed macroscopic dielectric-field boundary or empty-cavity field energy |

The scope is all-electron finite-molecule electrostatic equilibrium solvation.
Unsupported radii, ghost atoms, ECPs and incompatible requests are rejected.
External-charge sources with ddX, SMD/cavitation/continuum solvent dispersion/
repulsion, nonequilibrium dielectric dynamics, PBC and constant-potential
electrodes are not implemented. Requested molecular D3 retains QUICK's
existing molecular energy/gradient contribution. The direct applied background
and continuum reaction potential/field are excluded from public OEPROP output:
the sampled quantities describe the current polarized solute density/nuclei.

## Build and interface example

Tested: GNU Fortran 14.4.0, CUDA 13.0.88, Open MPI 5.0.11, CMake 4.2.3, installed
ddX 1.0, NVIDIA RTX PRO 5000 72 GB Blackwell (SM120), one physical GPU. The CMake
CPU/MPI/CUDA/CUDA-MPI targets build with optional ddX ON or OFF. Untouched
legacy CPU OFF also built; legacy configure rejects CUDA 13, an inherited
toolchain limit. Manual C++11 compilation of the three new CUDA units and
CAS-branch emulation on modern hardware are not old-device/toolkit acceptance.

```sh
cmake -S . -B build-ddx -DCOMPILER=GNU -DENABLEF=FALSE \
  -DCUDA=TRUE -DMPI=TRUE -DQUICK_DDX=ON \
  -DddX_DIR=/path/to/ddx-install/lib/cmake/ddX
cmake --build build-ddx --parallel 4
```

Example keyword string:

```text
HF BASIS=STO-3G GRADIENT DDX=PCM DDX_EPS=78.3553 EXTERNAL_EFIELD_X=0.0002 DDX_FIELD=SOLUTE
```

The API retains its 256-character keyword limit. Pass equivalent keywords in
the BOMD namelist and use `oeprop_stride=1` for every accepted step. Internal
ddX updates also call ESP/EFIELD; raw property-kernel counters therefore include
continuum work and public samples. Counters must be reconciled with explicit
SCF-cycle counts; they do not imply extra property SCFs.

## Completed numerical and backend checks

Counts describe named campaigns, not independent statistical samples. All
fixtures, original acceptance tolerances and retained failure classifications
are pinned in the companion manifest.

| Campaign | Accepted coverage/results |
|---|---|
| Uniform field | CPU 1,031 runs/158 checks; CUDA 188/50; CPU MPI2 smoke 47/14 |
| ddX full CPU and CUDA | Each 858 runs/6,278 checks: He, water, mixed-SP/polarized basis, open-shell water cation and PBE0, both models, solute fields, epsilon 1, invariance and grid controls |
| Full CPU/CUDA parity | 858 matched runs/8,837 checks; max ΔE = 5.18e-9 Eh, max Δg = 1.76e-8 Eh/bohr |
| CPU MPI2 | 310 runs/2,262 checks; CPU parity 310 matched/3,213 checks, max ΔE = 1.99e-13 Eh, max Δg = 1.03e-14 Eh/bohr |
| CUDA MPI1 | 166 runs/1,246 checks; CPU parity 166 matched/1,773 checks, max ΔE = 2.49e-9 Eh, max Δg = 6.30e-9 Eh/bohr |
| WATER27 dimer | CPU 303/2,167; CUDA 159/1,155; parity 159 matched/1,631 checks, max ΔE = 4.03e-9 Eh, max Δg = 5.67e-9 Eh/bohr |
| Nuclear finite differences | Full CPU/CUDA 48 gates each; maximum errors 2.34e-7/2.45e-7 Eh/bohr; dimer CPU/GPU maxima 3.38e-7/9.29e-8 |
| Frozen-density variational checks | CPU/CUDA smoke each 8 runs/106 checks; full campaigns each 84 direction/step checks, maxima 1.90e-9/1.89e-9 |
| Higher-grid PCM | Water+dimer, L8/N590 vs L10/N974 at fixed volume 100×434: unchanged 1e-5 Eh/5e-5 Eh/bohr pairwise gates met; all 4 matched GPU points/59 checks passed |

Component tests separately compare normalized contracted s/p/d/f/mixed-SP
moments, derivatives, ownership and screening; full molecular GPU evidence
remains within the tested supported basis angular momenta. Standalone CUDA
field matrix/gradient errors are 3.47e-18/5.20e-18; Coulomb matrix/gradient errors
5.55e-16/5.41e-16. Complete discretized volume-gradient FD error is 8.67e-9.
The shared-library cache contract passes 22 matrix/lifecycle cases (max 6.66e-16).
Field/cache memcheck and initcheck report zero errors; cache memcheck reports
zero leaked bytes. These proofs do not establish whole-QUICK f-SCF support.

## Regression, trajectory and final-source lineage

| Evidence | Result and practical limit |
|---|---|
| Canonical CPU | 191 passes + 5 explicit skips; 450 original numerical comparisons |
| Canonical CUDA | 184 passes + 5 explicit skips; 436 original comparisons; single GPU |
| Existing OEPROP/BOMD suite | 897 checks/48 cases across CPU, MPI2, CUDA and CUDA-MPI1; ddX disabled, unchanged tolerances |
| New coupled CUDA MPI1 BOMD | 12 short trajectories/408 checks: stride/split/restart, finite-water solvent/field and hybrid basis controls |
| Final default-OFF bridge | All 4 variant families compile/link; 6 executable no-ddX ldd checks; CPU 3 + CUDA 3 smokes for ordinary energy, field gradient and explicit ddX rejection; 2,692 compiled inputs match final W |
| Final mathematical objects | All 42 validated operator objects byte-identical across reviewed main/API/install-layout/validation integration; whole-library identity is not claimed |
| Earlier private cleanup proof | Exact printed energy/gradient parity for 3 original/private pairs; 4 coupled sanitizer runs, zero errors/leaks; final installed CUDA and configured-prefix CPU proofs are added below |

The first canonical CUDA run stopped at case 87 when a NUL-containing
unsupported-f banner broke plain-grep skip recognition. Exact old4a5 negative
control reproduced it without an SCF. Three text-mode grep scanners fix that
runner defect; the complete unchanged test list then passes with explicit
skips. The original failed report is preserved. Its preceding PSB3 raw energy/
dipole comparisons passed; the wrapper's extra PSB3 failure attribution is not
evidence of a numerical PSB3 regression.

The accepted pre-release WATER20 transfer uses the exact WATER27 H2O20es
geometry (60 atoms), 20 fixed probes, PBE0-D3/DEF2-SVP, PCM and solute field.
At 0.025 fs, two steps give 3 converged forces, 3 property samples, 0 extra SCFs,
field counts [3, 3], 48 solvent updates, volume counts [48, 48, 3], cache 45 hits/3 builds/
48 GEMVs and no direct/fallback route. Drift is 1.87e-9 Eh. This is short finite-
cluster transfer evidence, not an ensemble or a publication trajectory.

## Controlled performance measurements

These are isolated production-grid reverse-Coulomb matrix measurements using
the same WATER20 geometry, 500-AO basis, frozen actual gas-phase density,
reaction charges and libraries. They are not self-consistent solvent
trajectories. Volume has 1,359,000 nodes; PCM/COSMO cavities have 6,749/8,019 nodes.

| Operator measurement | PCM | COSMO |
|---|---:|---:|
| First cache fill, seconds |5.3351|6.4072|
| Warm cached matrix, seconds |0.006068|0.007168|
| Direct matrix, seconds |5.3172|6.3821|
| Warm repeated-matrix speedup |876×|890×|
| Estimated 4-evaluation amortized factor |3.97×|3.97×|
| Maximum cached/direct matrix difference |5.29e-17|6.94e-17|

The 4-evaluation factor is an **estimate from measured cold, warm and direct
costs**, not an actually timed 4-direct sequence. Cache storage is 6.766/8.038 GB
including owned workspaces; the observed budget was 8 GiB. A new geometry pays
the cold build again. Gradients remain direct; one unchanged direct gradient
was timed per model. Forward published ESP is unchanged. There is no measured
whole-BOMD speedup claim. Synthetic 500-AO field microbenchmarks also exist,
including allocations/uploads/copies/frees; they are not molecular SCF timings.

## Preserved numerical limits

- COSMO dimer is not converged in harmonic order: shared N5810 and fixed
  volume 100×434, L10→14 gives |ΔE| = 4.5681e-5 Eh and max |Δg| = 1.1582e-4 Eh/bohr,
  failing original 1e-5/5e-5 gates. Separate fixed-L angular and L14 volume
  refinements pass pairwise gates; they do not waive this failure.
- The independent exact-basis denser-XC-grid field oracle retains **eight
  absolute PBE0 gradient failures**: two h, zero/field and CPU/CUDA. Incremental
  field-response checks pass. A ten-geometry native-SG1 diagnostic reduces
  the largest tested O-z discrepancy from about 2.69e-4 to 4.40e-8 Eh/bohr at
  h = 0.0005; maximum energy difference 3.13e-9 Eh. It explains that component's
  quadrature dependence only. Other components and native bin/primitive
  screening were not reproduced; the original failures remain false.
- Original coarse 0.1 fs NVE failures and ordinary-screening PBE0/STO-3G
  timestep-refinement/CPU-CUDA energy failures remain visible. Separate
  `XCCUTOFF=BASISCUTOFF=1e-12` CPU/CUDA controls pass equal 2 fs comparisons at
  0.1/0.05/0.025 fs. For these fixtures 0.05 fs or smaller is a tested starting
  point; timestep, SCF, XC and continuum convergence still require independent
  assessment. These short checks do not validate production ensembles/spectra.
- No HIP runtime, multiple physical GPUs, periodic AIMD, polarizable QM/MM,
  external-charge ddX, nonequilibrium solvent or constant-potential electrode
  acceptance is claimed. All published method behavior is scoped to the
  executed controls; external Amber RESP/rwRESP remains unexecuted.


## Installed CUDA lifecycle acceptance

| Final installed CUDA check | Result |
|---|---|
| CMake package consumer | Passed; resolves the intended final installed QUICK CUDA library |
| CLI noDDX/COSMO/PCM plus uniform field | Six memcheck/initcheck runs passed; zero errors; memcheck zero leaks; unchanged input fixtures |
| Installed cache contract | Normal/memcheck/initcheck each passed 22 matrix cases and lifecycle assertions; zero sanitizer errors/leaks |
| Active-cache point-source rejection | CUDA and CUDA-MPI1 rejected the unsupported request after positive solvent updates; no numerical operator/object override |
| All-seven-case CUDA BOMD | 28 short trajectories / 936 checks passed against the actual final installed R library; stride/split/restart controls |

The lifecycle aggregate's first postprocessing report falsely classified
expected global-data COPY relocations as procedure overrides. A separate
reviewed report uses code/weak-code/IFUNC symbol types and retains raw readelf
evidence. The initial cache report also compared raw ldd text containing ASLR
addresses; reviewed identity compares resolved dependency paths and SHA256
maps, which are unchanged. Both original false-flag reports, raw ldd and
original runner are preserved and pinned in the manifest. These are
artifact-only postprocessing corrections: no scientific rerun, changed
numerical/sanitizer/rejection gate or runtime mutation occurred.

The three initial CPU CLI runs loaded the old A-prefix library through their
embedded configured prefix/RPATH and remain excluded from final installed CPU
acceptance. The exact frozen CPU BOMD driver subsequently passed as an
imported-target consumer of the intended R-installed library. A separately
configured-prefix relink/install, followed by the new-prefix CLI/BOMD smokes
below, closes the packaging gap. The original excluded runs and preflight
rejections remain preserved; they were not retroactively accepted.

## Final release and configured-prefix closure

| Final check | Accepted result |
|---|---|
| CPU imported-target package consumer | Exact frozen driver, actual R-installed CPU library: 28 trajectories/936 checks across all seven cases |
| CPU/CUDA trajectory parity | 212 checks passed with original 1e-7 energy/geometry/velocity and 2e-7 property gates |
| Final MPI2 point-charge gradients | 130 evaluations/32 gates: neutral and open-shell cases, nuclear and external-charge finite differences, actual installed MPI library |
| Final CUDA-MPI1 coupled BOMD | 12 trajectories/408 checks; actual installed CUDA-MPI library; one rank/one physical GPU |
| Proper configured-prefix package | CPU 596/CUDA 1,303 objects, 173/344 modules, 387/727 generated compiled inputs and all 3,216 source files unchanged; 30 installed ELF checks resolve their new prefix with LD overrides unset |
| New-prefix CPU controls | Three ordinary/coupled-field CLI fixtures and eight short BOMD trajectories/two cases passed |
| New-prefix CUDA controls | Six CLI memcheck/initcheck controls and eight short BOMD trajectories/two cases passed |
| New-prefix CPU/CUDA parity | 62 checks passed; actual CUDA process live maps confirm the new-prefix library and absent user loader overrides |
| Final WATER20 release bridge | 53 candidate gates and 60 cross-release checks passed; two 0.025 fs PCM+solute-field PBE0-D3/DEF2-SVP steps, exact accepted geometry/probes and source/runtime/counters |
| Protected worktrees | All 11 protected worktrees unchanged against the original inventory |
| Final source/runtime/install identity | Nine checks passed; 3,216 source files and both old/new installation identities preserved through scoped release testing |

The configured-prefix correction changes only CMAKE_INSTALL_PREFIX and permits
verified CMake Ninja dependency metadata plus link actions. No source
compilation occurred; every existing object, module, generated compiled input
and scientific source byte matches before/after. The first two strict metadata
guards stopped before build/install and remain archived. Old A/install and
R/install prefixes were preserved. All 30 installed ELF files in the new CPU/
CUDA prefixes resolve their intended QUICK library without LD_LIBRARY_PATH or
LD_PRELOAD overrides. This is actual packaging/runtime evidence, not an
assumption that `cmake --install --prefix` changes compiled runtime search paths.

The independent ddX source remains clean at
`23de728143825695f54a04aeec2dd836b2ca9419`. The installed ddX library and exact
report/runtime identities are pinned in the companion manifest. The final
WATER20 comparison independently verifies raw fingerprints, geometry/probes,
energies, properties, accepted force/SCF counts and actual native GPU routes.
Its two-step duration remains a finite-cluster software-transfer check.
Three converged forces and three property samples over 20 fixed probes use zero
extra SCFs (cycles 17/14/14). Native field counts are [3,3], volume counts
[48,48,3], cache 45 hits/3 builds/48 GEMVs with zero direct/fallback calls. Final
short drift is 1.8740e-9 Eh; no performance conclusion follows from this run.

The configured CUDA sanitizer launcher had no user loader overrides. Compute
Sanitizer injects loader variables/libraries into its child process; those
injections are recorded separately from actual new-prefix QUICK path/hash
maps. A supplementary unset-child-variable assertion initially false-flagged
the first successful memcheck. Its raw run was preserved/reused, the remaining
five ran once, and no scientific/sanitizer gate changed. An earlier read-only
path check also held on literal `cuda//lib` spelling before any launch; resolved
path comparison closes it. Both original reports remain pinned and do not
represent numerical failures or additional successful scientific samples.

The protection/identity snapshot precedes documentation adoption and the
intentional canonical feature-branch update. The external publication receipt
records that authorized final containing-commit SHA and verified remote state;
it is separate from these pre-adoption preservation measurements.

## What acceptance establishes

The documented finite-molecule, all-electron applied-field and electrostatic
continuum implementation is engineering-ready within the tested CPU/MPI2/CUDA/
CUDA-MPI1 scope. Final installed paths, numerical invariants, analytic forces,
retained-density sampling, restart/stride behavior, cache/lifecycle handling,
optional-OFF builds and unchanged published-method controls have explicit
accepted evidence. All original false flags and genuine numerical failures
remain traceable in their own reports.

Publication and production convergence work is not complete. COSMO harmonic-
order convergence, denser-XC-grid absolute PBE0 comparisons and historical NVE
screening/timestep failures retain their stated limits. Scientifically useful
trajectories require suitable geometries/probes, converged numerical/model
settings and sufficient statistics. Whole-job scaling and property-sampling
overhead still require controlled measurements. The 876x/890x repeated-matrix
results and estimated four-evaluation amortization do not establish whole-BOMD
speedup. This report makes no push claim; the containing commit's final SHA
and verified remote state belong in the external publication receipt.
