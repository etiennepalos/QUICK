# AGENTS.md — QUICK Quantum Chemistry Package

## Overview

QUICK is a GPU-accelerated quantum chemistry package written primarily in Fortran 90/95,
with C/C++ and CUDA/HIP code for GPU support. The codebase uses two parallel build systems:
a legacy `configure`+`make` system and a modern CMake system.

### Current Local OEPROP Context

This working tree has active one-electron-property work in `src/modules/quick_oeproperties_module.f90`
and the OEPROP include paths `src/modules/include/attrashell.fh` and
`src/modules/include/nuclearattra.fh`.

Current external-grid property keywords:

| Keyword | Meaning |
|---|---|
| `ESP_GRID` | Electrostatic potential on user-supplied grid points |
| `EFIELD_GRID` | Electric field on user-supplied grid points |
| `EFG_GRID` | Analytic electric-field-gradient tensor on user-supplied grid points |
| `EFG_GRID_NUMERICAL` | Central finite-difference EFG reference from `EFIELD_GRID` |
| `ESP_SURFACE` | Electrostatic potential on an internally generated electron-density isosurface |
| `EFIELD_SURFACE` | Electric field on an internally generated electron-density isosurface |
| `EFG_SURFACE` | Analytic electric-field-gradient tensor on an internally generated electron-density isosurface |
| `DENSITY_SURFACE_VALUE` | Electron-density isovalue in a.u.; default `1.0d-3` |
| `DENSITY_SURFACE_SPACING` | Approximate surface-point spacing in Angstrom; default `0.50` |
| `DENSITY_SURFACE_MAX_RADIUS` | Maximum atom-centered ray length in Angstrom; default `10.0` |
| `EXTERNAL_EFIELD_X/Y/Z` | Uniform external electric-field components in a.u. |
| `FINITE_FIELD_X/Y/Z` | Alias for the uniform external electric-field components |
| `EXTERNAL_EFIELD_GRADIENT_XX/YY/ZZ/XY/XZ/YZ` | Six symmetric components G_ij=dE_i/dr_j of a static spatial field gradient |
| `EFIELD_GXX/GYY/GZZ/GXY/GXZ/GYZ` | Compact spatial-gradient aliases |
| `EFIELD_FX/FY/FZ`, `EFIELD_OX/OY/OZ` | Compact field-vector and fixed-origin aliases |
| `XC_RADIAL`, `XC_ANGULAR` | Paired opt-in unpruned XC quadrature; original SG0/SG1 defaults are preserved |

Implementation notes for this feature:

- Explicit XC refinement reuses SG1's Euler--Maclaurin radial map, existing
  positive-weight Lebedev rules, SSW partitioning and native CPU/GPU XC paths.
  Use both `XC_RADIAL` (1--400) and `XC_ANGULAR` (supported positive rule up
  to 5810). Negative-weight rules 74/230/266 are rejected because existing
  XC screening assumes positive weights. Reject malformed new values with
  error 57; keep published keyword readers unchanged. Refined grids also
  require checking the independent `MATRIXZERO` weight-screening threshold:
  `1e-14` was qualified on the documented small-water PBE0 cases, without
  changing its default. Broadcast each new unbinned point count before MPI
  workers traverse temporary arrays after a grid-size change. See
  `validation/external_field/XC_CONVERGENCE.md` and
  `validation/external_field/XC_GRID_LIFECYCLE.md` for the tested scope.

- `EFG_GRID` is the default analytic route. `EFG_GRID_NUMERICAL` is retained as a
  regression/reference path.
- `getQuickFieldHessian(npoints,points_bohr,H,ierr)` returns the full
  `H(3,3,3,npoints)=d2 E_i/dC_j dC_k=-d3 V/dC_i dC_j dC_k` through the
  independent `quick_field_hessian_module` CPU/native-CUDA evaluators.
  Preserve the published ESP/EFIELD/EFG entry points and arithmetic. The raw
  Hessian contraction uses the upper density triangle and higher AO index
  ownership; compute ten symmetric components and expand to all 27.
  This is an observation-point tensor, not a nuclear energy Hessian. It is
  source-only, uses retained spin-summed density, and has no trace projection.
  BOMD enables it separately with `field_hessian=.true.` (default false).
- The printed tensor follows the QUICK field-gradient convention
  `G_ij(C) = d E_i(C) / d C_j`, i.e. the negative of the electrostatic-potential
  Hessian convention sometimes denoted EFG in quantum-chemistry texts.
- CPU serial and MPI paths are implemented. MPI follows the existing shell-pair
  distribution and reduces `9*npoints` tensor components to the master rank.
- The analytic electronic contribution uses Obara-Saika style attraction-integral
  recurrences, matching the structure already used in `src/hessian.f90`.
- Current local optimization notes: EFIELD/EFG shell-pair setup uses direct
  Gaussian product centers and primitive attraction prefactors; EFIELD/EFG
  nuclear kernels use direct inverse-distance powers; numerical EFG reuses a
  single displaced-coordinate work array; analytic EFG contracts grid points in
  128-point auxiliary blocks.
- Density-surface properties are generated in `quick_molsurface_module.f90` by
  atom-centered radial rays, first-crossing bracketing, bisection to
  `rho(r)=DENSITY_SURFACE_VALUE`, and light duplicate pruning. The generated
  points are stored in `quick_molspec%vdwpointxyz` and then passed through the
  same OEPROP grid evaluators used by external grids.
- In MPI density-surface calculations, the master rank generates the point cloud
  and broadcasts `nvdwpoint` and `vdwpointxyz`; ESP/EFIELD/EFG then use the
  existing shell-pair distribution and reductions.
- After the current master integration, API MPI clients call
  `setQuickMPI(comm, ierr)` before creating a job; QUICK duplicates that
  communicator. Library collectives and shell indexing use `quick_comm`,
  `quick_comm_rank`, `quick_comm_size` and `quick_mpi_error`. Do not introduce
  `MPI_COMM_WORLD` into job-local property/surface operations. The explicit
  world-rank bookkeeping in MPI initialization is a separate operation.
- Use the boundary-aware `found_keyword` parser for feature keywords rather
  than substring matches. The CPU regression
  `validation/oeprop_api/validate_master_integration.py` checks 18 parser cases
  and two independent two-rank jobs within a four-rank MPI launch; see
  `docs/gpu-electrostatics-bomd/master-integration-20261014.md` for its command.
- Historical OEPROP reports and `docs/gpu-electrostatics-bomd/manifest.json`
  describe the earlier frozen feature checkpoint. Preserve those files and
  record current source/runtime identities and acceptance results separately.
  `ESP_CHARGE` is an unrestrained QUICK fit; preserving its ESP/MKS output
  does not establish execution of the external Amber RESP/rwRESP fitters.
- CPU analytic EFG uses a recursive evaluator. GPU s/p/d EFIELD and EFG use a
  normalized Hermite expansion shared by Cartesian pairs and components.
  EFIELD constructs the required R_(t,u,v)^0 values directly from polynomials
  through total degree I+J+1 (at most 5), using `gpu_oeprop_efield_direct.h`,
  generated by `tools/generate_oeprop_efield_direct.py`. Before forming Cartesian
  powers, EFIELD uses the original GPU recurrence if any nonzero Cartesian displacement
  component has magnitude outside [1e-50, 1e50] bohr; exact zero components stay direct.
  This protects raw Cartesian products, not arbitrary Boys/factor scaling.
  Native FP64 EFG directly constructs its 80 required degree-2-through-6
  m=0 Coulomb slots using `gpu_oeprop_efg_direct.h`, generated by
  `tools/generate_oeprop_efg_direct.py`. This route requires the exponent sum
  g in [1e-20, 1e20] bohr^-2 and each nonzero Cartesian displacement magnitude in
  [1e-20, 1e20] bohr. Exceptional ranges and every legacy fixed-point EFG
  calculation retain the original iterative GPU recurrence. These bounds
  select arithmetic without screening contributions or using a CPU fallback.
  The s/p/d derivative
  paths seed Boys functions at the CPU shell-maximum order
  Qfinal[II]+Qfinal[JJ]+2, equal to I+J+2 for pure shells. This also matches the
  CPU's order-dependent small-argument approximation for combined SP shells;
  the R degree still uses the actual angular components I and J. The FP64 EFIELD
  and EFG routes sum Cartesian AO pairs locally and use three and nine
  component atomics per primitive, respectively; EFG contracts six independent
  tensor components and writes both symmetric locations. `USE_LEGACY_ATOMICS` retains per-AO-pair
  updates and fixed-point quantization. Higher angular momentum retains the
  unchanged recursive GPU evaluator; f-function runtime validation remains
  outside current evidence. Shared helpers in `gpu_oeprop_hermite.h` reuse
  existing per-thread scratch and preserve ESP/RESP and CPU evaluators.
- Dense GPU derivative grids (at least 128 points) accumulate nuclear and
  external point-charge contributions in 128-point contiguous host blocks.
  The source order, inverse-distance arithmetic, all nine tensor expressions,
  and final addition to the electronic output are preserved. Smaller GPU
  grids and every CPU/MPI-only build retain the original reference loops;
  ESP/RESP routines are unchanged.
- Finite external fields are Hamiltonian perturbations, not OEPROP grid
  evaluators.  They add \(F_i\langle \chi_\mu | r_i-r_{0i}|\chi_\nu\rangle\)
  to Hcore and add the matching classical charge-field term to `Ecore`.  The
  field-polarized density can then be analyzed by ESP, EFIELD, and EFG.
- `EXTERNAL_EFIELD_ORIGIN_X/Y/Z` and `FINITE_FIELD_ORIGIN_X/Y/Z` set the
  finite-field origin in bohr.  The default is the Cartesian origin; charged
  systems are origin-dependent.
- Finite external fields include analytic contracted-moment nuclear derivatives
  and classical nucleus/point-charge forces. CUDA builds use the independent
  `gpu_external_field.cu` FP64 operator and derivative kernels; CPU remains the
  moment-integral reference. Retain MPI AO-row ownership and master-only
  classical terms before the shared gradient reduction.
- A nonzero `external_efield_tensor` selects independent CPU/CUDA quadratic
  moment paths for `h=F.s+s.G.s/2`, with matching classical energy and analytic
  forces. Preserve the old uniform-field arithmetic when G is exactly zero.
  Use six symmetric components; do not accept an antisymmetric electrostatic
  gradient or silently remove its trace. Source-free applied fields require
  trace(G)=0; general symmetric tensors represent prescribed charge density.
  The tensor is fixed in the laboratory frame, in Eh/(e bohr^2). Nuclear
  energy derivatives and this spatial field gradient are different quantities.
- Optional CMake `QUICK_DDX=ON` enables independent continuous-density ddCOSMO
  and ddPCM modules. Do not move the solvent operator ahead of the ordinary
  `oSave` assignment/MPI reduction or apply it to delta density. Keep solvent
  gradients before the shared gradient reduction and release geometry/GPU
  caches before GPU job deletion. Combined fields require the explicit
  `DDX_FIELD=SOLUTE` convention. See
  `docs/gpu-electrostatics-bomd/applied-fields-and-ddx.md` for scope and controls.
  `DDX_SOURCE=REGULAR` preserves the original all-space source projection and
  remains the default. Explicit `DDX_SOURCE=COULOMB` uses a decaying exterior
  continuation on each parent sphere, consistently in energy, Fock and force
  terms. It is a distinct finite functional, with no density clipping. Include
  the source mode in geometry/cache identity. The documented small HF water
  dimer refinement is molecule-specific; see
  `validation/solvation/continuum_convergence.md` and the separate source
  acceptance record. Do not transfer its qualification to REGULAR or claim
  full external-code model equivalence.
- Regression coverage lives in the ESP/OEPROP subset of `test/testlist_full.txt`.
  The numerical EFG saved `.out` baseline may be ignored by `.gitignore`; use
  `git add -f test/saved/efg_grid_numerical_acetone_b3lyp_def2svp.out` when this
  feature is intentionally staged.
- The finite-field saved `.out` baseline is also ignored by `.gitignore`; use
  `git add -f test/saved/ene_H2O_external_efield_rhf_sto3g.out` when staging the
  external-field regression test.
- Density-surface tests currently use separate regression inputs for the harness:
  `esp_grid_density_surface_H2O_rhf_sto3g.in`,
  `efield_density_surface_H2O_rhf_sto3g.in`, and
  `efg_density_surface_H2O_rhf_sto3g.in`. The saved `.out` baselines are ignored
  by `.gitignore`; use `git add -f test/saved/*density_surface_H2O_rhf_sto3g.out`
  if those outputs should be staged.

---

## Build Commands

### CMake Build (recommended)

```bash
mkdir build && cd build
cmake .. -DCOMPILER=GNU -DENABLEF=FALSE -DCMAKE_INSTALL_PREFIX=$PWD/../install
cmake --build . --parallel $(nproc)   # Linux: nproc; macOS: sysctl -n hw.logicalcpu
cmake --install .
source ../install/quick.rc   # sets QUICK_BASIS, PATH, LD_LIBRARY_PATH, LIBRARY_PATH
```

> **macOS + Homebrew GCC note:** On macOS, `/usr/bin/gcc` is an Apple Clang shim, so
> `-DCOMPILER=GNU` will be rejected by the build system's compiler identity check.
> Use `-DCOMPILER=AUTO` and pass the real compiler paths explicitly:
> ```bash
> cmake .. -DCOMPILER=AUTO -DENABLEF=FALSE \
>   -DCMAKE_INSTALL_PREFIX=$PWD/../install \
>   -DCMAKE_C_COMPILER=/opt/homebrew/bin/gcc-15 \
>   -DCMAKE_CXX_COMPILER=/opt/homebrew/bin/g++-15 \
>   -DCMAKE_Fortran_COMPILER=/opt/homebrew/bin/gfortran
> cmake --build . --parallel $(sysctl -n hw.logicalcpu)
> cmake --install .
> source ../install/quick.rc
> ```
> Adjust `gcc-15`/`g++-15` to match the version installed by Homebrew
> (`ls /opt/homebrew/bin/gcc-*`).

**Key CMake options:**

| Option | Values | Description |
|---|---|---|
| `-DCOMPILER` | `GNU`, `CLANG`, `INTELLLVM`, `ONEAPI`, `PGI`, `AUTO` | Compiler family |
| `-DENABLEF` | `TRUE`/`FALSE` (default: `FALSE`) | Enable f-function ERIs (expensive) |
| `-DCMAKE_BUILD_TYPE` | `Debug`, `Release` | Build type |
| `-DMPI` | `TRUE` | Enable MPI parallel build |
| `-DCUDA` | `TRUE` | Enable NVIDIA GPU build |
| `-DQUICK_USER_ARCH` | `volta`, `turing`, `ampere`, `adalovelace`, `hopper`, etc. | CUDA GPU architecture |
| `-DWARNINGS` | `TRUE` | Enable compiler warnings |

To build with AMD GPUs, add `-DHIP=TRUE` (plus HIP-specific flags like `-DHIP_WARP64=TRUE`). See `quick-cmake/QUICKCudaConfig.cmake` for details.

See `CMake-Options.md` for the full list.

### Legacy Configure+Make Build

```bash
./configure --serial --enablef --shared --prefix $PWD/install gnu
make -j$(nproc) all install
source install/quick.rc   # sets QUICK_BASIS, PATH, LD_LIBRARY_PATH, LIBRARY_PATH
```

MPI variant: replace `--serial` with `--mpi`.
CUDA variant: replace `--serial` with `--cuda --arch volta` (architectures: `kepler`, `maxwell`, `pascal`, `volta`, `turing`, `ampere`, `adalovelace`, `hopper`).
HIP variants use `--hip` (single GPU) or `--hipmpi` (MPI + HIP). Multi-GPU CUDA builds use `--cudampi`.

---

## Test Commands

Tests require QUICK to be **installed** first. The `runtest` script lives in the install
directory (or can be run from the repo root as `tools/runtest`). `QUICK_BASIS` must be set
(done automatically by `source install/quick.rc`).

### Run the full test suite

```bash
cd install
./runtest --serial --full        # 186 CPU tests
```

### Run the short test suite (CI default)

```bash
./runtest --serial               # short set (31 tests)
```

### Run a single test

QUICK has no built-in single-test flag. Run the executable directly and compare via `test/ndiff.awk`:

```bash
export QUICK_BASIS=/path/to/install/basis
/path/to/install/bin/quick test/ene_H2O_rhf_sto3g.in
awk -f test/ndiff.awk test/saved/ene_H2O_rhf_sto3g.out ene_H2O_rhf_sto3g.out
```

### Run tests by category

```bash
./runtest --serial --ene    # energy tests only
./runtest --serial --grad   # gradient tests only
./runtest --serial --opt    # geometry optimization tests only
./runtest --serial --api    # API tests only
./runtest --serial --esp    # ESP tests only
./runtest --serial --chk    # checkpoint/restart tests only
```

### MPI tests

```bash
DO_PARALLEL="mpirun -np 2" ./runtest --mpi --full
```

### Useful environment variables

```bash
QUICK_BASIS            # path to basis set directory (required)
DO_PARALLEL            # MPI launcher, e.g. "mpirun -np 2"
CUDA_VISIBLE_DEVICES   # GPU IDs for CUDA tests (comma-separated)
HIP_VISIBLE_DEVICES    # GPU IDs for HIP tests (comma-separated)
PARALLEL_TEST_COUNT    # number of tests to run in parallel (GNU parallel)
```

---

## Diagnosing Failing Tests

### Test harness tolerances

The `runtest` script compares output against saved references using `test/dacdif`, which
calls `test/ndiff.awk` with a hardcoded absolute-error threshold per check type. There is
no per-test override; the thresholds are set in `tools/runtest`:

| Check function | Absolute-error threshold | Quantity compared |
|---|---|---|
| `check_energy` | `4.0e-5` | Total / electronic / nuclear repulsion energies (Hartree) |
| `check_gradient` | `4.0e-3` | Analytical gradient components (Hartree/Bohr) |
| `check_opt` | `4.0e-3` | Optimized geometry and force elements |
| `check_dipole` | `4.0e-3` | Mulliken/Löwdin charges and dipole (Debye) |
| `check_esp_charge` | `1.0e-5` | ESP-fitted atomic charges |
| `check_vdw_surface` | `2.0e-7` | ESP values on vdW surface |
| `check_esp_grid` | `2.0e-7` | ESP values on external grid point file |
| `check_chk` (density restart) | `1.0e-8` | Total energy after density-matrix restart |
| `check_chk` (xyz restart) | `2.0e-4` | Cartesian coordinates after xyz restart |

### Physics cutoff keywords

The following keywords in `.in` files control integral screening thresholds and directly
affect numerical results across platforms and compiler configurations:

| Keyword | Internal default | Standard test value | Tightest value used |
|---|---|---|---|
| `cutoff=` | `1.0e-7` | `1.0e-9` | `1.0e-12` |
| `gradcutoff=` | `1.0e-7` | *(not set explicitly in any test)* | `1.0e-12` |
| `xccutoff=` | `1.0e-7` | *(not set in most tests)* | `1.0e-12` |
| `basiscutoff=` | `1.0e-6` | *(not set in most tests)* | `1.0e-12` |

`gradcutoff=` applies only to gradient and geometry optimization tests. `xccutoff=` and
`basiscutoff=` apply only to DFT tests. `denserms=` (SCF convergence criterion) is
intentionally excluded — it controls iteration count, not the numerical precision of the
converged result, so tightening it does not help platform-consistency failures.

### Diagnostic workflow for a borderline failure

If a test fails with a numeric deviation slightly above the threshold:

1. **Find the exact deviation.** In `runtest-verbose.log` (or the `test/runs/` directory),
   locate the `ndiff.awk` summary line for the failing test:
   ```
   ### Maximum absolute error in matching lines = X.XXe-YY at line N field M
   ```

2. **Make a scratch copy** of the failing input and tighten the physics cutoffs. Apply
   only the keywords relevant to the test type:
   ```
   # All test types:
   cutoff=1.0e-12

   # Gradient and geometry optimization tests — also add:
   gradcutoff=1.0e-12

   # DFT tests — also add:
   xccutoff=1.0e-12
   basiscutoff=1.0e-12
   ```

3. **Run the binary directly** on the scratch copy:
   ```bash
   export QUICK_BASIS=/path/to/install/basis
   /path/to/install/bin/quick <scratch_copy>.in
   ```

4. **Compare against the saved reference** using `dacdif` with the threshold matching the
   failing check type (see table above). For example, for an energy test:
   ```bash
   test/dacdif -k -a 4.0e-5 test/saved/<stem>.out <scratch_copy>.out
   ```

5. **Interpret the result:**
   - Failure disappears → the production `.in` file's cutoffs are too loose for this
     platform/configuration. Update the cutoffs in `test/<stem>.in` and regenerate the
     reference output in `test/saved/<stem>.out`.
   - Failure persists → the root cause is elsewhere (algorithm change, compiler
     floating-point behaviour, platform ABI). Investigate the logic change or consider
     regenerating the reference output after confirming correctness.

---

## Code Style Guidelines

### Languages

- **Fortran 90/95** — primary language (all SCF, DFT, gradients, MPI, API code)
- **C/C++** — utility subroutines, octree/grid packer, timing code
- **CUDA C/C++** — NVIDIA GPU kernels (`src/gpu/cuda/`)
- **HIP C/C++** — AMD GPU kernels (`src/gpu/hip/`)

### Fortran Naming Conventions

- **Modules:** `quick_<component>_module` — file named `quick_<component>_module.f90`
  - Examples: `quick_method_module`, `quick_basis_module`, `quick_exception_module`
- **Types:** `quick_<component>_type` — e.g., `quick_method_type`, `gpu_calculated_type`
- **Module instances (singletons):** lowercase — e.g., `quick_method`, `quick_molspec`
- **Subroutines/Functions:** No single style is enforced. `snake_case` is the majority
  convention across both old and new code (`raise_exception`, `form_dft_grid`,
  `allocate_quick_scf`). `camelCase` and `PascalCase` appear in legacy files and some
  newer modules (`getEnergy`, `PrtAct`, `gridformSG0`, `printQuickOutput`). When adding
  new code, prefer `snake_case`.
- **Local variables:** lowercase, short abbreviations — e.g., `natom`, `nbasis`, `ierr`
- **Type member variables:** mixed styles coexist — camelCase (`integralCutoff`,
  `analGrad`), ALL_CAPS (`HF`, `DFT`, `MP2`), lowercase (`opt`, `grad`), and snake_case
  (`read_coord`, `esp_charge`). No single rule applies.
- **Constants/Parameters:** `ALL_CAPS` — e.g., `PI`, `BOHR`, `OUTFILEHANDLE`
- **Preprocessor macros:** `ALL_CAPS` — e.g., `RECORD_TIME`, `MPIV`, `ENABLEF`, `DEBUG`

### C/C++/CUDA Naming Conventions

- **Structs/Types:** predominantly `snake_case` with `_type` suffix — e.g.,
  `gpu_timer_type`, `gpu_simulation_type`. Some structs omit the suffix (`gpu_scratch`)
  or use mixed-case prefixes (`XC_quadrature_type`, `ERI_entry`).
- **Functions:** `snake_case` or `camelCase` — both styles appear freely with no clear
  rule (`get_oshell_eri`, `upload_sim_to_constant` vs. `getOEI`, `getGrad`).
- **Macros:** `ALL_CAPS` — e.g., `LOC2`, `LOC3`, `SQR`, `VDIM1`
- **Fortran-callable C functions:** use `extern "C"` with trailing underscore —
  e.g., `gpu_set_device_`, `gpu_upload_method_`

### File Naming

- Fortran modules: `quick_<component>_module.f90`
- Fortran subroutines: predominantly `snake_case.f90`; some legacy files use
  `camelCase.f90` or `PascalCase.f90` (`getEnergy.f90`, `CPHF.f90`)
- CUDA/C++ source: `snake_case.cu`, `snake_case.cpp`, `snake_case.h`
- GPU headers: `gpu_<purpose>.h` or `gpu_<component>_<type>.h`

### Indentation and Formatting

- **Fortran:** 3-space indentation (dominant convention); some legacy files use 2 or 4 spaces
- **C/CUDA:** 4-space indentation; opening `{` on same line as control structure
- **CMake:** 4-space (1-tab) indentation
- New Fortran code should use `implicit none` in every scope. Many legacy subroutines
  use `implicit double precision(a-h,o-z)` instead — do not add new code with implicit
  typing.
- Long lines are allowed (compiler flag `-ffree-line-length-none` is set for GNU Fortran)
- The `tools/amindent` utility can be used to normalize Fortran indentation

### Include/Use Ordering

**Fortran `use` statements** — place at the top of the program unit, before `implicit none`:
1. Standard/external library modules (e.g., `use mpi`)
2. QUICK utility modules (e.g., `use allMod`, `use quick_constants_module`)
3. Functional QUICK modules (e.g., `use quick_method_module, only: quick_method`)

**Preprocessor include:** Every Fortran source file must include the project-wide utility
header near the very top (before the `module`/`subroutine` statement):
```fortran
#include "util.fh"
```
This header (`src/util/util.fh`) provides timing macros (`RECORD_TIME`, `START_TIME`,
`STOP_TIME`) and the `OUTFILEHANDLE` constant.

**C/CUDA headers:** Standard library headers first, then local project headers:
```c
#include <stdio.h>
#include <string>
#include "gpu.h"
#include "../gpu_common.h"
```

### Error Handling

- **Fortran:** Use an integer error flag `ierr` declared `intent(inout)` and passed through
  the call chain. Non-zero values indicate errors. Check and raise via:
  ```fortran
  call RaiseException(ierr)   ! from quick_exception_module
  ```
  This calls `quick_exit(OUTFILEHANDLE, 1)` on failure. Error codes are defined in
  `src/modules/quick_exception_module.f90` with `select case(ierr)` messages.
- **MPI code:** Wrap MPI-only logic in `#ifdef MPIV` / `#endif` preprocessor guards.
  Only the master process should do I/O:
  ```fortran
  if (master) write(OUTFILEHANDLE, ...) ...
  ```
- **CUDA:** Check CUDA API return codes explicitly within GPU utility functions.

### Comment Style

**Fortran:**
```fortran
! Single-line comment

!-----------------------------------------------------------------------!
! Section header banner (width ~72 chars)                               !
!_______________________________________________________________________!

subroutine foo(x, y)  ! inline comment after code
```

**C/CUDA:**
```c
// Single-line comment
/* Block comment for longer explanations */
```

Every source file should carry a copyright/license header (MPLv2):
```fortran
!
! (C) Copyright <year> QUICK contributors
! All rights reserved.
! ...
! This Source Code Form is subject to the terms of the Mozilla Public
! License, v. 2.0.
!
```

### MPI/GPU Portability

- CPU-only, MPI, CUDA, and HIP builds share the same source tree. Use preprocessor guards:
  - `#ifdef MPIV` for MPI-specific code
  - `#ifdef CUDA_MPIV` for CUDA+MPI code
- GPU-callable device functions are annotated with `__device__` / `__global__` (CUDA) or
  the HIP equivalents.
- Fortran-to-GPU interoperability is done via `iso_c_binding` and `extern "C"` interfaces.

---

## Repository Layout (Key Paths)

```
src/modules/         Fortran modules (quick_*_module.f90) — all shared state/types
src/subs/            Fortran utility subroutines and C++ utility files
src/gpu/cuda/        NVIDIA CUDA kernels and GPU driver (gpu.cu)
src/gpu/hip/         AMD HIP kernels
src/octree/          C++ octree for DFT quadrature grids
src/libxc/           DFT XC functional library (libxc)
src/util/util.fh     Project-wide Fortran preprocessor header (MUST include)
quick-cmake/         Amber build-system glue and GPU helpers
test/                Regression test inputs (*.in), GPU lists, references (saved/*.out)
test/testlist_full.txt      Full CPU test list (~186 tests)
test/testlist_short.txt     Short CPU test list (31 tests)
test/testlist_full_gpu.txt  Full GPU test list (~187 tests)
test/testlist_short_gpu.txt Short GPU test list
tools/runtest        Test runner script
basis/               Basis set data files
.github/workflows/   CI definitions (serial + MPI)
unit-tests/          Unit/prototype harnesses
CMake-Options.md     Reference for all CMake build options
```

---

## CI

GitHub Actions workflows are in `.github/workflows/`:
- `build_test_serial.yml` — Serial builds (legacy + CMake) on Ubuntu 22.04/24.04 (x86/ARM) and macOS 14/15 across GNU 10–15, Clang 17/18/20, IntelLLVM 2024/2025, and NVHPC 25.1.
- `build_test_mpi.yml` — MPI builds (legacy + CMake) on the same OS spread using OpenMPI, MPICH, and Intel-MPI toolchains.

Each workflow installs HDF5 when needed, builds QUICK, runs `./runtest --serial --full` or `./runtest --mpi --full`, and archives `runtest.log`, `runtest-verbose.log`, and `test/runs/*` outputs.
