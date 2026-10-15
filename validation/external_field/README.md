# Applied-field validation

## Controlled spatial gradients

`quick_spatial_gradient_api_driver.f90` and `validate_spatial_gradient.py`
validate the static affine field `E(r)=F+G*(r-r0)`. Build the client with the
same Fortran compiler and installed module files as the library under test;
link `libquick` for CPU or `libquick_cuda` for CUDA. Optional ddX builds also
need their installed ddX shared-library directory in the linker/runtime
search paths. Do not mix modules or runtime libraries from different builds.

```sh
python3 validation/external_field/validate_spatial_gradient.py \
    --cpu-driver /path/to/cpu/spatial-gradient-api \
    --cpu-basis /path/to/cpu/install/basis \
    --cuda-driver /path/to/cuda/spatial-gradient-api \
    --cuda-basis /path/to/cuda/install/basis \
    --pyscf-python /path/to/pyscf-environment/bin/python \
    --workdir /path/to/new/spatial-gradient-validation
```

The independent oracle reconstructs the exact installed Cartesian Gaussian
basis and maps AO order and normalization explicitly. It compares the full
affine operator, all six unit tensor operators, fixed-density contractions,
SCF energies, and analytic nuclear forces against independent integrals and
SCF-energy differences. Additional checks cover two displacement sizes,
charged-system origin covariance, simultaneous coordinate/origin translations,
signed external charges, pure gradients, nonzero trace, and Cartesian f shells.
Nuclear force fixtures must explicitly request `GRADIENT`; the API entry-point
name alone does not activate gradients. The client rejects an inactive
analytic-gradient route. Input cards are checked against the API's existing
256-character limit and use the documented compact aliases.

`validate_spatial_gradient_cuda_kernel.py` directly tests the installed native
CUDA entry points using NumPy Gaussian quadrature, all Cartesian s/p/d/f
monomials on two centers, two-step force differences, AO-row masks, screening,
invalid requests and transactional error handling. It can be run under NVIDIA
Compute Sanitizer; pass `--tool memcheck --error-exitcode 86` to the sanitizer
and use a separate fresh work directory for each run.

`validate_spatial_gradient_coupled.py` checks PCM/COSMO and PBE0 compatibility,
PCM force differences, and short sampled/unsampled BOMD controls including
Hamiltonian-compatible restart. It requires NumPy, the API clients above,
and matching BOMD clients. Its force-history client is an instrumented copy
of the existing driver that only exports already-computed gradients; retain
the instrumentation diff with the evidence. Short trajectories establish
software and force consistency, not production sampling or convergence.

Every campaign writes to a fresh directory. Preserve failed attempts and
record exact source, client, runtime and reference identities with the final
acceptance report. Existing uniform-field and published property regressions
remain independent preservation controls.

## Uniform fields

`validate_external_field.py` runs one CLI backend per invocation. The smoke
suite evaluates water HF/STO-3G in a field with three nonzero components;
the full suite adds HF/6-31G** mixed-SP and d functions, UHF water cation,
PBE0/STO-3G, and a one-center helium case. Full-suite nuclear energy
differences cover both mixed-field signs and all six signed Cartesian fields.
Each run requires an explicitly converged SCF and finite results.

```sh
python3 -B validation/external_field/validate_external_field.py \
    --quick /path/to/cpu-build/bin/quick \
    --basis "$PWD/basis" --workdir /path/to/new/field-cpu-smoke \
    --suite smoke

python3 -B validation/external_field/validate_external_field.py \
    --quick /path/to/cuda-build/bin/quick.cuda \
    --basis "$PWD/basis" --workdir /path/to/new/field-cuda-full \
    --suite full
```

Use `--launcher 'mpirun -np 2'` for CPU MPI and a separate work directory.
CUDA MPI requires an appropriate device allocation; a one-rank MPI launch
establishes one-rank coverage only. `--cases` selects named models and
`--steps` sets the displacement scan. Jobs run sequentially, with separate
inputs and execution logs. The default water smoke suite requires 47 SCFs;
the five-model full suite requires approximately 1,026 SCFs. Neither count
is a timing or performance prediction.

Checks include analytic nuclear gradients versus central SCF-energy
differences, field-origin energy covariance, translation covariance,
`sum(gradient) = -Q F`, and `dE/dF = -mu`. QUICK prints dipoles about the
center of mass; the harness converts charged dipoles to the field origin
using QUICK's atomic masses. Coordinates are displaced using QUICK's input
Bohr conversion. The generated keyword cards fit its 300-character buffer;
each subprocess receives a short input basename for its 80-character
filename buffer.

The new force checks use absolute thresholds of `1e-5` hartree/bohr for HF
and `5e-5` for PBE0. Energy invariance uses `2e-8` hartree. The dipole check
allows `3e-5` atomic units plus the measured printed-energy rounding bound;
the legacy printed dipole has four decimal places in Debye. The report
records every finite-difference step and energy-rounding bound. These are
new validation thresholds; published regression fixtures and tolerances
are unchanged.

The same executable compares omitted and explicitly zero fields using
`1e-9` hartree and `1e-8` hartree/bohr envelopes. Optional
`--reference-quick /path/to/previous/bin/quick` and
`--reference-launcher '...'` also check the previous executable at zero
field with those strict envelopes. They accommodate GPU accumulation
roundoff while remaining much tighter than the standard regression suite.

The independent oracle runs in a separately selected Python environment
containing NumPy and PySCF:

```sh
python3 -B validation/external_field/validate_pyscf_external_field.py \
    --quick-reports /path/to/field-cpu-smoke/report.json \
                    /path/to/field-cuda-full/report.json \
    --python /path/to/pyscf-environment/bin/python \
    --workdir /path/to/new/field-pyscf-oracle
```

It parses the actual QUICK inputs, reconstructs Cartesian-basis HF/UHF or
PBE0 field Hamiltonians, and recomputes energies, full-precision dipoles,
AO densities and finite-difference gradients. It deduplicates equal inputs
across backend reports. Coordinates are supplied in Bohr to eliminate a
cross-code Angstrom conversion mismatch. HF comparisons use `1e-6`
hartree and `1e-5` hartree/bohr; PBE0 comparisons allow `5e-5` for different
DFT quadratures. Its own field-energy derivatives must match the
variational dipole within `2e-6` atomic units. Source inputs, runtimes,
reports and worker request/results are fingerprinted.

For external point charges, compile the small API client against the same
QUICK library and Fortran module directory used by the selected backend:

```sh
gfortran -cpp -ffree-line-length-none -I src/util \
    -I /path/to/cpu-build/amber-modules/quick/serial \
    validation/external_field/quick_external_field_api_driver.f90 \
    -L /path/to/cpu-build/src -Wl,-rpath,/path/to/cpu-build/src \
    -lquick -o /path/to/field-api-cpu

python3 -B validation/external_field/validate_point_charge_field.py \
    --driver /path/to/field-api-cpu --workdir /path/to/new/field-point-cpu
```

MPI clients use `mpifort -DMPIV`, the `quick/mpi` module directory and
`-lquick_mpi`; CUDA uses `quick/cuda` and `-lquick_cuda`; CUDA MPI uses
`quick/mpi_cuda` and `-lquick_mpi_cuda`. The client must match the compiler,
MPI implementation and backend of the linked QUICK library.

This separate harness places signed charges `+0.35` and `-0.20` near water
and differentiates the full-precision API energy with respect to all 15
QM/charge coordinates at two displacement sizes. It also checks the
combined net force, origin and translation covariance, and omitted versus
zero fields. It uses a three-component field and an x-directed field-origin
shift; the separate CLI suite covers arbitrary three-component origins.
API cards fit the existing 256-character interface and oversized cards are
explicitly rejected by the client. `--case` selects neutral HF/STO-3G, HF/6-31G**, charged UHF,
or PBE0. Select `--gradient-tolerance 5e-5` for PBE0 to match the DFT envelope
above. QUICK's returned QM energy excludes the MM charge-charge self-energy;
the tested gradients follow that same energy convention.

These suites cover finite molecules in static uniform fields. They do
not establish long-trajectory sampling, a constant-potential electrode,
continuum solvent, or a GPU speedup. Field
gradient assembly currently runs on the host even in GPU calculations;
the existing SCF and other gradient GPU paths remain active. OEPROP keeps
its source-only default: add `F` to obtain the total electric field, subtract
`F dot (r-origin)` for the total potential, and add zero to the EFG.
