# Continuous-density QUICK/ddX validation

`quick_solvation_api_driver.f90` links against a matching QUICK library and
Fortran module directory. It accepts `energy`, `gradient` or `frozen` and a
short input basename. API cards are limited to 256 characters; the client
reads a 300-character raw card and explicitly rejects an oversized API card.
The input contains a keyword card, atom count and rows
of atomic number and Cartesian coordinates in Angstrom. The driver prints
full-precision energy, gradients, nominal and quadrature electron counts,
solvent energy, solvent/GPU counters and the API step.

Example compilation for a serial CPU build with the same compiler used by
QUICK:

```sh
gfortran -cpp -ffree-line-length-none -I src/util \
    -I /path/to/build/cpu/amber-modules/quick/serial \
    -I /path/to/ddx-install/include/ddX \
    validation/solvation/quick_solvation_api_driver.f90 \
    -L /path/to/build/cpu/src -Wl,-rpath,/path/to/build/cpu/src \
    -lquick -L /path/to/ddx-install/lib -lddx \
    -Wl,-rpath,/path/to/ddx-install/lib -o /path/to/solvation-api-cpu
```

MPI uses `mpifort -DMPIV`, `quick/mpi` modules and `-lquick_mpi`. CUDA uses
`quick/cuda` and `-lquick_cuda`; CUDA MPI uses `quick/mpi_cuda` and
`-lquick_mpi_cuda`. The library needs `QUICK_DDX=ON` for active solvent jobs.

Start with a small frozen-density engineering campaign:

```sh
python3 -B validation/solvation/validate_solvation.py \
    --driver /path/to/solvation-api-cpu \
    --workdir /path/to/new/solvation-cpu-frozen \
    --suite smoke --cases helium water --models COSMO PCM \
    --checks energy frozen
```

This performs eight SCFs. Each frozen client then evaluates solvent energy
at two normalized symmetric AO-density perturbations and three displacement
sizes, without another SCF. Unrestricted calculations perturb alpha and beta
densities independently. The energy derivatives must match `Tr(V deltaP)`
within `1e-6` hartree per normalized density perturbation. The client also
checks that restoring density restores solvent energy/potential, leaves
the QUICK Fock and total/electronic energy state unchanged, and preserves
the API step. All MPI ranks apply the same perturbation and participate in
the collective source-property calculation.

For nuclear force checks:

```sh
python3 -B validation/solvation/validate_solvation.py \
    --driver /path/to/solvation-api-cpu \
    --workdir /path/to/new/solvation-cpu-forces \
    --suite smoke --cases water --models COSMO PCM \
    --checks energy frozen forces invariance eps1
```

All nuclear coordinates are displaced by `1e-3` and `5e-4` bohr. HF force
tolerance is `1e-5` hartree/bohr; PBE0 uses `5e-5`. Translation invariance,
zero net force and dielectric-one identity with the gas-phase calculation
are checked. Dielectric-one comparison uses strict `1e-9` hartree and
`1e-8` hartree/bohr roundoff envelopes. GPU runs require positive density,
matrix and requested gradient counters. Every process must contain one
explicitly converged SCF; the API step starts at one and returns two.

The smoke grid is explicitly an engineering choice: `DDX_LMAX=4`,
`DDX_NLEBEDEV=110`, `DDX_NRAD=50`, `DDX_NANG=110`. A pass does not establish
default-grid or quadrature-converged accuracy. `--suite full` selects
`6/302/75/302`, adds mixed-SP/d-basis HF, charged unrestricted water and
PBE0, and includes volume-grid convergence at `50x110`, `75x302` and
`100x434`, with the cavity grid held fixed. Default-to-dense differences
must be below `1e-5` hartree and `5e-5` hartree/bohr. All three volume
electron counts and energies/forces are retained; monotone convergence is
not assumed. The default electron-count envelope is `1e-3` electrons and
is recorded explicitly. Cavity harmonic/surface convergence should also be
examined by repeating with `--lmax 8 --nlebedev 590` before final accuracy
claims.

Use `--launcher 'mpirun -np 2'` with a matching MPI client and separate
evidence directory. CUDA MPI requires appropriate device allocation;
one-rank coverage is not multiple-GPU coverage. `--applied-field` exercises
the explicit `DDX_FIELD=SOLUTE` convention, its net-force identity and, when
`forces` is requested, full nuclear finite differences at both step sizes. It
means the field acts on the QM solute while the dielectric responds to
the induced solute density; it does not impose a macroscopic field boundary
condition on the dielectric.

All thresholds are new continuum acceptance checks. Existing regression
fixtures and tolerances are untouched. Reports retain command lines, raw
outputs, source/runtime fingerprints and individual failures. These tests
cover electrostatic ddCOSMO/ddPCM, not SMD, non-electrostatic solvation,
constant-potential electrodes, long trajectories or performance speedups.

The separate NumPy-based trajectory campaign exercises applied fields,
ddCOSMO, ddPCM, their defined combinations and PBE0:

```sh
python3 -B validation/solvation/validate_solvation_bomd.py \
    --driver /path/to/test-api-oeprop-bomd.cuda \
    --workdir /path/to/new/solvation-cuda-bomd
```

It compares property strides 0/1/2, split restarts, explicit converged SCF
sections and actual per-rank CUDA launches. Equal-duration trajectories at
0.05/0.025 fs (40/80 steps) test NVE time-step refinement over the same 2 fs.
The short property-stride/restart trajectories retain their 0.05 fs timestep.
Set `--energy-dt` and `--energy-steps` to choose the coarse refinement timestep
and step count; the refined trajectory uses half the timestep and twice the
steps. The default 2 fs refinement is a short numerical check; it does not
establish a production ensemble. Active
solvent jobs include internal forward ESP/EFIELD calls in their raw counts.
Successful volume CUDA calls and master solver/quadrature phase times are
reported separately. `--launcher` and `--ranks` select an MPI client. Run the
established ddX-disabled OEPROP/BOMD suite separately without changing its
expected counts or tolerances.


The NVE acceptance gates remain `1e-5` hartree maximum absolute energy drift
for each trajectory and refined drift no greater than `0.4` times the coarse
drift plus `5e-8` hartree. For an explicit default-protocol invocation:

```sh
python3 -B validation/solvation/validate_solvation_bomd.py \
    --driver /path/to/test-api-oeprop-bomd.cuda \
    --workdir /path/to/new/solvation-cuda-bomd-production \
    --energy-dt 0.05 --energy-steps 40
```

This protocol corrects the earlier 0.1/0.05 fs choice, which exceeded the
unchanged `1e-5` absolute-drift gate for six finite-water fixtures despite
proper second-order timestep refinement. Preserve those failed campaign
reports and raw outputs; a new production-protocol pass does not relabel
them. A default change is a timestep selection, not a relaxation of a gate.

Both PBE0 cases explicitly use `XCCUTOFF=1D-12 BASISCUTOFF=1D-12`, as do the
static DFT force tests. Ordinary QUICK loose-screening defaults remain
available to ordinary calculations. They are diagnostic settings for these
short refinement checks: the historical PBE0/STO-3G trajectory showed an
isolated XC/quadrature energy discontinuity with the ordinary screening,
which spoiled the strict maximum-drift refinement test. The matched tighter
screening control removed that anomaly. Retain the ordinary-screening failed
report as a diagnostic; do not treat it as a reliable refinement protocol or
silently exclude its offending frame. These validation settings do not
change published DFT algorithms, default user calculations, or tolerances.
