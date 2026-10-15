# OEPROP BOMD Driver

`test-api-oeprop-bomd` integrates molecular Born–Oppenheimer dynamics with
velocity Verlet through the public QUICK energy/gradient API. Serial CPU,
CPU MPI, CUDA, and CUDA MPI builds use the same driver. Each initial or
accepted step makes one energy/gradient API evaluation. Sampled steps then
evaluate ESP, EFIELD, and EFG from that live density, without another SCF.

Run the checked-in water example from the repository root:

```bash
export QUICK_BASIS=$PWD/basis
/path/to/test-api-oeprop-bomd validation/oeprop_bomd/water_fixed.nml
```

The namelist is `quick_bomd`. Important fields are:

| Field | Meaning |
|---|---|
| `geometry_file` | Initial single-frame XYZ; omitted for restart runs |
| `keywd` | QUICK method and basis; `GRADIENT` is added if absent |
| `nsteps`, `dt_fs` | Additional accepted steps and time step in fs |
| `oeprop_stride` | Sample absolute steps divisible by this integer; default 1; 0 disables probe generation and properties |
| `field_hessian` | Also sample analytic H_ijk=d2 E_i/dC_j dC_k; default `.false.`; uses the same stride |
| `temperature_k` | Deterministic initial kinetic-energy scaling; no thermostat or thermal ensemble sampling |
| `nfrozen`, `frozen_atoms` | Optional 1-based frozen atom indices |
| `checkpoint_file` | Text checkpoint written after each accepted step |
| `restart_file` | Existing checkpoint; preserves absolute step, time step, and reference NVE energy |
| `coordinate_atoms` | Two atoms defining the reported distance coordinate |

Stride applies to the initial and restart frame as well as newly accepted
steps. Energy and coordinate output remains present at every step. V2
checkpoints store the normalized method/basis keyword and reject a restart
with a different keyword. Legacy V1 files remain readable with an explicit
unverified-Hamiltonian warning. A restart performs one initial SCF/gradient
at the checkpoint geometry; the density and SCF iteration history are not
serialized.

Probe modes:

| `probe_mode` | Probe-file records | Motion |
|---|---|---|
| `fixed` | `x y z` | Fixed laboratory coordinates |
| `molecule` | `u v w` | Body-frame coordinates based on `anchor_atoms` |
| `atom` | `atom x y z` | Fixed Cartesian offset from a selected atom |
| `density` | none | Regenerated electron-density isosurface |

The first probe-file line is its record count. Checked-in fixtures use the
`.probe` extension. Coordinates are Angstrom; the OEPROP API receives bohr.
Moving probes change their position, while output EFIELD and EFG components
remain in laboratory Cartesian axes. Density-surface probe IDs are local to
each sampled frame and have no temporal correspondence. Probes at nuclear
singularities are rejected; this grid API is not a self-excluded nuclear EFG
implementation for quadrupolar NMR.

The property CSV reports QUICK atomic units with

```text
E_i(C) = -dV(C)/dC_i
G_ij(C) = dE_i(C)/dC_j.
```

`G` is the negative potential Hessian. Its trace can be nonzero where electron
density is present. Outputs are `<prefix>_energy.csv`, `<prefix>_trajectory.xyz`,
`<prefix>_oeprop.csv`, `<prefix>_summary.json`, `<prefix>_kernels.csv`, and the
checkpoint. The property CSV includes all tensor components, trace, Frobenius
norm, sorted eigenvalues, anisotropy, and a selected atom-pair distance.

With `field_hessian=.true.`, sampled frames also write
`<prefix>_field_hessian.csv`. This separate file contains the ten independent
entries `(xxx,xxy,xxz,xyy,xyz,xzz,yyy,yyz,yzz,zzz)` of
`H_ijk=d2 E_i/dC_j dC_k=-d3 V/dC_i dC_j dC_k`, in hartree/(e bohr^3).
This spatial tensor is distinct from the nuclear energy Hessian. Components
stay in laboratory Cartesian axes for every probe mode. The public
`getQuickFieldHessian(npoints,points_bohr,H,ierr)` API returns the full
`H(3,3,3,npoints)` array. It uses the retained density without another SCF;
the summary records sample and primary native CUDA evaluation launch counts
when enabled. The separate symmetry-expansion kernel is excluded from that
dispatch counter.
The default `.false.` preserves the existing property CSV layout and avoids
field-Hessian work. These properties include the QM density, nuclei and
enabled external charges; direct applied and ddX reaction fields are omitted.

The per-rank kernel CSV records API energy/gradient evaluation counts, property/probe calls,
actual ESP/EFIELD/EFG kernel launch deltas, and accumulated phase wall times
in seconds. CPU launch counts are zero. Timed force calls include the API's
internal SCF/gradient output and exclude driver trajectory/property CSV/XYZ
output; probe time includes generation and singularity checks; OEPROP time
covers its synchronous API call. The validator retains every rank and reports
the maximum rank's accumulated time for each phase, rather than treating
master-rank timing as MPI elapsed time.

Run the targeted validator with NumPy and the installed drivers:

```bash
python3 validation/oeprop_bomd/validate_oeprop_bomd.py \
  --bomd-driver /path/to/test-api-oeprop-bomd \
  --oeprop-driver /path/to/test-api-oeprop \
  --gpu-driver /path/to/test-api-oeprop-bomd.cuda \
  --mpi-driver /path/to/test-api-oeprop-bomd.MPI --launcher 'mpirun -np 2' \
  --gpu-mpi-driver /path/to/test-api-oeprop-bomd.cuda.MPI \
  --gpu-launcher 'mpirun -np 1' \
  --workdir /tmp/quick-oeprop-bomd --run-pbe0
```

GPU and MPI arguments are optional. Select GPU MPI ranks to match available
physical GPUs; a one-rank run validates that executable but provides no
multi-GPU scaling evidence. `--oeprop-driver` is the CPU API postprocessor.
For every requested backend, the suite covers water and a water trimer,
optional PBE0-D3/def2-SVPD, all four probe modes, stride 1/2/0, split restarts
including an unsampled restart boundary, and per-rank property kernel launch counts.
It rejects nonfinite values, missing/duplicate/reordered step or probe rows,
incorrect timestamps and counter totals. Probe-coordinate transforms are
recomputed independently. EFG eigenvalues are checked with NumPy's symmetric
eigensolver, alongside independent trace, norm, anisotropy, and symmetry checks.

Two different comparisons are reported explicitly: separate short trajectories
from identical inputs, and online properties versus independently converged
CPU SCFs at each backend's own saved geometries. The latter fixes geometry,
not density. An identical-density property-backend test belongs to the
[OEPROP acceptance suite](../oeprop_acceptance/README.md). The JSON summary
records commands, tolerances, fixture/source hashes, executables, and resolved
QUICK shared-library hashes, and checks that source and loaded libraries did
not change during validation.

Check GPU device initialization as well as numerical agreement. Run both an
s/p basis example and a d-basis namelist produced by `--run-pbe0`:

```bash
compute-sanitizer --tool initcheck --error-exitcode 99 \
  /path/to/test-api-oeprop-bomd.cuda validation/oeprop_bomd/water_fixed.nml
```

The GPU first-derivative assemblers omit unused angular blocks above their
initialized Boys order; all integrals consumed by the gradient are preserved.

The original CPU thresholds remain 1e-5 Eh for short NVE drift, 1e-9 for
properties, and 1e-10 for restart coordinate/velocity comparisons. Separate
GPU defaults are 2e-7 for property components, 1e-7 Angstrom for short
trajectory coordinates, and 1e-7 Eh for energies; command-line overrides are
recorded. These few-step tests establish software and lifecycle behavior.
They do not establish long-time energy conservation, a thermal ensemble,
converged fluctuation statistics, correlation times, or spectra. Deterministic
initial velocities and the reported kinetic temperature are diagnostic; the
mobile-atom temperature degree count is not a general constrained-ensemble
estimator when atoms are frozen.

## Paired whole-job property overhead

`benchmark_overhead.py` compares fresh property-disabled (`oeprop_stride=0`)
and property-enabled processes from the same geometry, method, deterministic
50 K velocity initialization, time step, step count and normal output/checkpoint
policy. It discards one complete pair, retains at least three more pairs and
alternates their execution order. Every raw run, including the discarded pair,
remains in its own directory. Use an empty work directory for each configuration:

```bash
python3 validation/oeprop_bomd/benchmark_overhead.py \
  --driver /path/to/test-api-oeprop-bomd.cuda \
  --workdir /tmp/quick-bomd-overhead-water-256 \
  --case water-hf --probe-count 256 --stride 1 \
  --nsteps 100 --dt-fs 0.05 --repeats 3 --threads 1
```

Cases include HF/STO-3G and PBE/PBE0-D3/def2-SVPD water and water-trimer
fixtures with the existing tight screening keywords. Probe generation retains
the first four checked-in fixed sites; larger counts add nested deterministic sites on
a sphere beyond the initial molecular extent. These distant points are a grid
cost control rather than a density-surface benchmark. `--launcher 'mpirun -np 1'`
and `--ranks 1` select an MPI driver; the recorded rank count must agree.
`--prepare-only` writes the inputs/provenance plan without running the driver;
use a separate empty directory for the actual benchmark. NumPy is required.

The whole-job clock starts immediately before process/launcher creation and
ends after it exits. It includes startup, setup, all SCF/gradient evaluations,
ESP/EFIELD/EFG work, normal API/driver output, checkpoints and teardown. Python
input generation, hashing and output validation are excluded. Ordinary buffered
filesystem behavior is retained; no cache flush or extra durable-write sync is
introduced. `overhead_summary.json` stores every paired wall-time delta and
percentage, plus retained median/minimum/maximum/sample standard deviation.
The maximum-rank accumulated property-API/force-API ratio is reported separately
and excludes driver output; it must not be called whole-job overhead.

Acceptance requires `nsteps+1` energy/gradient calls in each run, the exact
sampled-frame/probe totals, zero extra calls, finite outputs and the expected
per-rank property kernel-launch counts. Each pair compares all printed Cartesian
gradients, trajectory coordinates, energies and final checkpoint velocities;
fixed absolute gates are 2e-7 Eh/bohr, 1e-7 Angstrom, 1e-7 Eh and 1e-7 a.u.,
respectively. Printed gradients have 1e-10 Eh/bohr resolution. Repeated disabled
trajectories and enabled properties are also checked. Binary, every resolved
`ldd` library, scientific source, basis and input SHA256 hashes must remain
unchanged before/after every run. Hardware, rank/thread settings, output sizes,
raw hashes and timing boundaries are recorded. These tests measure software
overhead for the specified fixtures and do not establish long-time dynamics.

For physical correlation functions use a fixed sampling interval, a specified
laboratory/body frame or bond projection, equilibrated trajectories, and
statistical uncertainty estimates. Event-selected “significant” frames need
an explicit irregular-sampling treatment. Regenerated density-surface row IDs
cannot directly label time autocorrelations. A field autocorrelation alone
does not supply an IR spectrum or a calibrated vibrational frequency map.
Long trajectories and large molecular droplets require separate scientific
and performance validation; periodic bulk/slab simulations are outside this
finite-molecule driver.

### Property output blocks

The trajectory writer formats the step/time and probe-mode/reaction-coordinate
fragments once per sampled frame, then writes at most 128 probe rows per formatted
statement. A fixed 23,552-byte numeric buffer bounds memory use even when a density
surface changes size. The CSV column order, I0 and ES24.16 formatting, one record
per probe, tensor ordering, invariant expressions and checkpoint policy are
unchanged. Only the master rank writes. Exact-byte fixtures and actual CPU runs
cover partial blocks and changing frame sizes; GPU comparisons use the existing
numerical gates because independent atomics can change the last printed digits.

On the local RTX PRO 5000, three fresh paired processes per cell measured a median
2.38% lower whole-job wall time for HF water with 4,096 probes at every step and
0.73% when sampling every fifth step (100 steps at 0.05 fs). The 20-step
PBE-D3(0)/def2-SVPD water-trimer comparison at 256 probes showed no measurable
whole-job benefit. These short execution tests are not production sampling
benchmarks. Output-only timing and instrumented host attribution are separate
from the uninstrumented whole-job clocks. Raw data, reproduction scripts and
independent audits are saved outside the repository in
`quick_ep/host-output-opt-20261011/`.
