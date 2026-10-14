# OEPROP acceptance and property-only benchmarks

This standard-library Python harness uses `test-api-oeprop` from the same source
revision as QUICK. Default acceptance and benchmarks initialize one SCF per
process; the optional repeated-gradient lifecycle check initializes two. Serial CPU
writes its converged AO density (alpha and beta separately for UHF); MPI/GPU jobs
load that density after their one initialization SCF. This isolates property
kernel parity from small differences between SCF implementations. The energy in
an API CSV is the initialization SCF energy, not a recomputed energy of the
imported density. Density snapshots are validation fixtures, not a public restart
format.

Cases are water RHF/STO-3G, acetone PBE0-D3BJ/cc-pVDZ, NH4+ RHF/STO-3G, and OH
UHF/STO-3G (doublet). Acceptance evaluates 16 deterministic off-nuclear probes and
checks all three properties, changed point counts/order, repeated single-output
calls, calls after larger generated surfaces, genuine file dispatch, and genuine
density-surface dispatch. CPU and accelerator jobs evaluate identical AO densities
and point sequences. Property parity uses a maximum absolute tolerance of 2e-7 a.u.; initializer SCF
energy parity is recorded separately at the existing 4e-5 hartree regression
tolerance. Nonfinite values
and mismatched value counts fail.

The scientific checks hold density fixed and use central differences at h=1e-4
and 5e-5 bohr: E_i=-dV/dC_i and G_ij=dE_i/dC_j. Analytic EFG is also checked against
QUICK's numerical EFG implementation and tensor symmetry. Generated-surface EFG also satisfies
trace(G)=-4*pi*rho_e with rho_e=1e-3; the tensor is not assumed traceless in
regions with electron density. CSV tensor order is
xx,xy,xz,yx,yy,yz,zx,zy,zz; the API stores G(3,3,npoints) in Fortran order.

```bash
source /path/to/toolchain/activate.sh
export QUICK_BASIS=/path/to/QUICK/basis
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
python3 validation/oeprop_acceptance/validate.py \
  --cpu-driver /scratch/cpu/src/test-api-oeprop \
  --mpi-driver /scratch/cpu/src/test-api-oeprop.MPI \
  --gpu-driver /scratch/cuda/src/test-api-oeprop.cuda \
  --workdir /scratch/oeprop-acceptance --benchmark --repeats 3
```

Omit unavailable backends; the harness records exactly the backends supplied.
Use `--gradient --cases water oh` to initialize through the analytic-gradient
API with one SCF. Use `--lifecycle --cases water oh` to repeat the gradient SCF at
identical geometry after property/surface calls and verify energies, gradients,
and properties remain consistent. Every mode checks that property calls preserve
the SCF counter and that `getQuickOEPROP` returns error 47 after `deleteQuickJob`.
`results.lifecycle.csv` records the observed lifecycle checks. Lifecycle mode
performs exactly two gradient SCFs; the warmed benchmark mode always performs one
initialization SCF.
`--cases water` runs a tiny acceptance case. HIP binaries can be supplied through
the same GPU arguments. Supply `--gpu-mpi-driver` only for a supported multi-GPU configuration.
`--mpi-launch` accepts a launcher string; the default is
`mpirun -np 2`. GPU builds record successful ESP/EFIELD/EFG kernel-launch deltas
for every rank in `results.kernels.csv`; zero counts fail accelerator acceptance.
The optional `--gpu-evidence` adds a profiler's kernel-summary text as corroborating
evidence. Neither a GPU banner nor a speedup alone establishes kernel execution.

With `--benchmark`, water and acetone use 64, 256, 1024, 4096, 16384, and 65536
points by default. `--benchmark-max-points` caps this list at a supported size;
for example, `--benchmark-max-points 4096` retains 64, 256, 1024, and 4096 points.
The generated timing grid and expected trials use this same cap, and
`acceptance.json` records the selected sizes. Each size/property gets an
unrecorded warm call and three recorded trials; use `--repeats 5` for five trials.
ESP, EFIELD, EFG, and a combined API call are timed separately with a high-resolution
wall clock. MPI reports the maximum elapsed rank time. Timing includes the
synchronous API's density/grid uploads, kernel, nuclear contribution, transfers,
and reductions. It excludes SCF, setup, snapshot I/O, generated surfaces, and CSV
writing. The driver asserts that property calls do not advance the SCF call
counter. Raw trials are retained; median/min/max are reported, with no inferred
acceleration claim. Pointwise numerical acceptance covers the 16 external probes
and every generated surface point. The sums recorded on larger timing grids are
execution sanity values; they do not establish pointwise accuracy on those grids. `--benchmark-cases water` limits benchmark cost.
Each timing sum is compared with the CPU sum and the first trial of its own
backend, divided by npoints times 1/3/9/13 components for ESP/EFIELD/EFG/all,
respectively. These aggregate comparisons use 2e-7 a.u. and can hide cancellation;
they complement the pointwise acceptance checks. Missing trials, nonfinite
results, and a timing SCF count other than one fail.

Outputs stay in `--workdir`: `acceptance.json` includes numerical errors, source
revision, dirty-source status/diff hash, validation-source and driver hashes,
method/basis keywords, CMake cache/options and the configured compiler version,
hardware/tool versions, thread/device environment, conventions,
and exact process commands. Executable hashes and the actual QUICK shared-library
paths/hashes resolved by `ldd` are recorded before and after the complete run;
changed fingerprints or unsuccessful QUICK shared-library resolution fail the
provenance check. The adjacent-library inventory is retained separately.
`timings.csv` contains raw benchmark trials. Per-case
subdirectories retain properties, density fixtures, logs, and kernel counts. Do
not commit generated artifacts or calculation scratch.

The electronic kernels contract screened Gaussian attraction integrals with the
AO density, using a factor of two for off-diagonal AO pairs and alpha+beta density
for UHF. ESP uses the attraction-integral recurrence; EFIELD and EFG use analytic
first and second derivatives with respect to the observation point and Boys
auxiliary functions. GPU derivatives evaluate primitive pairs and accumulate in
double precision on the tested CUDA configuration. GPU s/p/d EFIELD and EFG use a
normalized Hermite expansion shared across Cartesian basis pairs and the three
field components or six independent Hessian components. EFIELD constructs only
the required R_(t,u,v)^0 values with direct Gaussian-derivative polynomials through
total degree I+J+1 (at most 5). The helper in `gpu_oeprop_efield_direct.h` is generated
by `tools/generate_oeprop_efield_direct.py`. Before Cartesian powers are formed,
EFIELD uses the original GPU recurrence when any nonzero Cartesian displacement
component has magnitude outside [1e-50, 1e50] bohr; exact zeros stay direct.
This protects raw Cartesian products rather than arbitrary Boys/factor scaling.
Native FP64 EFG constructs only its 80 required degree-2-through-6 m=0 slots,
through actual angular order I+J+2, using `gpu_oeprop_efg_direct.h` generated by
`tools/generate_oeprop_efg_direct.py`. It requires the exponent sum g in
[1e-20, 1e20] bohr^-2 and every nonzero Cartesian displacement magnitude in
[1e-20, 1e20] bohr. Exceptional ranges and all legacy fixed-point EFG calls
retain the original iterative GPU recurrence. These bounds select GPU arithmetic
without screening contributions or using a CPU fallback. First derivatives with respect
to C contract minus the shifted R values; second derivatives contract plus the
twice-shifted values, since PC=P-C. The s/p/d derivative paths seed Boys functions
at Qfinal[II]+Qfinal[JJ]+2, matching the CPU shell maxima. This equals I+J+2 for
pure shells and matches the CPU's order-dependent small-argument approximation
for combined SP shells. R values still use the actual angular components I and
J, through degree I+J+1 for EFIELD and order I+J+2 for EFG.

The R values and coefficients reuse each thread's existing interleaved
store/store2 scratch; each primitive writes every required value before use.
The FP64 EFIELD and EFG routes sum Cartesian AO-pair contributions locally and
perform three and nine component atomics per primitive, respectively. EFG sums
six independent tensor components before writing all nine tensor locations.
`USE_LEGACY_ATOMICS` retains the per-AO-pair
updates and fixed-point quantization. Direct R evaluation and local FP64 summation
change floating-point arithmetic association. Higher angular momentum retains
the unchanged recursive evaluator. EFG fills both symmetric off-diagonal
positions explicitly. ESP/RESP and CPU evaluators are unchanged by these GPU optimizations.
The numerical EFG route remains a central-difference reference.

The caller owns the API coordinate and output arrays. Each GPU property upload
replaces the owned device point/output buffers and starts electronic accumulation
at zero, so changed point counts and single-output calls do not reuse an old
accumulator. Current host density is uploaded for each property evaluation.
Molecular basis/geometry state survives energy and gradient API calls, is rebuilt
for a new SCF, and is released by job deletion. A property call can lazily upload
converged host state after optimization without another SCF. Checked launches and
synchronous result downloads report device errors before the host returns results.
Serial and MPI GPU initialization check an 8192-byte minimum thread stack for
recursive electrostatic derivatives, preserve any larger limit, and verify the
runtime readback. The s/p/d CUDA optimizer/new-job sequence was checked with
full-leak memcheck; HIP and instrumented f-function sufficiency remain untested.

`EXTERNAL_EFIELD` polarizes the SCF density. OEPROP reports sources from that
density, nuclei, and external point charges. The imposed uniform background is
excluded from these source properties. For a total potential and field, with
coordinates and origin in bohr, use V_total(C)=V_OEPROP(C)-F dot (C-origin) and
E_total(C)=E_OEPROP(C)+F. Its direct EFG contribution is zero. Applied-field
analytic nuclear gradients are currently rejected as unsupported.

CPU MPI distributes electronic shell pairs and reduces their contributions to
the master rank. Classical nuclear and external point-charge terms are then
added once on the host. The public API broadcasts each completed requested array
to all ranks. Density-surface coordinates are generated on the master and
broadcast before this same property path. Calls are collective in an MPI build.

The local acceptance evidence covers serial CPU, two-rank CPU MPI, and one NVIDIA
GPU. A CUDA MPI executable was built, but GPU MPI2 acceptance is unavailable with
one physical GPU; the supported launcher requires a distinct device for each
local GPU rank. CUDA MPI with one rank passed property, repeated-gradient,
standalone CLI, and optimizer/new-job checks. HIP was not built or tested. The builds use `ENABLEF=FALSE`, so
f-function support is outside this evidence; the cc-pVDZ case exercises s/p/d
functions. These are limits of this validation run.

For a tiny CUDA memory check, use the completed CPU water fixture and a fresh
working directory:

```bash
compute-sanitizer --tool memcheck --leak-check full \
  /scratch/cuda/src/test-api-oeprop.cuda acceptance \
  /scratch/oeprop-acceptance/water/molecule.xyz \
  /scratch/oeprop-acceptance/water/probes.dat results \
  'HF BASIS=STO-3G CHARGE=0 CUTOFF=1.0D-12 DENSERMS=4.0D-8 ENERGY' \
  /scratch/oeprop-acceptance/water/cpu/results.density.dat
```

## Extreme-coordinate regression

The standard-library-only `validate_gpu_range.py` evaluates 129 signed probes at
1e62, 1e70 and 1e80 Angstrom using the existing acceptance CLI. The GPU imports
the exact CPU-exported density; both backends must return finite ESP, EFIELD and
EFG values with one initialization SCF and no additional property SCFs. The
regression checks accelerator launch evidence and parses the omitted E in
Fortran output with three-digit exponents.

Use the acetone geometry created by the normal acceptance run and a fresh directory:

```bash
python3 -B validation/oeprop_acceptance/validate_gpu_range.py \
  --cpu-driver /scratch/cpu/src/test-api-oeprop \
  --gpu-driver /scratch/cuda/src/test-api-oeprop.cuda \
  --repo-root /path/to/QUICK \
  --geometry /scratch/oeprop-acceptance/acetone/molecule.xyz \
  --workdir /scratch/oeprop-finite-range
```

Optional `--cpu-source-root` and `--gpu-source-root` record source hashes;
`--prepare-only` prepares inputs without claiming validation. Resolved runtime
libraries, input/basis hashes, lifecycle counters and full output remain in the
work directory. This stress regression reproduces nonfinite EFIELD values in the
old direct route. Passing establishes finite behavior at these cancellation and
underflow dominated distances; it does not establish relative far-field accuracy
or performance. Normal-coordinate accuracy remains covered by `validate.py`.
