# One-pass frozen-density collector

`quick_electrostatics_snapshot.f90` uses the existing benchmark input and
`properties.dat`, `ao_basis.dat`, `density.dat` and `overlap.dat` formats. It
does one initial energy/SCF API call, one V/E/G API call and one H API call.
There are no warmups, repeats, chunk tests or displaced observations. It is
an application-data collector, not a timing-repetition benchmark.

```sh
/matching/snapshot-client /inputs/input.dat -
```

Run in a new directory with `QUICK_BASIS` and the loader paths matching its
install. The optional second argument replaces the retained density after
the initial SCF, exactly as in the benchmark client; `-` retains the newly
converged density. Input nuclear coordinates are Angstrom, probes are bohr.
The output contains full 3x3 G and 3x3x3 H tensors in Fortran component order,
with source convention and signs inherited unchanged from the native APIs:
electrons plus nuclei and enabled MM charges, `E=-grad(V)`, `G=grad(E)` and
`H=grad(G)`. Direct applied and reaction backgrounds are excluded; the
retained density can respond to them.

D/AO/S files are exported before expensive property evaluation. A successful
`result.dat` requires finite V/E/G/H and exact retained-density/Fock and API
step invariance. `timings.csv` contains two rows, VEG and H; GPU builds require
one native launch for each V/E/G/H route and CPU builds require zero GPU
launches. V/E/G are separate native evaluations under one API call, not a
fused kernel. `setup_scf_export_seconds` includes setup, the SCF, state copying
and D/AO/S export. Each route time covers its synchronous API call. The
setup-to-teardown clock additionally includes remaining normal file IO.

Build with the matching install's compiler and modules; CUDA adds `-DCUDA
-DGPU` and links `-lquick_cuda` instead of `-lquick`:

```sh
/matching/gfortran -cpp -ffree-line-length-none -DCUDA -DGPU \
  -I/path/to/QUICK/src/util \
  -I/matching/install/include/amber-modules/quick/cuda \
  validation/electrostatics_benchmark/quick_electrostatics_snapshot.f90 \
  -L/matching/install/lib -Wl,-rpath,/matching/install/lib \
  -lquick_cuda -o /new/snapshot-client
```

Add actual optional dependency loader/rpath-link directories where required.
Archive the exact compiled client source and compiler command. The production
source snapshot, loaded libraries, basis, executable and input hashes should
be recorded before/after actual runs. Verify native printed method/grid/
screening settings and exactly one converged SCF before interpreting data.
The collector alone does not validate convergence or generate a trajectory.

`collect_snapshot.py` provides the portable execution/provenance wrapper and
then runs the existing independent AO and offline analysis tools:

```sh
python3 validation/electrostatics_benchmark/collect_snapshot.py \
  --driver /matching/snapshot-client --backend cuda \
  --install-prefix /matching/install \
  --source-snapshot /immutable/production/source \
  --build-report /retained/build-report.json \
  --client-source /archived/exact-compiled-client.f90 \
  --compile-record /retained/client.compile.json \
  --prepared /prepared/application-plan --case-id water20-def2svp \
  --oracle-python /python/with/pyscf \
  --expected-basis-functions 500 \
  --library-path /matching/dependencies/lib \
  --workdir /new/snapshot-evidence --plot --prepare-only
```

The compile record contains `argv`, `returncode`, `source_sha256` and
`binary_sha256`, naming the exact archived client source in its arguments.
Preparation verifies runtime dependencies with `ldd` and launches no native
job. For actual collection, omit `--prepare-only` and use another new output
directory. CPU mode is available for explicitly selected small cases; it is
not needed for the planned 500-AO GPU snapshot. The run mapping is generated
automatically for the one requested backend, so the wrapper launches no
reference CPU SCF. The independent PySCF worker evaluates Cartesian AO
values/derivatives from the exact exported density without an SCF.

`report.json` retains native commands/timing, complete provenance before/after,
one-converged-SCF evidence, one-pass counters, native method/settings checks,
full-tensor symmetry discrepancies and independent Poisson residuals. The
existing Cartesian overlap gate is `2e-10`, electron-count gate `1e-8`, and
the G/H Poisson absolute residual gates `2e-8`. Finite Taylor sample errors
and first-two-radius observed orders are retained rather than promoted to a
continuous sphere bound. `--plot` produces an aggregate PDF and one PDF per
center; CSV/JSON retain each radius and direction.

```sh
python3 validation/electrostatics_benchmark/test_collect_snapshot.py \
  --output /new/snapshot-io-tests
```

These 29 synthetic export-schema checks cover accepted CPU/CUDA formats,
repeated calls/SCFs, state mutation, launch/phase counts, finite values,
probe ordering/positions, full tensor permutations and density/overlap/AO
dimensions. They launch no native process, SCF or GPU kernel and establish
I/O behavior only. The initial v6/plan-v4 prepare-only audit also launched
zero native jobs; actual scientific acceptance remains a separate report.

## Retained WATER20 application result

The one-pass v6 client with corrected application-plan-v4 subsequently passed
the full `water20-one-pass-snapshot-v1/report.json` campaign on 2026-10-15.
It used the initial WATER27 H2O20 geometry, isolated PBE0-D3 with Cartesian
def2-SVP (500 AOs), the explicit 150x590 XC grid and the exact screening/SCF
card retained in the report. The SCF converged in 15 cycles at its unchanged
1e-10 density criterion. Twenty centers, six radii 0.01/0.02/0.05/0.1/0.2/0.4
bohr and 32 directions gave 3860 total direct queries. This was one static
frozen density, with no CPU WATER20 reference SCF or MD extension.

One native launch for each V/E/G/H route, zero extra SCFs and exactly zero
density/Fock mutation passed. The independent Cartesian AO overlap error was
1.87e-15, electron-count error 8.53e-14, and maximum G/H Poisson residuals
5.97e-15/1.25e-14. All source/runtime/basis/input/client/harness hashes remained
unchanged. Observed first-two-radius orders across the 20 centers were
0.990–1.015, 1.983–2.013 and 2.990–3.009 for constant, linear and quadratic
field reconstructions. At 0.2 bohr the worst sampled errors were
0.01079/0.002735/0.0004953 field atomic units. Twenty individual PDFs and one
aggregate PDF accompany full CSV/JSON. These are finite directional samples,
not continuous sphere bounds, bulk/PBC results or ensemble statistics.

The NVIDIA RTX PRO 5000 72GB run used GPU0, CPU core31 and one OpenMP/BLAS
thread, with about 18GB device allocation and 4.6GB host RSS observed. A root
CPU build ran concurrently; single-shot phase times (setup/SCF/export about
1000.67s, V/E/G 0.297s, H 3.950s) are execution diagnostics, not performance
estimates. The ten-minute review allowed this same converging job to continue
without relaxing any control, with an explicit 1800s native deadline. The
prepared wrapper's original internal default was 7200s; an archived external
PID/start-time/executable-checked monitor enforced the explicit 1800s limit
without restarting the SCF. Native completion occurred after about 1005.6s,
and the monitor exited without sending a signal. GPU0 was released before
the independent offline analysis. The decision, timeout correction and monitor
status are retained separately under `workflow-benchmark-preparation`.

The initial 2026-10-15 preparation compiled this new client against the v6
CUDA install, with no native execution during preparation. Its immutable
source, binary and compile record are retained as
`clients/snapshot-one-pass-cuda-v6.*`; `water20-snapshot-command-v2.json`
points to the corrected application-plan-v4 card with `D3` and explicit
`BASISCUTOFF=1e-12`. The earlier benchmark-style collector and older plans
remain historical artifacts.
