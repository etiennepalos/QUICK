# Paired BOMD overhead including the field Hessian

`benchmark_workflow.py` adds a separate benchmark for V/E/G/H sampling. It
reuses the existing overhead benchmark's trajectory/checkpoint readers,
full-gradient parsing, physical comparison gates and runtime fingerprinting.
The previous benchmark, BOMD driver and property defaults remain unchanged.

Each pair starts two fresh native processes from identical geometry, method
card, deterministic 50 K initial velocities, time step and normal output
settings. OFF uses `oeprop_stride=0, field_hessian=.false.`. ON uses the
requested positive stride and `field_hessian=.true.`. Pair order alternates
OFF/ON then ON/OFF. The defaults are three retained pairs after one discarded
pair, two MD steps of 0.025 fs and sampling every step: three force evaluations
and three V/E/G/H samples per enabled job, including step zero.

The primary clock covers process/launcher startup, QUICK setup, every SCF
and gradient, probes/properties, driver/API output, checkpoints and teardown.
Python input preparation, provenance hashing and validation are outside this
clock. Processes are fresh but filesystem caches are not flushed. Discarded
pairs can warm caches or the device state; they do not retain CUDA contexts.
There is no added fsync. The synchronous V/E/G/H API / force API ratio is
reported separately; it excludes driver property-file IO and is not whole-job
overhead. Native V/E/G routes remain separate calls, not a fused kernel.

## Portable invocation

Supply an accepted BOMD executable and its actual install, immutable production
source snapshot, one static XYZ geometry, a count-prefixed probe file, and a
complete one-line QUICK card. Probe units default to Angstrom; `--probe-units
bohr` explicitly converts observation sites to the driver's Angstrom input.
The keyword card requires GRADIENT and BASIS. Use the public `D3` keyword;
`DFTD3` is an internal variable name ignored by QUICK and is rejected here.

For the selected isolated PBE0-D3/Cartesian def2-SVP protocol, a card is:

```text
PBE0 D3 BASIS=DEF2-SVP GRADIENT CHARGE=0 MULT=1 SCF=400 DENSERMS=1e-10 CUTOFF=1e-12 GRADCUTOFF=1e-12 BASISCUTOFF=1e-12 MATRIXZERO=1e-14 XCCUTOFF=1e-12 XC_RADIAL=150 XC_ANGULAR=590
```

Other explicitly identified cards can be supplied. No continuum model is
enabled by default. Geometry, fields and all numerical controls must match
within every pair. Direct imposed and explicit reaction backgrounds are
excluded from source properties; a retained density can respond to them.

```sh
python3 validation/electrostatics_benchmark/benchmark_workflow.py \
  --driver /accepted/bomd-cuda --backend cuda \
  --install-prefix /matching/quick/install \
  --source-snapshot /immutable/compiled/source \
  --build-report /retained/build-report.json \
  --geometry validation/oeprop_bomd/water.xyz \
  --probes validation/oeprop_bomd/water_fixed.probe \
  --keyword-file /inputs/pbe0-d3-def2svp.card \
  --protocol-json /inputs/protocol-controls.json \
  --expected-basis-functions 25 \
  --library-path /matching/dependencies/lib \
  --nsteps 2 --dt-fs 0.025 --stride 1 \
  --repeats 3 --warmup-pairs 1 --threads 1 \
  --workdir /new/workflow-evidence --prepare-only
```

Preparation records inputs, the alternating order, source/runtime identities
and the exact namelists. It launches no native job. Run the command without
`--prepare-only` in a new evidence directory when the execution window is
available. Every trial is retained; existing output directories are refused.
CPU uses `--backend cpu` and its matching driver/install. An optional MPI
launcher and rank count can be supplied, but completed acceptance below is
serial CPU and a single native CUDA device.

The case catalog and its actual execution status are:

| Geometry | Probes | Cartesian def2-SVP AOs | Execution status |
|---|---|---:|---|
| `validation/oeprop_bomd/water.xyz` | bundled `water_fixed.probe` | 25 | paired CPU/CUDA workflows completed |
| `validation/oeprop_bomd/water_trimer.xyz` | bundled `water_fixed.probe` | 75 | plans prepared; workflow unexecuted |
| `validation/solvation/fixtures/water27_h2o20es.xyz` | matching bundled `.probe` | 500 | separate one-snapshot analysis completed; no whole-workflow benchmark |

Default 3+1 pairs cost eight fresh processes, hence 24 force SCFs for a
two-step case. If cost requires no discarded pair, set `--warmup-pairs 0` and
retain at least three pairs when affordable. Fewer retained pairs are labeled
software smoke tests; they do not support a performance conclusion. These
0.05 fs paths check integration and overhead, not an equilibrium ensemble.

## Numerical and lifecycle checks

Every native job must report exactly `nsteps+1` converged force SCFs and the
same number of energy/gradient API calls, with zero extra property API/SCF
calls. ON V/E/G and H rows must have the same exact sampled step/probe
identities, timestamps and coordinates. The ten H columns have the standard
order `xxx,xxy,xxz,xyy,xyz,xzz,yyy,yyz,yzz,zzz`; native master H launches must
equal the sample count on CUDA and zero on CPU. OFF must perform no probe
or property work and produce no H output. Requested D3 and refined XC-grid
settings and requested screening thresholds are checked against the native
method output. `--protocol-json` supplies an exact expected token/value map;
it also rejects an omitted requested control before execution. An optional AO
dimension gate guards Cartesian basis expectations.

Paired runs and deterministic repeats compare every energy and coordinate
frame, every Cartesian nuclear gradient, and all 3N velocity components in
the final checkpoint. Intermediate velocities are not separately exported by
the existing driver. The unchanged legacy absolute gates are 1e-7 hartree
for energies, 1e-7 Angstrom for coordinates, 2e-7 hartree/bohr for gradients,
and 1e-7 for velocities in atomic units. V/E/G/H repeats use 2e-7 in the
respective atomic units. All measured errors are retained, not just pass flags.

By default, complete gradients come from the existing printed tables
(quantized to 1e-10 hartree/bohr), or from `bomd_forces.dat` when an accepted
instrumented client provides it. `--force-source printed` or `raw` fixes the
choice. Raw exports must contain every atom/component at every force call and
agree with the printed values within their output precision. Extra diagnostic
IO in an instrumented client remains inside the whole-job clock for both
paired runs; use the standard driver when reporting its normal workflow cost.

Source, basis, inputs, executable and resolved library hashes are checked
before/after every trial. An optional retained build report links the source
manifest and loaded QUICK library. Runtime paths must resolve to the selected
install. No changes to the Hamiltonian, force evaluator, published ESP/RESP
methods or default property selection are introduced.

## Reports and preparation tests

`workflow_summary.json` retains raw commands, provenance, counters, physical
comparisons, phase timings and every paired wall-time delta. `trial_timings.csv`
and `paired_overhead.csv` export the values for publication tables.
`--plot` writes a standalone `workflow_overhead.pdf` with all paired timings
and overhead samples. Statistics include median, minimum, maximum and sample
standard deviation. Negative paired deltas are retained as timing noise;
they are not clipped or interpreted as a guaranteed speedup.

```sh
python3 validation/electrostatics_benchmark/test_benchmark_workflow.py \
  --output /new/workflow-schema-tests
```

These tests launch zero QUICK jobs, SCFs or GPU kernels. They exercise
positive stride/component/native-counter cases; missing/nonfinite/reordered
H data; wrong units/sign/timestamps/probe positions; full raw-force identity;
silent dispersion-token mistakes; native method/basis/grid evidence; matched
namelists; and a preparation path that fails if any subprocess is launched.
They validate workflow tooling, not native numerical performance.

## Completed water workflow controls

The retained `workflow-water-cpu-v2/workflow_summary.json` and
`workflow-water-cuda-v2/workflow_summary.json` both passed on 2026-10-15.
They used the explicit PBE0-D3/Cartesian def2-SVP card above, four probes,
two 0.025-fs steps and three force evaluations. Each backend ran eight fresh
processes: one discarded OFF/ON pair and three retained alternating pairs.
Every enabled job sampled V/E/G/H at all three frames, with exactly zero extra
SCFs and the expected direct-H/native-route counters. Source, loaded runtime,
input and harness fingerprints stayed unchanged.

CPU paired physical differences were exactly zero. CUDA paired coordinates
and every printed gradient matched exactly, with maximum energy and final
velocity differences 1.57e-13 hartree and 4.00e-18 velocity atomic units.
The repeated V/E/G/H comparisons and native printed protocol checks passed.

| Backend | Median OFF whole job (s) | Median ON whole job (s) | Median property API / force API |
|---|---:|---:|---:|
| CPU, one thread | 17.745 | 17.695 | 0.3246% |
| CUDA, one GPU | 16.220 | 16.218 | 0.07914% |

Paired whole-job timing deltas spanned -0.399 to +0.050 s on CPU and -0.049
to +0.045 s on CUDA. For this small case the total overhead was unresolved
within timing noise; negative deltas do not establish a speedup. The separate
API ratios exclude driver IO and are not whole-job overhead. These local,
three-retained-pair results support the tested short workflow only.

The trimer workflows were not executed. WATER20 used the separate one-pass
collector for one 500-AO frozen-density snapshot and 3860 observation points;
it was not a paired BOMD/whole-job overhead campaign. Its independent accuracy
results and finite-sampling limits are documented in [SNAPSHOT_COLLECTOR.md](SNAPSHOT_COLLECTOR.md).
