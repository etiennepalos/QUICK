# Read-only direct/adaptive BOMD comparison

`validate_adaptive_bomd.py` analyzes existing native driver outputs. It launches
no process, SCF or GPU kernel and does not change retained inputs or outputs.

```sh
python3 validation/adaptive_electrostatics/validate_adaptive_bomd.py \
  --direct /retained/direct \
  --adaptive /retained/requested-approximation \
  --adaptive /retained/cap-forced-fallback \
  --expect-all-fallback --output /new/read-only-comparison
```

Use `--expect-accepted` when at least one accepted approximation frame is
required. Omit both flags to classify whatever the actual diagnostics report.
Directory names never determine acceptance: a directory called `approximate`
can contain only fallback frames, as the first CPU control did.

Each directory supplies `control.nml`, the native summary/energy/XYZ/property/
direct-H/rank-counter files, the complete printed QUICK gradients and a final
checkpoint. Prefix, probe and checkpoint names come from the controls; files
are not renamed. This validator currently requires complete step-zero paths,
fixed probes and `field_hessian=.true.` on both sides. Method cards, dynamics
controls, geometry and probes must match. It checks every energy and coordinate
frame, every Cartesian gradient, and all final 3N velocities, using the
unchanged legacy gates: 1e-7 hartree, 1e-7 Angstrom, 2e-7 hartree/bohr and
1e-7 velocity atomic units. Intermediate velocities are not separately
exported. Direct H comparisons use 2e-7 in their atomic units. Full measured
discrepancies are retained even when they are exactly zero.

The optional current GPU telemetry is checked semantically when present.
Every row must report CPU anchors and the actual query backend. Requested
and accepted CUDA frames require a CUDA query and a launch for every attempted
cover level. A fallback row reports `native direct`; it can legitimately retain
CUDA launches from rejected candidate levels. The summary launch count must
equal the frame sum, its request flag must match the controls, CPU anchors and
bounds must be declared, device bytes live after the run must be zero, and
actual CUDA launches require a positive recorded last work allocation.
Partial telemetry is rejected. Older CPU files without these columns remain
supported, with `gpu_telemetry.available=false` and an explicit statement
that these dispatch/allocation checks were unavailable. A requested CUDA
case cannot pass using missing legacy telemetry.

```sh
python3 validation/adaptive_electrostatics/test_adaptive_gpu_telemetry.py \
  --output /new/telemetry-contract-tests
```

The 19 synthetic fault-injection tests check missing launches, false backend
claims, inconsistent sums, retained allocations, accepted GPU depth counts,
candidate-launch-plus-fallback semantics and legacy CPU compatibility. They
launch no native jobs or GPU kernels; actual backend acceptance is separate.

Actual native convergence events and API/rank counters must give exactly one
SCF per force evaluation and zero extra property SCFs. V/E/G and H sampled
steps/probe positions/time stamps must match the stride. The H convention and
units, native method settings and printed invariants are checked. GPU direct
V/E/G launch counts must equal the direct/fallback sample count; accepted
interpolation work uses its separate implementation and does not imply a
published-direct kernel launch.

Every adaptive CSV frame retains its fallback flag, degree/mode, counts,
truncation budget and reason. Accepted frames require finite nonnegative
budgets within requested tolerances. Failed preparation can leave NaN
truncation budgets; those are allowed only on fallback frames and represented
as JSON null while raw CSV remains unchanged. `numeric_budget_known` must be
false and the summary must say values are rebuilt each frame and H is direct.
The total numerical/anchor budget remains unavailable, never assumed zero.
The CSV does not export a numeric total budget, so its value is not inferred.

At every sampled frame, empirical differences from direct V/E/G use full
vector/Frobenius norms. Fallback gates use the existing 2e-7 property tolerance;
accepted-frame diagnostic gates use each requested tolerance plus 2e-7. These
are comparisons at actual supplied probes, not a continuous total-error
certificate. Returned analytic-majorant budgets concern Taylor truncation;
unknown integral/anchor/floating-point errors remain separate.

The first CPU campaign `adaptive-bomd-cpu-v1/analysis-v2/report.json` passed
both three-frame comparisons. Both the requested-approximation case and the
tight resource-cap case fell back for all three frames. Forces, energies,
coordinates, final velocities, V/E/G and direct H matched exactly. This is
**whole-frame fallback acceptance**, with zero accepted approximation frames.
The requested G tolerance was 1e-3, while its attempted maximum G truncation
budget was about 7.45e-3. That campaign alone did not demonstrate an accepted
approximation; the following finite-region cases did.

The later CPU v8 small-region case did accept all three actual frames. Its
fixed box was `[-.025,-.025,4.975]` to `[.025,.025,5.025]` bohr, a 0.05-bohr
box around `[0,0,5]`, using the same first probe triplet. One cubic anchor
per frame met the requested V/E/G tolerances `[1e-5,1e-4,1e-3]`. Worst
observed scalar/vector/Frobenius differences from direct were
`1.62e-10 / 3.81e-8 / 1.10e-5` in the respective atomic units. All physical
trajectory values and direct H remained exactly unchanged. The final compatible
analysis is `adaptive-bomd-small-roi-cpu-v8/analysis-accepted-telemetry-compatible-v3/report.json`;
its current CPU query/zero-device-allocation telemetry passed. Automatic-domain
and forced-cap controls still fell back for all frames. This demonstrates the
finite-region dependence on those three short frames, not a large-domain
acceptance, production trajectory, GPU dispatch or performance conclusion.

The corresponding native CUDA v8 campaign also passed. These were tiny
HF/STO-3G water controls: two 0.025-fs integration steps, three sampled frames
including step zero, the same fixed 0.05-bohr region and first probe triplet,
and the same `[1e-5,1e-4,1e-3]` V/E/G tolerances. The accepted run used one
cubic anchor and one actual CUDA query launch on every frame (three launches
total). Anchors and support bounds were evaluated on CPU; only the query
evaluation used CUDA. Every frame rebuilt its density-dependent data from
the current full SCF density, with three force SCFs, three property samples
and zero extra property SCFs/API steps. Direct H remained enabled separately.

`adaptive-bomd-small-roi-cuda-v8/analysis-accepted-v1/report.json` records
three accepted frames with maximum observed V/E/G scalar/vector/Frobenius
differences `1.62e-10 / 3.81e-8 / 1.10e-5` in the respective atomic units.
Every printed nuclear gradient and coordinate matched direct exactly;
maximum energy and final velocity differences were `3.25e-19` hartree and
`6.24e-19` velocity atomic units. The maximum direct-H difference was
`2.78e-17` in H atomic units. Actual backend/counter telemetry passed, with
zero device bytes live after the run.

The CUDA forced-fallback control
`adaptive-bomd-small-roi-cuda-v8/analysis-fallback-v1/report.json` passed all
three frames as whole-frame native-direct fallback. Each frame first attempted
one CUDA candidate query; its V/E/G results were then replaced by the direct
route and matched the reference exactly. The actual query backend, candidate
launch sum, direct-route counters and zero live device allocation were checked.
Together these short controls establish accepted finite-region and fallback
integration behavior, not an ensemble, broad-domain acceptance or performance
claim. The total numerical/anchor error budget remains unavailable.

Retrospective SHA-256 hashes identify all retained evidence and prove that
this read-only analysis left it unchanged. They do not establish an omitted
historical runtime fingerprint. Execution-time source/executable/dependency
provenance must come from the campaign's separate retained records. These
short paths do not supply an equilibrium ensemble or performance statistics.
