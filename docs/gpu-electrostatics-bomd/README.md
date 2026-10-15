# QUICK GPU electrostatics and BOMD

## Applied fields and optional equilibrium solvation

The working branch adds uniform-field analytic gradients and optional
continuous-density ddCOSMO/ddPCM with CPU/CUDA operators and live-density
ESP/EFIELD/EFG sampling. See the [interface, model conventions and accuracy
controls](applied-fields-and-ddx.md). The archived checkpoints below
predate the new per-force SCF-convergence guard; their evidence and limits
remain historical. Current release acceptance uses separately pinned
source/runtime reports and retains failed convergence checks.

The [2026-10-15 engineering acceptance report](fields-ddx-acceptance-20261015.md)
and [evidence manifest](fields-ddx-acceptance-20261015.json) record the applied-field
and ddX implementation, GPU validation, optimizations and remaining accuracy limits.

## Master integration, 2026-10-14

The feature includes upstream master
`f33543cae174c3b46d204c22e9a89ab57ac86a4e` in merge
`7db88f1e0019dc1e50d7050aa3ee85a9f4b74e86`. CPU/MPI and CUDA/MPI builds,
all 742 selected standard regression cases, 2,084 electrostatics/BOMD checks,
GPU memory checks and the exact WATER20 zero-step replay passed.
Measured source/runtime identities and accepted results are recorded in the
[integration report](master-integration-20261014.md) and
[integration manifest](master-integration-20261014.json).
That report documents the communicator API, parser/subcommunicator regression,
preservation checks and coverage limits. Historical passes below describe the
earlier checkpoint. Native/CPPE and MBX remain separate projects.

## Historical feature checkpoint, 2026-10-13/14

Checkpoint date: 2026-10-13. Scientific source base:
`208ddbb3468bfdd9198e7952f32f4365f0d432d5`.

Source distribution date: 2026-10-14. Branch: `feature/gpu-electrostatics-bomd`.
The complete development history and manuscript drafts remain in local archival
worktrees and a verified Git bundle. This distribution starts from upstream
master and preserves the accepted implementation without manuscript files
or the manuscript-bearing development commits.

This branch brings together the committed CPU/GPU electric-field and analytic
field-gradient implementation with the nonperiodic BOMD driver. Sampled frames
evaluate ESP, EFIELD and EFG from the live density without an additional SCF or
gradient evaluation. Existing published ESP/RESP routines were preserved.
The engineering notes and historical evidence summaries accompany the accepted
scientific implementation; exporting this branch did not change its source.

The proposed JCIM Application Note focuses on the GPU derivative implementation,
its accuracy and performance, and trajectory-resolved electrostatic analysis.
Native/CPPE polarizable embedding and MBX development remain separate projects.

## Historical evidence before the current master merge

- The accepted engineering run passed 2,084 API, lifecycle and short-BOMD gates
  across serial CPU, CPU MPI2, CUDA and CUDA MPI1, with one physical GPU.
- Derivative transfer checks covered seven models and 8,462 probes. Ten inherited
  near-nuclear ESP discrepancies remain; the prior GPU ESP results were preserved.
  RESP source was preserved, but a fresh RESP-specific runtime test was not run.
- WATER20 has one accepted PBE-D3/DEF2-SVP initial GPU calculation: 13 converged
  SCF cycles, 500 printed basis functions and 20 finite fixed-probe property rows.
  `nsteps=0` means this calculation contains no trajectory. Its geometry is
  unrelaxed, and its timings are single-call diagnostics.
- The earlier diffuse DEF2-SVPD attempt was interrupted without accepted SCF
  convergence. Its negative result remains part of the local evidence.

[GPU engineering summary](evidence/gpu-engineering-summary.txt) and
[WATER20 feasibility summary](evidence/water20-feasibility-summary.txt) are
verbatim historical reports. Statements about local branches not being pushed
describe their state when those reports were sealed. Absolute paths identify
workstation evidence; the full raw archive is not included in this checkpoint.
[The historical manifest](manifest.json) records the report hashes, scientific
file hashes and measured CUDA runtime identities for that checkpoint. It and
the two historical reports are retained unchanged. Their hashes and runtimes
must not be relabeled as the current master integration. Matching source bytes
does not imply that a fresh build reproduces binary bytes or performance.

## Runnable examples and validation

See the [OEPROP acceptance suite](../../validation/oeprop_acceptance/README.md),
[property API](../../validation/oeprop_api/README.md), and
[BOMD driver and overhead benchmark](../../validation/oeprop_bomd/README.md).
These documents provide build-dependent executable paths and example commands.
Manuscript drafts are retained locally and excluded from this source
distribution. Their claims and tables require reconciliation with this evidence.

QUICK prints `G_ij = dE_i/dC_j`, the negative electrostatic-potential Hessian.
Standalone BOMD uses finite molecules without PBC or a thermostat. Current
probe modes do not independently rotate a local frame for each water, and the
current driver rejects unconverged SCF results before accepting a force
evaluation; historical checkpoints predate this guard. This does not establish periodic or
polarizable QM/MM dynamics.

## Work remaining for submission

1. Recheck the final implementation against independent references; measure
   absolute CPU/GPU scaling and paired whole-job property-sampling overhead.
   Existing optimization percentages use specified historical baselines.
2. Prepare and relax selected WATER20 structures, validate rotating probes,
   converge numerical settings and timestep, and complete an interpretable
   trajectory demonstration with sampling appropriate to the claimed observable.
3. Compare with existing implementations, update the manuscript, and archive
   portable inputs, analysis scripts and raw results for the publication release.

WATER20 geometry source:
<https://github.com/esoteric-ephemera/water_cluster_density_errors>,
pinned upstream commit `6ae4a79e958efb20c3d99f05498b6c358b0f3f9b`.
