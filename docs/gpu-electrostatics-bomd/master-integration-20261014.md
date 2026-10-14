# Master integration validation: 2026-10-14

Status: **accepted within the scope below**. The
[machine-readable manifest](master-integration-20261014.json) records identities,
accepted evidence and coverage limits. All required code-integration scientific
gates passed; none was waived. Raw evidence is sealed locally, with no public
archive identifier.

Measured source: `d1d7653a01643f60d03d29614ceb038523d5addf`; tree
`52b3eda060ded827c1ba3be9fb5331a7b80bc0b8`. This integrates feature export
`dba43da1` with upstream `f33543ca` through merge `7db88f1e`, followed by
optimizer compatibility and exact numerical-grid keyword fixes. The manifest
fingerprints 3,165 source/build/generator/validator/basis/fixture inputs, including
all 1,632 `src/` files, and all four successful build/runtime records. A later
documentation-only publication commit must preserve these measured inputs.

| Evidence | Accepted result |
|---|---|
| Fresh master CPU/CUDA and integration CPU/MPI + CUDA/MPI builds | Configure/build phases returned zero; executable/library identities verified |
| API, derivatives and lifecycle | **1,187/1,187** checks: 533 + 297 + 357 |
| Frozen short BOMD | **897/897** checks; 48 cases and 60 retained commands |
| Parser/split communicator | **18/18** cases; two independent two-rank groups within four CPU MPI ranks |
| Standard master CPU / integration CPU / master CUDA / integration CUDA | **183/183, 191/191, 184/184, 184/184** selected cases; zero failures/skips/differences/mutations |
| Optimizer endpoints | **20** cases across CPU, CPU MPI2, CUDA and CUDA MPI1 |
| Standalone analytic/numerical EFG grids | **16** jobs across four backends; two methods and 663 probes/job |
| Published ESP preservation | **16/16** runs; eight primary and eight CPU/CUDA comparisons passed |
| Extreme-coordinate stress | 129 probes; finite API properties at `1e62`, `1e70`, `1e80` Angstrom |
| CUDA memcheck/initcheck | Four instrumented runs: zero errors; both memchecks zero leaks; eight common-density comparisons passed |
| Exact WATER20 initial evaluation | Passed: 500 AOs, 20 probes, `nsteps=0`; no dynamics |
| Protected worktrees and advanced target | 22,318 files before advance; 19,141 mapped nonadvanced files after; all 3,165 validated inputs match the advanced target |

The [independent evidence audit](/home/epalos/software/quick_ep/master-integration-20261014/accepted-evidence-independent-review.json)
recomputed accepted API comparisons, frozen BOMD and WATER20 metrics from raw
files without launching science, and checked the remaining raw/provenance gates.
The [sealed local evidence manifest](/home/epalos/software/quick_ep/master-integration-20261014/final-evidence-manifest.json)
maps 3,905 files; SHA-256:
`82b27baeccc3b5b41b8af46dadcd43fe177b705e285be1814d6bd490bb9f0794`.
All sealed files, protected mapped files and advanced-target input fingerprints
were independently rehashed with zero mismatches. Later publication metadata is
kept separate from the seal; a new `publication-source-proof.json` can bind a
documentation-only publication commit to the same measured inputs.

MPI clients use `setQuickMPI(comm, ierr)`; QUICK duplicates the caller's
communicator and uses job-local `quick_comm`, rank and error state. Runtime
coverage includes CPU MPI2 and CUDA MPI **one rank on one physical GPU**.
The API report's internal `gpu_mpi2` label does not change the actual rank count.
Frozen BOMD records 28 launches of each property per CUDA backend and zero extra
property-triggered SCF calls; maximum short-test energy drift is
`1.0912440018273628e-7` hartree. Each GPU optimizer endpoint records three
successful launches per property. Standalone quick grid jobs have GPU/dispatch
and numerical evidence, without exported kernel-counter CSVs.

Published ESP fitting/surface/grid outputs remain within the original absolute
`1e-5` electron, `2e-7` table and `4e-5` hartree tolerances. Maximum
same-backend ESP differences are `1.2140000002491824e-7` a.u. on CPU and
`9.999998051846148e-11` a.u. on CUDA; maximum charge difference is
`2.0000000000575113e-6` electron. All eight primary printed-energy differences
are zero. Runtime outputs are not claimed byte-identical. `ESP_CHARGE`
remains unrestrained; no external Amber RESP/rwRESP executable was available,
so those fitter runtime coverage flags remain false.

The CPU selected list contains 182 shared legacy cases plus nine feature cases:
it replaces the B3LYP external-grid entry with PBE0-D3BJ/cc-pVDZ and adds eight
other cases. The B3LYP grid is separately covered by published ESP preservation.
CUDA selected lists match master. Shared inputs/references are unchanged; all
12 original `check_*` functions and their thresholds remain byte-identical.
Four f-function fixtures (`ENABLEF=FALSE`) and one indexed checkpoint fixture
(`HDF5=OFF`) are explicitly excluded, unvalidated coverage.

Initial targeted CPU controls found five optimizer regressions. The scoped
compatibility fix restores ordinary CLI screening/endpoint defaults while
retaining evaluated endpoints for live API/derivative/surface workflows; all five
now pass geometry and dipole checks, and all backend endpoint checks pass.
The initial CUDA unsupported-f-function/harness abort remains a failed,
unaccepted attempt. The v1 acceptance campaign is superseded. The v2 BOMD
inventory failure when `validate_grid_parser.py` became tracked was never
accepted; its frozen rerun supplies the 897 passes. Failed and superseded
attempts remain retained.

The exact WATER20 PBE-D3/def2-SVP replay converged in 13 SCF cycles. Historical
control, geometry and probes are unchanged. Energy differs by
`3.637978807091713e-12` hartree and the largest property difference is
`1.4566126083082054e-13` a.u. There is one force/energy-gradient API call,
one property call and one ESP/EFIELD/EFG launch each, with zero extra SCFs.
This is a nonperiodic step-zero regression against the earlier engine, not an
independent electronic-structure oracle, trajectory or full-gradient comparison.

The extreme-coordinate stress test establishes finite API behavior, without
relative far-field accuracy or a speedup claim. Its legacy fixed-width ESP
coordinate columns can print asterisks at those distances. Sanitizer
common-density property differences are at most `8.526512829121202e-14` a.u.
There is no new performance, HIP, multi-GPU, periodic/polarizable BOMD or
production-length sampling claim. Historical checkpoint reports remain unchanged.
The local-only archive status is explicit and does not block code acceptance.
