# Master integration validation: 2026-10-14

Status: **pending**. This is a validation scaffold, not an acceptance report.
The [machine-readable manifest](master-integration-20261014.json) deliberately
leaves final identities, counts and results unset until the corresponding raw
evidence is reviewed. No scientific pass is inferred from the historical files.

## Source scope

The source-only feature export `dba43da12d0a0528c2af667b29671ea21c870ab3` was
merged with upstream master `f33543cae174c3b46d204c22e9a89ab57ac86a4e` in
`7db88f1e0019dc1e50d7050aa3ee85a9f4b74e86`. Follow-up optimizer compatibility
work is part of the integration under validation; the merge alone does not
identify the final measured source. Record its final commit or exact dirty-tree
snapshot, binary/library identities, build commands and inputs before acceptance.

This integration must retain upstream's overlap, SCF, basis and LRI changes,
the established ESP/RESP numerical methods, and the feature's optimized
EFIELD/EFG kernels. Ordinary CLI optimizer behavior must agree with master;
the live-density API and new derivative/surface workflows additionally require
geometry, density and energy to refer to the same evaluated endpoint. Validate
both optimizer routes and repeated jobs after the compatibility fix is frozen.

## MPI and keyword contracts

MPI API clients now call `setQuickMPI(comm, ierr)` before `setQuickJob`.
QUICK duplicates the supplied communicator. The old rank/size argument pair is
not the current interface. `MPI_COMM_WORLD` is a valid caller choice; a split
communicator is also valid.

Library operations use `quick_comm`, `quick_comm_rank`, `quick_comm_size` and
`quick_mpi_error` from `quick_mpi_module`. Surface broadcasts, shell partitioning,
electronic reductions and returned property broadcasts must stay within that
communicator. Electronic contributions are reduced to its rank zero; classical
nuclear/external-charge terms are added once before returning the full result.
World-rank bookkeeping during API initialization is distinct from collectives
that operate on a QUICK job.

Feature keywords use upstream's boundary-aware `found_keyword` parser. The
regression checks property aliases, numerical EFG implication, density controls,
field/origin aliases, rejected substrings, repeated matches and tab boundaries.

## Parser and split-communicator regression

From the repository root, after building the CPU serial and MPI API drivers:

```bash
QUICK_INTEGRATION_EVIDENCE=/home/epalos/software/quick_ep/master-integration-20261014
python3 validation/oeprop_api/validate_master_integration.py \
  --serial-driver "$QUICK_INTEGRATION_EVIDENCE/build/integration-cpu/src/test-api-oeprop" \
  --mpi-driver "$QUICK_INTEGRATION_EVIDENCE/build/integration-cpu/src/test-api-oeprop.MPI" \
  --launcher 'mpirun -np 4' \
  --workdir "$QUICK_INTEGRATION_EVIDENCE/parser-subcomm-final"
```

Use a new evidence directory for every attempt. This standard-library Python
harness runs 18 parser cases, two serial reference jobs, and four CPU MPI ranks
split into two independent groups of two. The groups use different molecular
jobs, probe counts, surface isovalues and property-call counts; each must agree
with its own serial reference. Interleaved groups and a timeout expose accidental
world collectives. Default absolute tolerances are `1e-10` hartree for energy,
`1e-8` Angstrom for coordinates and `1e-9` a.u. for properties. These are small
regression comparisons, not independent accuracy or large-rank scaling claims.

The driver also exposes `parser-contract`, `subcommunicator-reference` and
`mpi-subcommunicator` modes. Use the Python harness for acceptance because it
checks raw outputs, ordering, convergence/lifecycle state and identities.

## Published ESP preservation

The local evidence runner `compare_published_esp.py` in
`/home/epalos/software/quick_ep/master-integration-20261014/` compares four
unchanged published test inputs on pristine master CPU/CUDA and integration
CPU/CUDA. It retains commands, outputs, source/build/runtime identities, explicit
SCF convergence and finite-value checks. Expected coverage is three MKS surface
and unrestrained ESP-charge fits (neutral, anionic and unrestricted cases) plus
an external-grid case, all at B3LYP/def2-SVP. The original absolute thresholds
are `1e-5` electron for fitted charges, `2e-7` for ESP table entries and `4e-5`
hartree for energy. Its local coverage file provides the full invocation.

`ESP_CHARGE` is an **unrestrained** QUICK fit. No Amber `resp` executable is
available in this environment. Preserving QUICK's ESP/MKS input for downstream
RESP does not establish a fresh external RESP or rwRESP fit; those runtime
coverage flags remain false. Two-stage restraints, orientation averaging and
reweighting are not exercised by the four QUICK fixtures.

## Evidence to complete

| Evidence | Status |
|---|---|
| Final source snapshot and four fresh build/runtime identities | Pending |
| Upstream CPU/CUDA regression controls and integration differences | Pending |
| Published ESP input/surface/charge/grid preservation | Pending |
| Parser and independent split-communicator jobs | Pending |
| Final API, optimizer/lifecycle and derivative acceptance | Pending |
| Short BOMD consistency and actual GPU property launches | Pending |
| External Amber RESP/rwRESP execution | Unavailable; not claimed |

Record baseline failures as well as integration-only failures. Keep failed or
superseded runs with their actual identities; only accepted runs on the final
source may populate the current acceptance summary. Retained raw evidence is
local until separately archived. This scaffold supplies no new speedup,
multi-GPU, HIP, periodic, polarizable QM/MM or long-time sampling result.

## Historical evidence

[manifest.json](manifest.json), [GPU engineering summary](evidence/gpu-engineering-summary.txt)
and [WATER20 feasibility summary](evidence/water20-feasibility-summary.txt) describe
the earlier source/runtime checkpoint. They remain unchanged and separate from
this integration report. WATER20 is still a single unrelaxed `nsteps=0` initial
calculation, not a trajectory or an independent reference comparison.
