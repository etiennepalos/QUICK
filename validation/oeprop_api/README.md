# OEPROP API Validation

`test-api-oeprop` exercises the no-I/O `getQuickOEPROP` interface after a
converged QUICK density. Probe coordinates passed to the API are in bohr; the
trajectory helper reads probe coordinates in Angstrom and converts them.

The driver preserves these three original modes:

```text
validate   Run water RHF/STO-3G and write ESP, EFIELD, and EFG at four probes.
trajectory Read a multi-frame XYZ file and write one CSV row per frame/probe.
benchmark  Reuse one water density on deterministic grids of 64-65536 points.
```

It also provides `acceptance`, `acceptance-gradient`, `lifecycle-gradient`, and
`benchmark-case` modes. The [acceptance harness](../oeprop_acceptance/README.md)
documents fixed-density CPU/MPI/GPU comparisons, gradient/job lifecycle checks,
and warmed property-only timings for supplied molecular cases.

`optimizer-endpoint [prefix] [lopt|dlfind|both]` checks that an optimizer returns
the last SCF-evaluated geometry together with its matching density and energy.
It covers the five-cycle limit and controlled convergence; DL-FIND also covers
a one-cycle limit. Convergence thresholds deliberately select the exit branches
without claiming a physical minimum. Each case compares returned coordinates
with the final `GEOMETRY INPUT`, then runs an independent fresh SCF at those exact
coordinates with matching initial and final integral screening. The reference switches from
an inline optimizer keyword to an ENERGY template and checks new-job state reset.
Energy, density, ESP,
EFIELD, and EFG must agree within `2e-7`; repeated property calls must preserve
the API step and values. GPU builds must restore released molecular state,
execute all three kernels, and reject properties after job deletion with error 47.
The summary, returned coordinates, property comparisons, and per-rank kernel
counts are written under the supplied prefix.
`both` runs the LOPT cases followed by default DL-FIND cases in one process,
verifying optimizer selection resets across jobs.

The Python validation compares file-based `ESP_GRID`, `EFIELD_GRID`, and
`EFG_GRID` output with the no-I/O API, checks the three-frame trajectory, and
verifies that a pre-density API call fails cleanly:

```bash
python3 validation/oeprop_api/validate_oeprop_api.py \
  --quick /path/to/quick \
  --api-driver /path/to/test-api-oeprop \
  --workdir /private/tmp/quick-oeprop-api
```

For a two-rank MPI check, use matching MPI executables:

```bash
python3 validation/oeprop_api/validate_oeprop_api.py \
  --quick /path/to/quick.MPI \
  --api-driver /path/to/test-api-oeprop.MPI \
  --launcher "mpirun -np 2" \
  --workdir /private/tmp/quick-oeprop-api-mpi
```

`getQuickOEPROP` is collective in an MPI build. Electronic shell-pair
contributions are reduced internally, classical source terms are added once,
and each requested output is broadcast so all ranks return identical arrays.

The checked-in `grid_scaling_m4_pro_serial.csv` records one local Apple M4 Pro
run. It is a scaling smoke test, not a portable performance baseline.
