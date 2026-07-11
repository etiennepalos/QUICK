# OEPROP API Validation

`test-api-oeprop` exercises the no-I/O `getQuickOEPROP` interface after a
converged QUICK density. Probe coordinates passed to the API are in bohr; the
trajectory helper reads probe coordinates in Angstrom and converts them.

The driver has three modes:

```text
validate   Run water RHF/STO-3G and write ESP, EFIELD, and EFG at four probes.
trajectory Read a multi-frame XYZ file and write one CSV row per frame/probe.
benchmark  Reuse one water density on deterministic grids of 64-65536 points.
```

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
