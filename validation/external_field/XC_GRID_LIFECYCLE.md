# Optional XC-grid parser and job lifecycle validation

`quick_xc_grid_lifecycle_driver.f90` is a test-only serial/MPI public-API
client. In one process it runs water PBE0/STO-3G with this sequence:

```text
default SG1 -> 75x302 -> default SG1 -> 100x434 -> 50x194
            -> default SG1 -> SG0 -> SG0
```

Every job requests analytic gradients, performs one energy/gradient API call,
then obtains E/G at a retained-density probe. The client checks optional-grid
method reset before and after the SAD guess, equal per-rank initial point
counts, valid grid sizes, released temporary arrays, unchanged density/Fock
state and zero additional energy API calls. Full nuclear gradients and E/G
are exported to `lifecycle.csv`; each rank also exports its grid counts.

Compile against the explicitly selected install and matching compiler:

```sh
gfortran -cpp -ffree-line-length-none -I src/util \
  -I /selected/cpu/install/include/amber-modules/quick/serial \
  validation/external_field/quick_xc_grid_lifecycle_driver.f90 \
  -L /selected/cpu/install/lib -Wl,-rpath,/selected/cpu/install/lib \
  -lquick -o /new/xc-lifecycle-cpu
```

For MPI, use the matching `mpifort`, add `-DMPIV`, select the `quick/mpi` module directory and link
`-lquick_mpi`. A ddX-enabled install can additionally require ddX library
`-Wl,-rpath` and `-Wl,-rpath-link` options. These clients use no CUDA.

The portable orchestrator executes parser cases and the successive jobs:

```sh
python3 validation/external_field/validate_xc_grid_lifecycle.py \
  --cpu-driver /new/xc-lifecycle-cpu --cpu-basis /selected/cpu/install/basis \
  --mpi-driver /new/xc-lifecycle-mpi --mpi-basis /selected/mpi/install/basis \
  --mpiexec /matching/mpiexec --library-path /matching/runtime/lib \
  --library-path /selected/ddx/lib --workdir /new/lifecycle-evidence
```

MPI flags can be omitted for a serial-only campaign. The subprocesses use
one thread per rank. Parser checks run without SCF: missing controls, zero,
negative and oversized dimensions, unsupported angular rules, known negative
weight rules 74/230/266, HF and SG0 rejection, keyword-boundary lookalikes,
and supported minimum/maximum dimensions. The v5 extension additionally
checks both controls for decimals, NaN, text, sign-only values, integer
overflow, exponent/comma input, missing equals/value and duplicates, plus
valid whitespace/tabs, positive signs and leading zeros (44 parser cases).
Maximum dimensions are parsed only; the campaign does not build that large
grid.

Fixed gates are 2e-12 hartree for same-backend repeated energies and 2e-11
for repeated gradients/E/G; serial/MPI gates are 2e-10 hartree and 2e-9
for gradients/E/G. State mutation and extra API calls must be exactly zero.
The report retains commands, runtime library/source/client hashes, every
check, native inputs/outputs and execution logs. This validates software
lifecycle behavior; it does not establish a converged XC quadrature for a
chosen scientific application.

The early job-local broadcast of the new `init_ngpts` in grid construction
is needed before worker ranks traverse fresh temporary arrays: the existing
SSW MPI setup previously read that saved count before broadcasting it.
Moving from a larger refined grid to a smaller grid exercises that boundary.

Evidence from 2026-10-15:
`/home/epalos/software/quick_ep/adaptive-electrostatics-20261015/xc-grid-lifecycle-v1/report.json`
passes 189 checks / 21 native runs with v4 serial and two-rank MPI installs.
Default SG1/SG0 repeated energies, full gradients and E/G are exact. Maximum
serial/MPI differences are 1.99e-13 hartree, 2.31e-13 hartree/bohr in nuclear
gradients, and 5.58e-15 over E/G components.

Final v5 evidence is `xc-grid-lifecycle-v2/report.json` PASS220/46 native
runs (serial/MPI2) and `xc-grid-lifecycle-default-off-v2/report.json`
PASS129/45 runs, including all 44 parser cases. The additional lexical/I/O
checks return error 57 through the client for malformed new controls.
`xc-grid-final-comparisons-v1/report.json` PASS48 compares every valid-case
energy, full gradient and E/G across compiled ddX ON/OFF and parser v4/v5;
all differences are exactly zero. Earlier v4 reports are retained intact.
