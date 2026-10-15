# Short WATER20 CUDA transfer check

The bundled `fixtures/water27_h2o20es.xyz` is the byte-identical neutral
singlet WATER27/H2O20es structure used for this transfer fixture: 20 waters,
60 atoms, in Angstrom. Its coordinates are unchanged. The adjacent fixed
probe file, atom-labelled selections, and provenance JSON record the exact
20 probes and their hashes. No workstation path is required.

From the repository root, with NumPy available:

```sh
python3 -B validation/solvation/validate_condensed_phase_transfer.py \
    --driver /path/to/test-api-oeprop-bomd.cuda \
    --workdir /path/to/new-water20-transfer --model PCM
```

The defaults use the bundled fixtures and this checkout's basis directory.
`--repo`, `--basis`, `--geometry`, and `--probes` accept explicit alternatives;
all actual inputs, including an external basis directory, are fingerprinted.
`--coordinate-atoms` defaults to `1 4` for this fixture. The selected indices
must be distinct and present in an alternative geometry.

This requires a serial native CUDA library with ddX enabled. It performs
PBE0-D3/def2-SVP, tight XC/basis screening, the default ddX cavity/volume
grids, a static solute-only x field of 0.0002 atomic units, and ESP/EFIELD/EFG
at every accepted step. The default is two 0.025-fs steps. `--model COSMO`
selects ddCOSMO instead of ddPCM. Use a fresh workdir for each run.

All original gates remain: explicitly converged SCFs, no extra property
SCFs, actual native CUDA field/volume/property calls, solvent/cache lifecycle,
independent energy/drift identities, maximum absolute drift 1e-5 hartree,
net gradient 3e-7 atomic units, and tensor invariants 2e-9. Additional
structural checks verify the initial geometry, checkpoint, fixed probe
coordinates/mode/timestamps, raw-versus-summary quantities, and cache shape
and allocation. Loaded libraries, source, controls, fixtures, and raw outputs
are fingerprinted before/after where applicable.

The probes remain fixed Cartesian locations during the trajectory. They
were initially chosen 1.5 Angstrom from each water's oxygen along a labelled
local direction maximizing clearance to all nuclei. The harness verifies
at least 0.8 Angstrom clearance at every frame; that is a geometric exclusion
criterion, not a demonstrated property-accuracy bound. The values follow
QUICK's ordinary total-QM probe convention with the solvent-polarized
density. They are not environment-only fields.

This is a finite-cluster software/force-transfer check. Two short steps do
not establish an equilibrated ensemble, long-time energy conservation,
spectra, periodic-boundary support, or a timing comparison.
