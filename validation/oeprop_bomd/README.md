# OEPROP BOMD Driver

`test-api-oeprop-bomd` is a deterministic CPU velocity-Verlet driver around
the public QUICK energy/gradient and OEPROP APIs. At every accepted step it
performs exactly one SCF/gradient call and then evaluates ESP, EFIELD, and EFG
from that live density. OEPROP does not trigger another SCF.

Run the checked-in water example from the repository root:

```bash
export QUICK_BASIS=$PWD/basis
/path/to/test-api-oeprop-bomd validation/oeprop_bomd/water_fixed.nml
```

The namelist is `quick_bomd`. Important fields are:

| Field | Meaning |
|---|---|
| `geometry_file` | Initial single-frame XYZ; omitted for restart runs |
| `keywd` | QUICK method and basis; `GRADIENT` is added if absent |
| `nsteps`, `dt_fs` | Additional accepted steps and time step in fs |
| `temperature_k` | Deterministic initial kinetic temperature; no thermostat |
| `nfrozen`, `frozen_atoms` | Optional 1-based frozen atom indices |
| `checkpoint_file` | Text checkpoint written after each accepted step |
| `restart_file` | Existing checkpoint; preserves absolute step, time step, and reference NVE energy |
| `coordinate_atoms` | Two atoms defining the reported distance coordinate |

Probe modes:

| `probe_mode` | Probe-file records | Motion |
|---|---|---|
| `fixed` | `x y z` | Fixed laboratory coordinates |
| `molecule` | `u v w` | Body-frame coordinates based on `anchor_atoms` |
| `atom` | `atom x y z` | Fixed Cartesian offset from a selected atom |
| `density` | none | Regenerated electron-density isosurface |

Probe-file coordinates are Angstrom. `getQuickOEPROP` receives bohr. The
property CSV reports QUICK atomic units with

```text
E_i(C) = -dV(C)/dC_i
G_ij(C) = dE_i(C)/dC_j.
```

Outputs are `<prefix>_energy.csv`, `<prefix>_trajectory.xyz`,
`<prefix>_oeprop.csv`, `<prefix>_summary.json`, and the checkpoint. The OEPROP
CSV includes all tensor components, trace, Frobenius norm, three eigenvalues,
anisotropy, and the selected distance coordinate.

Run the targeted validation with an installed serial build:

```bash
python3 validation/oeprop_bomd/validate_oeprop_bomd.py \
  --bomd-driver /path/to/test-api-oeprop-bomd \
  --oeprop-driver /path/to/test-api-oeprop \
  --workdir /private/tmp/quick-oeprop-bomd \
  --run-pbe0
```

Add `--mpi-driver /path/to/test-api-oeprop-bomd.MPI --launcher "mpirun -np 2"`
for the two-rank check. The validator covers NVE drift, on-the-fly versus
post-processed properties, split-run restart reproducibility, all four probe
modes, a water trimer, PBE0-D3/def2-SVPD, EFG eigensystem invariants, and
serial/MPI parity.

This is a compact production/API driver, not a thermostat or path-integral
dynamics package. Large droplets, slabs, and long trajectories belong on HPC.
