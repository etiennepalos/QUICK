# HPC Template

Prepare `microdroplet_initial.xyz`, define chemically meaningful probes in
`reactive_site_probes.dat`, copy `microdroplet.nml.template` to
`microdroplet.nml`, and adjust charge, multiplicity, atom indices, method,
time step, and allocation for the target system. Submit with:

```bash
export QUICK_ROOT=/path/to/quick/install
sbatch run_oeprop_bomd.slurm microdroplet.nml
```

Start with a short NVE equilibration and verify SCF convergence, energy drift,
checkpoint restart, and OEPROP throughput before requesting a long trajectory.
The template intentionally does not include a large coordinate file or claim
that eight ranks is optimal; benchmark the actual system and machine.

## PCET application path

For a proton-coupled electron-transfer trajectory, define donor, acceptor,
and transferring-proton atom indices before production. Use off-nuclear
probe points near the donor/acceptor and along the proton-transfer axis;
atom-coincident ESP/EFIELD/EFG needs an explicit self-source exclusion rule.
Record at least the donor-acceptor distance, the signed proton-transfer
coordinate, projected field, axial EFG, and a state-energy gap when two
electronic states are available.

Reactant and product calculations must use independent state-specific
workers and restart densities. For QUICK-MBX, re-equilibrate induced dipoles
for each state instead of copying polarization between states. Run and
validate each state separately before launching paired energy-gap sampling.

The existing uniform external field is not a constant-potential electrode.
Always record the field model, origin, total charge, electrode potential
convention, and any externally supplied double-layer potential or electrode
density of states. See `paper/quick_pcet_electrochemistry_roadmap.md` for the
staged QUICK-to-ASE-to-pyPCET workflow and its acceptance tests.
