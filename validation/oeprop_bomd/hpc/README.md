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
