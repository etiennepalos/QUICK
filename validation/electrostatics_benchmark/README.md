# Direct fixed-density electrostatics benchmark

This serial API client measures V, E, G, H, V/E/G and V/E/G/H separately.
V/E/G uses the current three native property routes; it is not a fused kernel.
Each property clock includes the selected API's allocations, uploads, native
evaluation, copies and direct classical contributions. Output writing and the
initial SCF are outside those property clocks. The setup/SCF and full client
lifetime are separately reported; they are not a whole-BOMD benchmark.

Compile against an explicitly selected installed CPU or CUDA QUICK library:

```sh
python3 validation/electrostatics_benchmark/compile_client.py \
  --compiler /path/to/matching/gfortran --prefix /path/to/quick/install \
  --ddx-lib /path/to/ddx/lib --backend cuda --output /new/artifact/benchmark-cuda
```

Run in a new directory with `QUICK_BASIS` pointing to the matching basis data:

```sh
/new/artifact/benchmark-cuda input.dat /absolute/path/to/density.dat 3
```

An optional fourth argument is `VEG` or `VEGH` (default). `VEG` omits both
Hessian timing routes and writes explicitly unavailable `NaN` Hessian columns,
with `hessian_requested F` in the result. Use it for matched V/E/G experiments
whose CPU reference would otherwise spend most of its time evaluating H.

The input matches the existing field-Hessian API client's `api` input:
one keyword line (at most 256 characters), `natoms ncharges npoints`, then
`Z x y z` atom rows in Angstrom, `x y z q` external-charge rows in Angstrom,
and observation-point triples in bohr. Pass `-` for the density argument to
retain the initial converged density. A replacement density has the existing
`density.dat` layout: `nbasis T/F`, column-major alpha density, then beta
density if unrestricted. Geometry, basis and spin must match the exported
density; callers are responsible for verifying the complete frozen-state key.
The initial SCF energy is not the energy of an arbitrary replacement density.

`timings.csv` retains every call and native launch counter. Iteration zero is
the first call for that route; subsequent samples can estimate warmed cost.
It is not a fresh-process cold call because earlier routes may initialize
shared infrastructure. `properties.dat`, `density.dat`, `ao_basis.dat` and
`overlap.dat` use the existing field-Hessian validation layouts. `result.dat`
records convergence, unchanged density/Fock state and extra SCF API calls.

For CPU/CUDA comparisons, export one density and use that exact file for both
backends. Fingerprint input geometry, AO basis, density, executable, loaded
libraries and source before interpreting timing or numerical agreement.
The observable includes electrons, nuclei and explicitly enabled MM charges.
Direct imposed fields and explicit ddX reaction backgrounds are excluded.
No existing Hamiltonian, force routine, property API or default is modified.

`benchmark_direct.py` orchestrates matched-density runs, including immutable
build/source and runtime-library hashes. It archives its exact helper sources
in each new result directory. Use `--outputs VEG`, `--source-snapshot` and
`--build-report` to select the output set and retained compiled provenance.
`plot_direct.py` exports all retained phases to CSV and V/E/G(/H) figures.
