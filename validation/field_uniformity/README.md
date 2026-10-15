# Offline field variation at a frozen QUICK density

This analysis compares constant, linear and quadratic reconstructions with
direct observation-point queries at one retained density and geometry:

\[
 E^{(0)}(u)=E(C),\quad E^{(1)}(u)=E(C)+G(C)u,\quad
 E^{(2)}(u)=E(C)+G(C)u+\tfrac12 H(C)[uu].
\]

The conventions are `G_ij=dE_i/dC_j` and
`H_ijk=d²E_i/dC_j dC_k=-d³V/dC_i dC_j dC_k`.
Coordinates are in bohr and fields/derivatives are in atomic units. These
are observation derivatives, not nuclear energy derivatives. Source-only
queries include electrons, nuclei and explicitly enabled MM point charges.
Direct imposed-field and explicit continuum-reaction backgrounds are
excluded. The retained density may already respond to those Hamiltonian
terms. Nothing here changes QUICK's Hamiltonian, forces or property routes.

## Query, evaluate independently, convert, analyze

Frozen-density analysis applies to any explicitly identified retained density,
including gas-phase HF. Publication conclusions require convergence assessment
of the chosen physical protocol (SCF, basis, XC quadrature and continuum
parameters where applicable). The scripts themselves perform no SCF, GPU or
molecular-dynamics execution. Each output directory must be new.
The existing serial client in `validation/electrostatics_benchmark` can
collect all native values; it currently evaluates V/E/G/H at every query.
Only E/G/H at centers and direct E at displaced points are needed here.

1. Supply `centers.csv` with `center_id,x_bohr,y_bohr,z_bohr`; center IDs must
   be unique. Supply a `model.json` with `atoms` as `[Z,[x,y,z]]` rows in
   Angstrom, `charges` as `[x,y,z,q]` rows in Angstrom, and `basis`, `charge`
   and `spin` (Nalpha-Nbeta). Supply one complete accepted QUICK keyword card
   in `keyword.txt`, at most 256 characters; MM sources require EXTCHARGES.

   ```sh
   python3 validation/field_uniformity/query_plan.py \
     --centers centers.csv --radii-bohr 0.01 0.02 0.04 0.08 \
     --fibonacci-count 32 --model model.json --keyword-file keyword.txt \
     --output /new/plan
   ```

   This writes `query_plan.csv`, `query_plan.json`, `probes_bohr.dat` and
   `input.dat`. Explicit normalized directions can instead be supplied with
   `--directions-json` (a JSON list of vectors). Centers precede samples.
   `query_index` is one-based, matching native `properties.dat`. The plan
   stores absolute point coordinates and u separately and verifies
   `sample_point=center+u`. Avoid classical source coincidences. A finite
   set of points cannot establish that the intervening region is nonsingular.

2. In a new run directory, run the selected fully linked CPU or CUDA
   benchmark client on `input.dat`. For backend comparisons, replace the
   density on both backends with the same exported alpha/beta density file:

   ```sh
   /selected/benchmark-cuda /new/plan/input.dat /retained/density.dat 1
   ```

   This native client performs an initial SCF before replacing its density.
   Its reported initial SCF energy is not the energy of a replacement
   density. Record client/source/loaded-library hashes and the exact model,
   keyword card, basis and retained density. The run exports `properties.dat`,
   `ao_basis.dat`, `density.dat`, `overlap.dat`, `result.dat` and timings.
   The converter requires a converged initial SCF, zero extra API calls, and
   exactly unchanged retained density and Fock state during property calls.

3. Compute electron number density and its gradient independently from
   Cartesian AO values and derivatives. Use the exact QUICK Gaussian basis
   file, exported contracted-AO descriptors and density; this helper maps AO
   order/phases and checks the normalized overlap matrix within 2e-10.
   No SCF is performed. PySCF, SciPy and NumPy are required for this step.

   ```sh
   /python/with/pyscf validation/field_uniformity/ao_density_oracle.py \
     --export-dir /native/run --model model.json \
     --basis-file /selected/quick/basis/EXACT.BAS --centers centers.csv \
     --output /new/ao-reference
   ```

   `density.dat` contains the restricted total density or alpha followed by
   beta, flattened in Fortran order. The oracle sums UHF spin components.
   n and grad(n) come from `phi D phi` and its Cartesian AO derivative,
   never from a trace of G or H. `density_reference.csv` retains center
   coordinates as well as n/grad(n); JSON retains basis/density/model/AO
   hashes, normalization checks and AO mappings.

4. Convert the existing native export format and analyze supplied samples:

   ```sh
   python3 validation/field_uniformity/convert_api_export.py \
     --query-plan /new/plan/query_plan.csv --export-dir /native/run \
     --native-input /new/plan/input.dat --model model.json \
     --density-reference /new/ao-reference --backend cuda \
     --output /new/analysis-input
   python3 validation/field_uniformity/field_uniformity.py \
     --manifest /new/analysis-input/manifest.json --output /new/analysis \
     --tolerance-au 1e-4 1e-3 --plot
   ```

   Native rows have 44 columns: one-based index, xyz, V, E3, G9 and H27;
   tensors flatten in Fortran order. Full G/H permutation discrepancies
   are measured before packing. Non-symmetric tensors are rejected
   (`rtol=1e-12, atol=1e-14`); they are never silently symmetrized. The
   adapter verifies source geometry/MM coordinates, point ordering and
   actual frozen-state assertions as well as AO-reference fingerprints.

## Input schema and outputs

The manifest schema is `quick-field-uniformity-input-1`, with
`input_kind=quick-frozen-density` for native data or `analytic-validation`
for explicitly mathematical test fixtures. `source_convention` must be
exactly
`electrons+nuclei+enabled_mm; direct_applied_and_reaction_backgrounds_excluded`.
`units` must match the dictionary in `field_uniformity.py`. Native manifests
require SHA256 density, model/geometry and exact basis fingerprints, and a
literal true assertion of identical density and geometry for all queries.

`centers_csv` points to a table with these columns:

```text
center_id,x_bohr,y_bohr,z_bohr,e_x,e_y,e_z,
g_xx,g_xy,g_xz,g_yy,g_yz,g_zz,
h_xxx,h_xxy,h_xxz,h_xyy,h_xyz,h_xzz,h_yyy,h_yyz,h_yzz,h_zzz,
n_e,grad_n_x,grad_n_y,grad_n_z
```

`samples_csv` points to:

```text
center_id,sample_id,direction_id,radius_bohr,
u_x_bohr,u_y_bohr,u_z_bohr,e_direct_x,e_direct_y,e_direct_z
```

Sample IDs are unique per center. A direction ID has the same direction at
every supplied radius. The displacement norm must match `radius_bohr`.
Actual CSV headers occupy a single line. The converter emits both files.

Analysis exports `centers_analysis.csv` (invariant norms, trace/STF parts,
Poisson checks), `sample_errors.csv` (all three vector predictions and
absolute field-error norms), `radius_summary.csv` (maximum, RMS, median and
95th-percentile errors over supplied samples), `analysis.json` (provenance
and empirical acceptance), and optionally a standalone `field_error.pdf`.
The plot reports the maximum over supplied samples; exact zeros are
explicitly omitted from logarithmic curves without an artificial floor.

The optional tolerances are absolute field errors chosen for an application.
The largest accepted radius is the largest **tested prefix radius** for
which every supplied sample at that and all smaller tested radii passes.
It is not a continuous bound on a sphere, unsampled directions/radii, an
evolving trajectory, or a relaxed electronic response. Taylor accuracy need
not improve monotonically with polynomial order at every finite radius.
No novelty or sampling claim follows from these offline calculations.

## Source and symmetric trace-free (STF) decomposition

Away from classical point sources, positive electron number density obeys

\[
 \operatorname{tr}G=-4\pi n_e,\qquad t_i=H_{ijj}=-4\pi\partial_i n_e.
\]

The trace/source components are physical and retained:

\[
 G^{src}_{ij}=\tfrac13\delta_{ij}\operatorname{tr}G,\qquad
 H^{src}_{ijk}=\tfrac15(\delta_{ij}t_k+\delta_{ik}t_j+\delta_{jk}t_i).
\]

Subtracting these gives exact STF tensors, with invariant identities

\[
 \|G\|_F^2=\|G^{STF}\|_F^2+(\operatorname{tr}G)^2/3,\quad
 \|H\|_F^2=\|H^{STF}\|_F^2+3\|t\|^2/5.
\]

The G packed multiplicities are `(1,2,2,1,2,1)` and H multiplicities are
`(1,3,3,3,6,3,1,3,3,1)` in the component order above. Naively taking the
Euclidean norm of the packed entries omits off-diagonal permutations.
STF/source tensors are orthogonal in tensor norm; their contracted field
increments along an individual u need not be orthogonal.

## Small, independent mathematical validation

```sh
python3 validation/field_uniformity/validate_math.py --output /new/math
python3 validation/field_uniformity/validate_adapters.py --output /new/adapters
/python/with/pyscf validation/field_uniformity/validate_ao_density.py \
  --output /new/ao-math
```

These use analytic static fields, synthetic file-format labels and primitive
AO polynomials only. They perform zero QUICK calls, SCFs and GPU launches.
They check rotations, multiplicities, STF norms, analytic Poisson signs,
finite-difference derivatives, Taylor orders 1/2/3, Fortran tensor ordering,
source/provenance rejection, and independently normalized Cartesian s/p/d/f
AO density/derivatives including spin summation and AO permutations/phases.

A normalized electron Gaussian centered at C has H(C)=0 but nonzero G(C).
Its linear and quadratic predictions coincide, yet have nonzero finite-radius
error. The explicit counterexample prevents using a zero center H as a
finite-radius certificate. No test fixture represents a generated trajectory.

Current preparation evidence (2026-10-15) is retained under
`/home/epalos/software/quick_ep/adaptive-electrostatics-20261015/field-uniformity/`:
`math-v2/report.json` PASS451, `adapters-v1/report.json` PASS121 and
`ao-math-v1/report.json` PASS11. These mathematical preparation results do
not establish native molecular accuracy.

## Static geometries and selected actual frames

`application.py prepare` reads a manifest with
`schema=quick-field-uniformity-application-1`. It writes native inputs and
an immutable case index; it launches no molecular calculations. The manifest
contains explicit protocol controls, positive radii, a Fibonacci direction
count, optional additional unit vectors, and cases:

```json
{
  "schema": "quick-field-uniformity-application-1",
  "protocol": {
    "keyword": "PBE0 BASIS=STO-3G ENERGY CHARGE=0 MULT=1 SCF=400 DENSERMS=1e-10 CUTOFF=1e-12 MATRIXZERO=1e-14 XCCUTOFF=1e-12 XC_RADIAL=100 XC_ANGULAR=434",
    "expected_controls": {"PBE0": null, "XC_RADIAL": 100, "XC_ANGULAR": 434, "MATRIXZERO": 1e-14},
    "basis_file": "/selected/quick/basis/STO-3G.BAS"
  },
  "radii_bohr": [0.005, 0.01, 0.02, 0.04, 0.08, 0.16, 0.32],
  "fibonacci_directions": 32,
  "cases": [{
    "id": "water",
    "geometry_xyz": "actual-geometry.xyz", "geometry_units": "angstrom", "frame_index": 0,
    "probes": "centers.probe", "probe_units": "bohr",
    "basis": "STO-3G", "charge": 0, "spin": 0
  }]
}
```

`expected_controls` is supplied by the caller rather than a universal
method requirement. A null value means a required bare keyword; numbers and
strings require matching explicit values. Each case can provide its own
`keyword` and `basis_file`. `frame_index` is zero-based in an existing XYZ
source; identities/order must agree across its frames. Optional
`atom_indices_1based` and `probe_indices_1based` retain explicit subsets.
Probe files have an integer count followed by triples; their source units
must be explicit. Preparation retains source hashes, actual frame/comment,
atom/probe selections, input/model/plan hashes and exact basis identity.
An old trajectory can supply actual coordinates, but its density is never
inferred: recompute or explicitly identify D for the exact chosen protocol.

```sh
python3 validation/field_uniformity/application.py prepare \
  --manifest application.json --output /new/prepared
```

After separate native collection, supply a JSON mapping such as
`{"water":{"cuda":"/actual/cuda/export","cpu":"/actual/cpu/export"}}`.
The analysis command verifies prepared inputs have not changed and runs the
independent AO evaluator, converter and CSV/plot analysis for each export:

```sh
python3 validation/field_uniformity/application.py analyze \
  --prepared /new/prepared --runs-json actual-exports.json \
  --oracle-python /python/with/pyscf --output /new/application-analysis \
  --tolerance-au 1e-4 1e-3 --plot
```

`validate_native_frozen_case.py` is the optional two-backend collector for a
selected small case. It first obtains one GPU retained density, then supplies
that exact export to the CPU client. Both clients necessarily perform one
setup SCF; property calls must perform no additional SCF. The collector
checks native CUDA E/G/H launch counters, exact retained-D export identity,
same-D V/E/G/H within 2e-10, independent Poisson traces within 2e-8, electron
count within 1e-8, and unchanged inputs, source and loaded runtime files. It
records observed Taylor slopes without making them a general guarantee.

The first actual molecular campaign is
`field-uniformity/native-small-water-v1/report.json` PASS32: gas-phase
PBE0/STO-3G water on the 100x434 grid, three centers spanning electron
densities 0.487, 0.333 and 0.000462 bohr^-3, 32 directions at seven radii
0.005–0.32 bohr (675 native query points). Same-D CPU/CUDA E/G/H discrepancies
are at most 5.69e-14; independent Poisson residuals are at most 1.91e-14.
Observed small-radius error orders are 1.005–1.013, 1.997–2.009 and
3.004–3.009 for constant, linear and quadratic reconstructions. At the outer
center and 0.32 bohr, the observed maximum errors are 0.009365, 0.002298 and
0.0004473 a.u. This is a finite frozen-density demonstration for one small
molecule, with no trajectory statistics or continuous bound.

`validate_application_plan.py --basis-file /exact/STO-3G.BAS --output /new/plan-tests`
checks actual WATER27 geometry/probe selection and prepared-input mutation
rejection without launching chemistry. Its current report
`application-plan-math-v2/report.json` passes 20 checks. The prepared
WATER20/DEF2-SVP cases in **`application-plan-v4`** use explicit 150x590
quadrature, `BASISCUTOFF=1e-12`, bare `D3`, and optional Fx=0.0002 a.u.
Use the one-pass collector documented in
`validation/electrostatics_benchmark/SNAPSHOT_COLLECTOR.md`; it checks native
printed controls rather than assuming the parser used every input token.
These plans alone are not results or evidence of larger-system transfer.
Earlier v1/v2 plans used an ignored `DFTD3` token; v3 corrected it to `D3`
but still omitted the explicit basis cutoff. All were unexecuted and remain
preserved with correction records. The actual small-water campaign above
used its own explicitly recorded PBE0/STO-3G protocol without dispersion.
