# Electric-field spatial Hessian validation

The public property API is

```fortran
call getQuickFieldHessian(npoints, points_bohr, h, ierr)
! points_bohr(3,npoints), h(3,3,3,npoints)
```

It returns `H(i,j,k)=d² E_i/(dC_j dC_k)=-d³ V/(dC_i dC_j dC_k)`, in
hartree/(e bohr³). The three indices are fully symmetric for this electrostatic
observable. This is an observation-coordinate derivative, distinct from a
nuclear energy Hessian. Away from classical singularities, contraction over
the final two indices satisfies `sum_j H(i,j,j)=-4*pi*d rho_e/dC_i` for positive
electron number density `rho_e`; the tensor is not generally traceless.

Sources match the existing ESP/EFIELD/EFG API: spin-summed electronic density,
nuclei, and supplied external point charges. Direct imposed and ddX reaction
backgrounds are omitted. The density can nevertheless reflect those
self-consistent perturbations. Querying this property does not change the
Hamiltonian, forces, density, or SCF count.

There is no new standalone QUICK input keyword. The BOMD validation driver
offers the opt-in namelist setting `field_hessian=.true.`; the default is false.
It uses the existing `oeprop_stride` and writes a separate
`<prefix>_field_hessian.csv`. Its ten unique components are ordered

```text
xxx xxy xxz xyy xyz xzz yyy yyz yzz zzz
```

The public API retains all 27 components. Invalid dimensions return 46;
missing live density returns 47; invalid numeric values, unsupported angular
momentum, or a classical source singularity return 55; backend/allocation/MPI
failure returns 56. Error returns preserve the caller's output.

`quick_field_hessian_api_driver.f90` provides two test modes. `api` converges
one SCF, optionally reads a supplied frozen density or `ZERO`, calls the public
API, and exports AO/basis/density metadata. `raw` bypasses SCF for synthetic
normalized Gaussian functions and calls the native electronic CPU/CUDA
evaluator. Raw contributions are additive; a triangular AO pair belongs to
the owner of its higher AO index, and the density's upper triangle determines
the symmetric contraction. The same input and semantic checks run on both
backends.

Compile each client with the compiler, module directory and library matching
its installed QUICK backend. The CUDA client requires `-DCUDA -DGPU`; MPI
clients additionally require `-DMPIV`, the MPI compiler and matching module/
library variants. The independent raw mode must use a fully linked QUICK
runtime for release acceptance. This workstation's exact compilation and
force-export instrumentation records are retained under the artifact root's
`clients/` directory.

Run the complete native campaign with NumPy and an independent PySCF/SciPy
Python environment:

```sh
python3 -B validation/field_hessian/validate_field_hessian.py \
  --cpu-driver /path/to/api-cpu --cuda-driver /path/to/api-cuda \
  --cpu-basis /path/to/cpu/install/basis \
  --cuda-basis /path/to/cuda/install/basis \
  --cpu-bomd-driver /path/to/bomd-cpu \
  --cuda-bomd-driver /path/to/bomd-cuda \
  --oracle-python /path/to/pyscf/bin/python \
  --workdir /path/to/NEW/evidence-directory
```

BOMD clients require the test-only full-force export used by the accepted
artifact clients; its complete diff is retained beside their build commands.
Use `--skip-bomd` when running only property/API/raw validation. Every evidence
directory must be new; the validator refuses to overwrite accepted evidence.

The fixed campaign covers HF water in STO-3G, Cartesian 6-31G** and 64-AO
Cartesian def2-TZVPD, unrestricted water cation, signed external charges,
zero density, 257-point grids and 17-point API chunks. CPU/CUDA comparisons use
the same exported density. Every Cartesian tensor permutation, all three
directions of frozen-density EFG finite differences at .001/.0005 bohr, repeated
calls, unchanged ESP/EFIELD/EFG and live state, and lifecycle/error returns are
checked. Raw cases cover normalized s Gaussians at several exponent/length
scales, center coincidence, every s/p/d/f AO pair, signed synthetic density,
ownership partitions, additive seeds, zero work, admitted small density
asymmetry, and invalid numeric/range inputs.

`reference_field_hessian.py` is independent of QUICK's recurrences. It uses
libcint `int1e_ipipiprinv` and `int1e_ipiprinvip` to form the eight bra/ket
product-rule terms in the third observation derivative of the attraction
integral. Exact installed Gaussian basis definitions, Cartesian normalization,
AO permutation and spin-summed density are checked explicitly. An analytic
normalized-s-Gaussian formula and displaced second-derivative integrals check
the oracle's sign and index ordering. Classical point-charge derivatives and
the Poisson/density-gradient contraction supply separate checks.

Fixed molecular gates are 2e-8 a.u. against the independent analytic oracle,
2e-10 for CPU/CUDA at identical density, 2e-6 for EFG finite differences, and
2e-11 for permutation/repeat/chunk and existing-property preservation checks.
Gaussian raw references use relative 2e-11 gates at each point, including
small and far-field values. Gates are retained in machine-readable reports.

The accepted main report is
`/home/epalos/software/quick_ep/field-hessian-20261015/validation-v1/report.json`:
674 checks passed, including 97 independent-oracle checks, 50 API/raw runs and
six short BOMD controls. The largest molecular analytic-oracle difference is
3.48e-14 a.u.; identical-density CPU/CUDA disagreement is 3.22e-15 a.u.
Four .025 fs steps sample five frames without extra SCFs. These are software
and force-consistency checks, not production trajectory statistics, long-time
stability, periodic systems, arbitrary angular momentum or other GPU hardware
validation. The artifact root also retains native memcheck and coupled
field/continuum boundary controls separately; their reports define their exact
accepted scope.
