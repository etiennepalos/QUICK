# Applied electric fields and equilibrium continuum solvation

This feature adds analytic nuclear forces for static applied electric fields,
including a controlled spatial gradient, and an optional continuous-density interface to ddX's electrostatic
ddCOSMO and ddPCM models. They can be used through the QUICK energy/gradient
API and the existing velocity-Verlet BOMD driver, together with ESP, EFIELD
and EFG sampling. These are finite-molecule models.

The integration is default-disabled. Ordinary gas-phase SCF calculations and
the established ESP/MKS/RESP interfaces retain their existing methods and
settings. Validation results and the exact tested build identities belong in
a separate acceptance report; the examples below describe the interface.

## Uniform applied fields

Add any nonzero Cartesian component, in atomic units:

```text
HF BASIS=STO-3G GRADIENT EXTERNAL_EFIELD_X=0.0002
```

`EXTERNAL_EFIELD_Y` and `EXTERNAL_EFIELD_Z` set the other components.
`FINITE_FIELD_X/Y/Z` are aliases. The default field origin is zero;
`EXTERNAL_EFIELD_ORIGIN_X/Y/Z` (or `FINITE_FIELD_ORIGIN_X/Y/Z`) sets it in
bohr. For field vector F and origin r0, the added terms are

```text
h(mu,nu) = + F . <chi_mu | r-r0 | chi_nu>
E_classical = - sum_A q_A F . (R_A-r0).
```

Energies are in hartree and API nuclear gradients are +dE/dR in
hartree/bohr; mechanical forces are their negatives. ESP, EFIELD and EFG
use hartree/e, hartree/(e bohr), and hartree/(e bohr^2), respectively.
API atom/point-charge coordinates are in Angstrom; API property probes
and the applied-field origin are in bohr. For total charge Q, shifting
the origin by d changes the field energy by +Q F.d; translating the
whole system by d at fixed origin changes it by -Q F.d.

Analytic forces include the contracted Gaussian moment derivatives and the
classical forces on nuclei and supplied external point charges. Restricted
and unrestricted densities use the same operator. The ordinary SCF Pulay
terms are retained. CPU MPI distributes the same upper-triangular AO rows
as Hcore; the classical term occurs once before the normal gradient sum.

CUDA builds evaluate the field AO matrix and its electronic nuclear
derivatives in separate FP64 CUDA kernels. They use the normalized Cartesian
basis and the existing primitive Hcore cutoff, with independent buffers.
Unsupported angular momentum is rejected explicitly. This does not establish
whole-QUICK f-function support; use the angular-momentum coverage in the
acceptance report. HIP retains the Fortran field reference implementation.

A constant scalar potential, constant-potential electrode, time-dependent
field and periodic field boundary conditions are separate models and are
not provided by these keywords. The energy of a charged system depends on
the chosen origin; translating it in a fixed applied field changes the
energy by the corresponding classical total-charge term.

## Controlled spatial field gradients

Six independent components define a constant, symmetric spatial gradient
`G_ij = dE_i/dr_j`. The applied field varies linearly with position:

```text
s = r-r0
E_applied(r) = F + G*s
phi_applied(r) = -F.s - 0.5*s.G.s
h_applied(mu,nu) = <chi_mu | F.s + 0.5*s.G.s | chi_nu>
E_classical = -sum_A q_A * (F.s_A + 0.5*s_A.G.s_A)
dE_classical/dR_A = -q_A * (F + G*s_A)
```

The tensor is an electric-field gradient, **not** the potential Hessian and
not a nuclear energy gradient. Its units are hartree/(e bohr^2). The existing
field vector `F` is the field at the fixed origin `r0` (in bohr). Both
electronic and classical quadratic terms are included in the SCF energy and
analytic nuclear forces. Off-diagonal components occur once in the quadratic
polynomial: `Gxy*sx*sy`, not `Gxy*sx*sy/2`.

| Long keyword | Compact alias |
|---|---|
| `EXTERNAL_EFIELD_GRADIENT_XX` | `EFIELD_GXX` |
| `EXTERNAL_EFIELD_GRADIENT_YY` | `EFIELD_GYY` |
| `EXTERNAL_EFIELD_GRADIENT_ZZ` | `EFIELD_GZZ` |
| `EXTERNAL_EFIELD_GRADIENT_XY` | `EFIELD_GXY` |
| `EXTERNAL_EFIELD_GRADIENT_XZ` | `EFIELD_GXZ` |
| `EXTERNAL_EFIELD_GRADIENT_YZ` | `EFIELD_GYZ` |

Components default to zero. Each off-diagonal input sets both symmetric
entries; reversed `YX`, `ZX` and `ZY` keywords are rejected. Finite values
and exact symmetry are checked, including direct method-state requests.
Conflicting long/compact aliases are rejected. A symmetric gradient gives a
curl-free field. In a source-free region it must also satisfy `trace(G)=0`.
QUICK accepts a general symmetric tensor and prints its trace; a nonzero
trace represents an imposed charge density under Poisson's equation. No
automatic trace removal or symmetrization is performed.

Compact `EFIELD_FX/FY/FZ` and `EFIELD_OX/OY/OZ` aliases specify the existing
field vector and origin. They allow combined vector, tensor and solvent
requests within the public API's existing 256-character keyword limit.
For example, a traceless gradient with an off-diagonal component is:

```text
HF BASIS=DEF2-TZVPD GRADIENT EFIELD_FX=2D-4 EFIELD_GXX=1D-5 EFIELD_GYY=-1D-5 EFIELD_GXY=2D-5
```

A pure gradient (`F=0`) is supported. An exactly zero tensor retains the
established uniform-field evaluator. The nonzero-tensor route uses analytic
Gaussian second moments and their center derivatives; CUDA uses independent
FP64 kernels with Cartesian s/p/d/f support, while CPU is the moment-integral
reference. Existing angular-momentum restrictions in other QUICK methods
still apply. Numerical failures in the new operator are rejected explicitly.

The vector, tensor and origin remain fixed in the laboratory frame during
BOMD. No waveform or molecule-following gradient is implied. A neutral
molecule can have a nonzero net force in an inhomogeneous field, so a
zero-net-force criterion is inappropriate. Use weak perturbations suitable
for the molecular region and check basis/SCF convergence; this polynomial
potential does not model electrodes or periodic boundary conditions.

For an origin change `r0' = r0+d`, the *same physical field* requires
`F' = F+G*d`. This changes the potential by the constant
`c = F.d + 0.5*d.G.d`. The total energy changes by `Q_total*c`, while
forces are invariant. Moving the origin while holding `F` fixed changes
the physical field when `G` is nonzero.

The existing `DDX_FIELD=SOLUTE` convention also applies to spatial gradients:
the perturbation acts on the QM solute, and the continuum responds to that
solute. It does not impose a spatially varying macroscopic dielectric
boundary condition. The existing property-grid convention is retained:
direct applied contributions are excluded. To obtain them analytically at a
probe, use `phi_applied`, `F+G*s` and `G`, respectively; explicit continuum
reaction properties remain separate from the public grid outputs.

## Optional ddX dependency

The optional interface uses an installed ddX 1.0 CMake package. The existing
local clone can be built and installed into an isolated prefix, for example:

```bash
cmake -S /home/epalos/software/ddX -B /tmp/ddx-build \
  -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=ON \
  -DCMAKE_INSTALL_INCLUDEDIR=include -DCMAKE_INSTALL_PREFIX=/tmp/ddx-install
cmake --build /tmp/ddx-build --parallel 2
cmake --install /tmp/ddx-build

cmake -S . -B build-ddx -DCOMPILER=GNU -DENABLEF=FALSE \
  -DCUDA=TRUE -DQUICK_DDX=ON \
  -DCMAKE_INSTALL_PREFIX=/tmp/quick-ddx-install \
  -DddX_DIR=/tmp/ddx-install/lib/cmake/ddX
cmake --build build-ddx --parallel 4
cmake --install build-ddx
```

Select the compiler, BLAS/LAPACK and CUDA architecture for the workstation.
The explicit `CMAKE_INSTALL_INCLUDEDIR` accommodates the installed include
directory exported by the local ddX 1.0 checkout. This does not modify ddX's
source. Dynamic-library search paths must resolve both QUICK and ddX.
QUICK's exported CMake package records the optional dependency.
Set QUICK's final installation prefix when configuring, then install without a
prefix override so the executables resolve the intended installed libraries.

`QUICK_DDX` defaults to `OFF`. Requesting a solvent in an OFF build produces
an explicit error. The optional dependency is configured through CMake;
legacy make retains the ddX-disabled implementation. Existing CPU, MPI,
CUDA and CUDA MPI targets share the same interface. Runtime evidence for
HIP, multiple physical GPUs and other compilers must be assessed separately.

## Solvent controls

```text
DFT PBE0 BASIS=DEF2-SVP GRADIENT DDX=PCM DDX_EPS=78.3553
```

| Keyword | Default | Meaning |
|---|---|---|
| `DDX=COSMO` or `DDX=PCM` | disabled | Electrostatic continuum model |
| `DDX_MODEL` | same as `DDX` | Model-name alias |
| `DDX_EPS` | 78.3553 | Relative dielectric constant, at least 1 |
| `DDX_LMAX` | 6 | Maximum spherical-harmonic degree |
| `DDX_NLEBEDEV` | 302 | Angular cavity integration nodes per sphere |
| `DDX_NRAD` | 75 | Radial volume quadrature nodes per atom |
| `DDX_NANG` | 302 | Angular volume quadrature nodes per atom |
| `DDX_RADIUS_SCALE` | 1.1 | Multiplier on supported Bondi radii |
| `DDX_TOL` | 1e-10 | Native forward/adjoint iterative-solver tolerance |

The volume quadrature covers the continuous electronic density. No fitted
atomic charges replace that density. Nuclear multipoles are exact; electronic
multipoles use regular solid harmonics and a separate atom-centered volume
quadrature. The volume partition has analytic moving-parent derivatives.
It does not change QUICK's XC grids. Both the cavity discretization and
volume quadrature need convergence checks for the intended system.

Electronic projection uses B_lm^A(r) = [4 pi/(2l+1)]
(|r-R_A|/a_A)^l Y_lm on all atom-partitioned volume nodes, including
those outside their parent sphere. This is Psi4/ddX-style all-space regular-solid-harmonic
coupling. The stock PySCF ddCOSMO
outside-sphere density/leakage correction is absent. Reference comparisons
must match this projection, basis, cavity and volume grids, and the volume
partition and quadrature weights.

The COSMO finite-dielectric factor is f=(epsilon-1)/epsilon; PCM uses
factor 1 because its solve already includes epsilon. Native ddX options
are fixed to eta=0.1, shift=-1 for COSMO or 0 for PCM, switching=0,
kappa=0, incore=0, maxiter=200 and jacobi_ndiis=20. FMM is enabled with
pm=pl=max(8,DDX_LMAX). Native nproc=1 and the host OpenMP setting is
restored after calls. These are model/solver definitions, not input knobs.

At epsilon=1 the interface bypasses the solvent and recovers gas-phase
energies and forces. Supported Bondi radii are checked; missing radii, ghost
atoms, ECP calculations, coincident nuclei, unsupported models and incompatible requests fail
explicitly. External point-charge sources combined with ddX are not yet
implemented. Their use with a uniform field without ddX is supported.

The coupling adds its full-density reaction operator after QUICK's ordinary
incremental operator has been saved and reduced. It adds the variational
solvation energy once. Analytic gradients include cavity, density, AO-center,
moving-node and partition contributions. The retained final density is
refreshed for the force without another SCF.

CUDA evaluates volume AO quantities, volume reaction matrices and derivatives,
and the reverse Coulomb matrix and its derivatives. The ddX forward/adjoint
linear solves, cavity construction, volume-grid/partition setup and harmonic
projection run on the CPU. Classical applied-field charge forces also run
on the CPU. The host OpenMP setting is restored around native ddX calls.
New GPU allocations are independent of the published
ESP/XC/OEI buffers. Geometry caches must be invalidated on nuclear moves and
released before QUICK's GPU job is destroyed.

The reverse Coulomb operator caches its packed FP64 AO-pair integrals at a
fixed geometry and reuses them through cuBLAS matrix-vector products. The
cache budget is the smaller of 8 GiB and one quarter of free GPU memory,
including its persistent workspaces. A size or allocation limit selects the
direct CUDA operator. Basis, coordinates, cavity nodes and device identify
the cache; density and solvent-charge changes reuse the same integrals.
The public matrix-call counter counts successful evaluations; separate
build, GEMV, hit, direct and fallback counters identify the actual route.

Only electrostatic equilibrium solvation is included. There are no SMD,
cavitation, continuum/non-electrostatic solvent dispersion, repulsion,
nonequilibrium dielectric or explicit solvent dynamics terms. Molecular D3
requested through QUICK retains its existing energy/gradient contribution. Native ddX cavity derivative routines return the
positive energy gradient despite their upstream “forces” names; QUICK's
energy-gradient API retains dE/dR, and the BOMD integrator changes its sign.

## Applied fields with a continuum

The combination requires an explicit convention:

```text
HF BASIS=STO-3G GRADIENT DDX=PCM EXTERNAL_EFIELD_X=0.0002 DDX_FIELD=SOLUTE
```

`DDX_FIELD=SOLUTE` means that the applied perturbation acts on the QM solute.
The continuum responds self-consistently to its polarized electronic density
and nuclear charges. It does not impose a macroscopic applied-field boundary
condition on the dielectric or include an empty-cavity field energy. In
particular, this convention does not establish dielectric saturation or a
constant-potential electrochemical model. A combined request without the
explicit convention is rejected.

## BOMD and reported properties

Pass the same keywords in the existing `quick_bomd` namelist's `keywd`
string. QUICK's public API retains its 256-character keyword limit. The
driver samples ESP, EFIELD and EFG from the retained density at the requested
stride, including every accepted step when `oeprop_stride=1`. Sampling does
not introduce an SCF or alter the integrator.

The established OEPROP quantities describe the solute nuclei, any supplied
external point charges, and the current QM density, including polarization
caused by the applied field and solvent.
They exclude the applied background field and explicit continuum reaction
potential. EFG retains QUICK's convention, G_ij=dE_i/dC_j, the negative
potential Hessian. This distinction matters when comparing to a total
electrostatic field acting on an embedded probe.

With active ddX, internal solvent updates also call the established forward
ESP and EFIELD evaluators. Raw kernel counts therefore include both solvent
work and requested output sampling. They must be reconciled with explicit
SCF-cycle counts rather than interpreted as extra property SCFs. The solvent
driver summary records successful volume CUDA operations and master phase
wall times separately.

New validations live in `validation/external_field` and
`validation/solvation`. The established ddX-disabled OEPROP/BOMD acceptance
suite remains a separate regression control. Short NVE and restart tests
validate software behavior and conservative forces; production ensembles,
long-time drift, spectra and statistical convergence require suitable
trajectory lengths and scientific protocols.

## Accuracy and trajectory controls

The default cavity and volume grids are starting quadratures. Agreement
between CPU/CUDA, density derivatives and nuclear finite differences verifies
the discretized functional; system-specific grid/order convergence needs its
own checks. In the tested HF/STO-3G water/dimer PCM pair, L8/N590 to L10/N974
at volume 100x434 met 1e-5 hartree and 5e-5 hartree/bohr comparison gates. For
COSMO dimer, L10 to L14 at shared N5810 still changed energy by 4.57e-5 hartree
and the largest gradient component by 1.16e-4 hartree/bohr. No converged COSMO
dimer or universal default-grid accuracy is established by these tests.

The independent exact-basis CPU/CUDA field oracle retains eight absolute PBE0
gradient failures across zero/applied field, two displacement sizes and both
backends against a denser XC grid. A separate ten-geometry native SG1 diagnostic
explains the largest tested O-z discrepancy: matching the moving native grid
reduces about 2.69e-4 hartree/bohr to 4.40e-8 at h=0.0005 bohr. Other components
were not tested in that diagnostic; native bin/primitive screening was not
reproduced. The original failures and SG1 quadrature-convergence question remain.

The BOMD force consumer rejects unconverged SCF results before accepting a
step. For the tested finite-water fixtures, 0.05 fs or smaller is the validated
starting timestep; check a smaller step at equal physical time. Some 0.1 fs
tests exceeded the 1e-5 hartree short-drift gate. The ordinary PBE0/STO-3G
screening control also missed strict timestep-refinement and CPU/CUDA energy
gates. A separate control with XCCUTOFF=1D-12 and BASISCUTOFF=1D-12 passed
the 0.1/0.05/0.025 fs comparison. Use tight SCF/integral/DFT screening and
independently converge the XC grid, continuum grids and trajectory protocol.
These are short software/force checks, not production ensemble validation.

The [spatial-gradient acceptance record](spatial-gradient-acceptance-20261015.json)
pins the tested working-tree source and CPU/CUDA libraries, report hashes,
independent reference comparisons, retained development failures and scope
limits for the static affine-field extension.
