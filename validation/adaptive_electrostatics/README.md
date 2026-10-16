# Experimental adaptive electrostatics

`getQuickAdaptiveElectrostatics` is an opt-in library API that approximates the
electronic scalar potential and differentiates that same representation for
V/E/G. The existing `getQuickOEPROP` and `getQuickFieldHessian` APIs retain their
direct implementations and defaults. No new SCF is performed by this API.

```fortran
use quick_api_module, only: getQuickAdaptiveElectrostatics, &
  adaptive_electrostatics_options, adaptive_electrostatics_report
type(adaptive_electrostatics_options) :: options
type(adaptive_electrostatics_report) :: report
! Establish a converged QUICK job first; points(3,np) are in bohr.
! v(np), e(3,np), g(3,3,np) must already be allocated.
options%tolerances = [1d-5, 1d-4, 1d-3]
! Optional: explicitly request native CUDA query evaluation on a CUDA build.
options%gpu_queries = .true.
call getQuickAdaptiveElectrostatics(np, points, options, v, e, g, report, ierr)
```

The default query backend is CPU. Anchors and support majorants use the new
CPU full-density integral implementation on both backends. CUDA requests on
unsupported builds fail with error 56, without silently running CPU queries.
CPU/MPI requests are collective on the QUICK job communicator, with identical
options and points required on every rank. CPU preparation uses the complete
spin-summed density on the master and broadcasts results; split communicators
are tested. Adaptive CUDA/MPI and HIP execution remain unverified.

V includes electrons, nuclei and enabled MM charges. Only the electronic
potential is approximated; classical sources are evaluated directly. Direct
applied-field and explicit continuum-reaction backgrounds are excluded, while
their effect on the converged density may be present. E is `-grad(V)` and G is
`grad(E)`. Coordinates are bohr and output units are Eh/e, Eh/(e bohr) and
Eh/(e bohr^2). H remains a separate, direct API observable.

Each call rebuilds all density-dependent anchors and bounds. There is no
electronic cache to reuse across changed densities or BOMD frames. Uniform
cover levels are refined according to the requested error budget; the current
implementation is not a locally refined octree. It uses compact C3 weights,
at most eight overlapping patches and all normalized-weight derivative terms.

The CPU Coulomb recurrence processes only the active degree-ordered Cartesian
prefix at each layer, retaining explicit zero output tails. Boys functions,
contractions, support bounds, status counters and resource controls retain
their arithmetic. The isolated 64-AO water raw-controller pilot reduced the
median from 6.557 s to 2.652 s (2.47x); it excludes SCF, classical sources
and the public API wrapper. The final v3 linked CPU/MPI/default-ddX-OFF suite
passes 521/521 checks, including unchanged state/direct H and no extra SCFs.

Matched complete-API trials used that same frozen HF/Cartesian def2-TZVPD
water density and a small exterior cube of halfwidth 0.025 bohr. Three
independent baseline/candidate pairs ran in AB/BA/AB order, with each first
call excluded. At 100,000 probes with CUDA queries, adaptive API medians
were 6.653 s and 2.554 s (median paired ratio 2.60x), including CPU
anchor/bound preparation, transfers, querying and classical sources. At
1,000 CPU probes, medians were 6.594 s and 2.467 s (2.67x); direct CPU
V/E/G remained slightly faster at 2.357 s. Adaptive outputs, budgets,
counters and retained density/AO data were bitwise identical. The GPU
direct comparator differed by at most 4.45e-15. Initial SCF and query-point
direct H are outside these property clocks; H anchors are inside preparation.
The original numerical limits and unknown floating-point budget are
unchanged. Receipts are under
`quick-optimization-20261015/adaptive-paired-{gpu100k,cpu1k}-v3/report.json`;
these are single-density local-query results, not a general or BOMD speedup.

The default `analytic_majorant` mode evaluates whole-support Taylor
inequalities in ordinary FP64. These are truncation budgets, with unknown
anchor/integral/accumulation error. `report%total_error_budget` remains NaN,
`numerical_budget_known` is false and `certified` is false. The `indicator`
mode is explicitly heuristic. Requesting `certified` returns error 59;
invalid controls return 58. No Poisson trace projection is applied to G.

Preparation, memory, depth, anchor and overlap limits select a whole-batch
direct fallback with `ierr=0` and `report%fallback=.true.`. Invalid inputs and
backend failures return an error with caller outputs unchanged. A fixed box
that does not cover a finite query batch also selects direct fallback.
Preparation time limits are cooperative between batches. The memory limit
uses a conservative work estimate for route selection, rather than imposing a
hard process-RSS or preallocation limit. Device-buffer counters exclude CUDA
context/driver storage and the separately reported kernel stack reservation.

`fixed_domain=.true.` with `lower` and `upper` fixes the domain, but the
controller may select a different level for a different query batch. Finite
differences of a single interpolant must keep the density, selected cover,
anchors and bounds fixed; use the raw PoU evaluator for those tests. A fixed
domain by itself does not guarantee a fixed interpolant across API calls.

Validation and retained evidence are described in:

- [DESIGN.txt](DESIGN.txt): integral recurrence, support bounds, controller,
  CPU/MPI acceptance, resource-limited pilot and negative cost results.
- [NATIVE_POU.md](NATIVE_POU.md): independent scalar differentiation, tensor
  symmetry, gauge, support-boundary and transactional CPU checks.
- [cuda_query_validation.md](cuda_query_validation.md): executed CUDA query
  checks, all four sanitizer modes and kernel resources.
- [ADAPTIVE_BOMD.md](ADAPTIVE_BOMD.md): property-only trajectory integration,
  direct-H comparisons and actual-backend telemetry.
- [LITERATURE.txt](LITERATURE.txt): established ingredients and the limited
  scope of the initial literature comparison.

`benchmark_adaptive.py` compares complete native V/E/G and adaptive API calls
using one exact retained density and Cartesian AO layout. It includes CPU
anchor/bound work, query evaluation, transfers and fallback, while excluding
the initial SCF from property clocks. Its direct baseline uses three existing
property routes, not a newly fused kernel. Loose settings are implementation
controls and cannot supply a matched-accuracy speed claim. `plot_adaptive.py`
exports maximum/RMS errors, budgets and complete costs from completed reports
to CSV and PNG/PDF without launching QUICK. These scripts are experimental
validation tools, not a default property selection or a production performance
promise.
