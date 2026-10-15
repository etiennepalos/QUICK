# Standalone native partition-of-unity acceptance

`validate_native_pou.py` compiles an immutable copy of the production
`quick_partition_of_unity_module.f90` with bounds checking, then runs
`quick_native_pou_driver.f90`. It launches no QUICK job, SCF or GPU kernel.
NumPy is the only third-party Python dependency.

```sh
python3 validation/adaptive_electrostatics/validate_native_pou.py \
  --compiler /matching/toolchain/bin/gfortran \
  --repo /path/to/QUICK --workdir /new/native-pou-evidence
```

The retained source, compiler command, executable hash, complete inputs and
outputs, stdout/stderr and `report.json` identify exactly what was tested.
An existing evidence directory is refused. A failed gate stops the campaign
and remains in its report; acceptance gates are declared in the script.

The native routine interpolates an electronic scalar potential over one fixed
uniform cover. It returns `E=-grad(V)` and `G=grad(E)=-Hess(V)`; no classical,
applied or reaction background is added by this standalone module. Anchor
order is x, then y, then z fastest. Each support has halfwidth equal to its
leaf width. Scalar compact weights are `(1-t*t)**4` inside the support and
zero outside; all normalized-weight derivatives enter E and G. At most eight
patches are active. Quadratic and cubic local Taylor polynomials require
symmetric G and fully symmetric H anchors, respectively. Invalid tensors are
rejected rather than projected.

The validation uses three distinct checks:

* The existing independent vectorized `reference_adaptive.FixedCover`
  supplies V/E/G for random symmetric anchor jets. It uses the full support
  radius for its coarse budget. The native query-radius budget must be no
  larger; the two budgets are intentionally not required to be equal.
* A second-order scalar automatic-differentiation algebra enumerates every
  anchor, differentiates products and reciprocals, and supplies normalized
  weights and the native query-radius budget independently of its eight-patch
  candidate search and explicit derivative formula.
* A known quartic polynomial supplies exact jets and true V/E/G. Absolute
  monomial coefficient bounds enclose every third/fourth derivative over each
  **whole support box**. Their full Frobenius norms bound Taylor remainders.
  True interpolation errors must fit the returned budgets, including tests
  with explicit nonzero, known anchor perturbations and their supplied norm
  budgets. No sampled maximum is treated as a support majorant.

Cases cover noncubic boxes with 1, 2 and 4 leaves per axis, degrees 2 and 3,
corners/faces/edges, leaf and support boundaries and their one-sided limits,
and query counts 1/7/127/128/129/257. Frozen-cover central differences at two
steps check both derivative levels. Gauge tests use a moderate common shift
and an exactly representable large constant with exactly zero derivatives.
Negative tests cover dimensions, degree, leaf count, nonfinite data, negative
majorants/anchor budgets, uncovered points, G/H asymmetry and a late budget
overflow. All V/E/G/budget output sentinels must survive a rejected call.

The reference V/E/G gate is `3e-11 + 3e-12*max(abs(reference))`. Independent
budget agreement uses `2e-10 + 5e-12*max(abs(reference))`. Frozen-cover finite
differences use absolute E/G gates `2e-5` and `3e-4` plus a two-step stability
check; these diagnose differentiated interpolation, not QUICK integral
accuracy. Full actual discrepancies are retained for every gate.

The 2026-10-15 campaign `pou-native-validation-v2/report.json` passed 693
checks in 95 standalone native processes. Production source SHA-256 was
`2429d64195c09419e10398f7fca7221dc7ed5f35ce05bc112a63b890f17928fc`.
The preceding v1 failed a Python fixture shape before launching the native
runner and remains archived. `pou-native-counterexample-v1` used the already
guarded module, so it establishes rejected asymmetric inputs and unchanged
outputs; it does not establish execution of the earlier unguarded code.

These are real-arithmetic support/Taylor inequalities evaluated in FP64.
The total arithmetic/anchor error budget is unavailable when it has not been
supplied, rather than assumed zero. These tests establish standalone module
behavior; fully linked QUICK state, API, MPI, native CUDA integration and
performance require separate acceptance. They make no molecular or novelty
claim.
