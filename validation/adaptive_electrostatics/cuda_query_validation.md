# Native CUDA cover-query validation

`validate_cuda_pou.py` exercises the independent FP64 CUDA query backend for a
fixed electronic potential cover. It compares V, E, G and pointwise budgets with
the retained production CPU query, a separate vectorized implementation, and an
independent second-order automatic-differentiation weight/budget oracle. It runs
no SCFs. Anchors and whole-support derivative majorants are supplied inputs.
Classical electrostatics and directly evaluated H remain the caller's work.

Run with an exclusive GPU allocation and a NumPy-enabled Python environment:

```bash
CUDA_VISIBLE_DEVICES=0 taskset -c 30 "$PYTHON" -B \
  validation/adaptive_electrostatics/validate_cuda_pou.py \
  --gpu-library "$QUICK_CUDA_LIBRARY" \
  --cpu-driver "$RETAINED_NATIVE_POU_DRIVER" \
  --workdir "$NEW_OUTPUT_DIRECTORY"
```

The output directory must not exist. `report.json` records runtime/source hashes,
every check and raw call, native launch/allocation diagnostics, and any exception.
Each case retains its input/output arrays. The CPU client is built from
`quick_native_pou_driver.f90` and `quick_partition_of_unity_module.f90`; its
scientific query body must match the tested production body. Successful raw calls
overwrite all outputs transactionally. Errors preserve the scientific outputs.

The suite covers nside 1/2/4, polynomial degree 2/3, domain and support boundaries,
query counts 1/7/127/128/129/255/256/257, nonzero anchor errors, quartic polynomial
support bounds, two-step scalar/field finite differences, potential gauges,
invalid inputs following successful calls, late device-side numerical failure,
cleanup and subsequent recovery. Assumed-shape Fortran array validation belongs
to the public wrapper; the raw C interface receives explicit scalar dimensions.

Compute Sanitizer can prefix the same command with:

```bash
compute-sanitizer --tool memcheck --leak-check full --error-exitcode 99 \
  --log-file "$MEMCHECK_LOG" "$PYTHON" -B \
  validation/adaptive_electrostatics/validate_cuda_pou.py \
  --gpu-library "$QUICK_CUDA_LIBRARY" \
  --cpu-driver "$RETAINED_NATIVE_POU_DRIVER" \
  --workdir "$NEW_MEMCHECK_DIRECTORY"
```

Separate runs use `--tool initcheck`, `--tool racecheck` and `--tool synccheck`.
Each needs a new output directory. Race/synchronization diagnostics exercise one
independent-query kernel with no shared-memory communication. Sanitizer timings
are diagnostic evidence, without a performance claim.

## Accepted v8 result, 15 October 2026

The fully linked CUDA library has SHA256
`0b085ca0c4cf7c658f0b3aa38571614771e3d9383ce6b438ab01217ba9e1ad9e`.
The CUDA source has SHA256
`1f03deaba24da15af29877e113e7ba41fe136b1f4c2994b7525d760394808748`.
The retained CPU scientific body matches the frozen v8 body exactly.

The ordinary run and each of the four sanitizer runs pass 1,228 checks across
89 raw calls. Memcheck reports zero errors and zero bytes leaked in zero
allocations; initcheck and synccheck report zero errors; racecheck reports zero
hazards, zero errors and zero warnings.

| Comparison | Maximum absolute difference |
| --- | ---: |
| CPU/CUDA V | 1.333e-15 |
| CPU/CUDA E | 1.599e-14 |
| CPU/CUDA G | 3.127e-13 |
| CPU/CUDA budgets | 3.553e-14 |
| Independent AD point-budget oracle | 4.264e-14 |

Every raw call returns zero live explicitly owned device-buffer bytes. The
largest simultaneous explicit device-buffer allocation is 63,232 bytes in this
suite. For a successful call it is exactly
`8 * (19*npoints + (42 + 4*has_anchor_errors)*nside^3) + 8` bytes. Rejected calls
that do not allocate report zero last-call allocation, including after previous
successful calls. The 75 native kernel launches include the late numeric failure
and recovery; invalid host inputs do not launch.

The actual sm_120 v8 kernel reports 128 registers/thread, a 1,856-byte/thread
stack, zero shared memory, zero statically declared local bytes, and 1,072
constant bytes. The explicit buffer counters exclude CUDA context/driver memory
and the implicit kernel stack. Therefore these counters are a device-buffer
highwater, not a total device-memory certificate.

The retained evidence sits beside the worktrees under
`adaptive-electrostatics-20261015`:

- `adaptive-cuda-query-v3/report.json`: ordinary acceptance, CPU-body provenance
  and `kernel-resources.txt`.
- `adaptive-cuda-query-{memcheck,initcheck,racecheck,synccheck}-v1/report.json`:
  complete numerical results under each sanitizer; adjacent `.log` files hold
  sanitizer summaries.
- `adaptive-cuda-query-acceptance-v1/assessment.json` and `seal.json`: combined
  result, source/runtime identities and immutable evidence hashes.

This qualifies the query implementation on the tested CUDA architecture. It
does not establish molecular interpolation accuracy or an end-to-end speedup.
CPU anchor/majorant preparation, native direct-query cost, buffers and kernel
stack must remain visible in performance comparisons.
