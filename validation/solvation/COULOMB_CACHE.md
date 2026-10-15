# CUDA Coulomb-cache regression and controlled profiling

Both standalone clients link to the actual shared CUDA QUICK library and
its matching exported Fortran modules. They do not override production
Coulomb objects or modules. Use the compiler matching that build; no CPU,
HIP, or MPI execution is claimed by these serial CUDA clients.

Example regression compilation:

```sh
FC=/path/to/matching/gfortran
"$FC" -cpp -DCUDA -ffree-line-length-none -fopenmp -O2 \
    -I /path/to/QUICK/src/util \
    -I /path/to/build/amber-modules/quick/cuda \
    validation/solvation/test_gpu_solvation_coulomb_cache.f90 \
    -L /path/to/build/src -Wl,-rpath,/path/to/build/src -lquick_cuda \
    -L /path/to/ddx/lib -Wl,-rpath,/path/to/ddx/lib -lddx \
    -o /path/to/test-gpu-coulomb-cache
/path/to/test-gpu-coulomb-cache
```

The synthetic Cartesian s/p/d/f cases compare CPU and native GPU matrices
and gradients within `1e-11`. They exercise changes in reaction charges and
density without rebuilding the geometry operator, inactive-primitive reuse,
and invalidation by active coefficients/exponents, primitive counts, angular
powers, AO-center mapping, nuclear geometry, and cavity coordinates/count.
Empty cavities, repeated cleanup, preserved versus reset statistics, a
second job, and matrix symmetry/gradient translation are checked. A one-byte
memory cap must take the direct GPU fallback and preserve agreement; the
test also restores caching and checks rejected negative limits.

When available, memory checks can run on the same executable:

```sh
compute-sanitizer --tool memcheck --leak-check full --error-exitcode 99 \
    /path/to/test-gpu-coulomb-cache
compute-sanitizer --tool initcheck --error-exitcode 99 \
    /path/to/test-gpu-coulomb-cache
```

These checks use the current device. They do not cover cross-device
invalidation or induce physical `cudaMalloc` exhaustion; the explicit tiny
cap is a deterministic fallback regression.

Compile the benchmark separately against the same frozen CUDA library and
matching Fortran modules. It requires ddX-enabled QUICK and ddX module files:

```sh
"$FC" -cpp -DCUDA -ffree-line-length-none -fopenmp -O2 \
    -I /path/to/QUICK/src/util \
    -I /path/to/build/amber-modules/quick/cuda \
    -I /path/to/ddx/include/ddX \
    validation/solvation/benchmark_solvation_coulomb_cache.f90 \
    -L /path/to/build/src -Wl,-rpath,/path/to/build/src -lquick_cuda \
    -L /path/to/ddx/lib -Wl,-rpath,/path/to/ddx/lib -lddx \
    -o /path/to/benchmark-solvation-coulomb-cache
```

Stage a fresh profiling directory:

```sh
mkdir /path/to/new-coulomb-profile
cp validation/solvation/fixtures/water27_h2o20es.xyz \
    /path/to/new-coulomb-profile/geometry.xyz
cp validation/solvation/coulomb_cache_benchmark_pcm.nml \
    /path/to/new-coulomb-profile/benchmark.nml
cd /path/to/new-coulomb-profile
QUICK_BASIS=/path/to/QUICK/basis /path/to/benchmark-solvation-coulomb-cache \
    benchmark.nml
```

The example retains the controlled production settings: full ddPCM cavity,
`lmax=6`, 302-point cavity grid, `75x302` volume grid, radius scale 1.1,
dielectric 78.3553, and an 8-GiB requested cache cap. The production cache
still applies its actual free-device-memory limit. The client's default
gas-phase PBE-D3/def2-SVP SCF supplies a real fixed QM density. Native ddX
produces the cavity reaction charges from that density. Cached and direct
GPU matrix contractions then use exactly the same geometry, basis, density,
and charges in one process; no second SCF or ddX solve changes the control.
The client requires cached/direct agreement within `1e-10`, records cache
statistics, and separately times forward ESP and direct analytic gradients.

The benchmark is a fixed-density operator profile, not a self-consistent
solvent trajectory or a scientific force-acceptance campaign. It writes
timing CSV and metadata with `status='new'`; use a fresh directory/prefix.
Run with exclusive device access for performance comparisons and preserve
the input, source/build identity, loaded-library resolution, GPU identity,
raw timing CSV, and metadata. Overlapping correctness jobs cannot support
whole-run speedup claims. `controlled_direct=.false.` permits a deliberately
uncached configuration; that is a separate experiment from the paired
cache/direct control. No machine-specific launcher is supplied.
