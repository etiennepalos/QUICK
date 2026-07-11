# QUICK-MBX CPU Validation

The analytic-gradient probes exercise the corrected stationary QUICK-MBX
energy, including cross 2B, cross 3B, and dispersion terms. MBX is an external
dependency and is not vendored.

Configure a serial build with the MBX library and public headers:

```bash
cmake -S . -B /private/tmp/quick-mbx-build \
  -DCOMPILER=AUTO -DMBX=TRUE \
  -DMBX_LIBRARY=/path/to/libmbx.a \
  -DMBX_INCLUDE_DIR=/path/to/MBX/src \
  -DMBX_EXTRA_INCLUDE_DIRS=/path/to/fftw/include \
  -DMBX_EXTRA_LIBRARIES=/path/to/libfftw3 \
  -DCMAKE_C_COMPILER=/opt/homebrew/bin/gcc-15 \
  -DCMAKE_CXX_COMPILER=/opt/homebrew/bin/g++-15 \
  -DCMAKE_Fortran_COMPILER=/opt/homebrew/bin/gfortran
cmake --build /private/tmp/quick-mbx-build --parallel 8 \
  --target test-api-mbx-analytic-gradient test-api-mbx-analytic-md-smoke
```

Run the four-case force probe:

```bash
export QUICK_BASIS=$PWD/basis
/private/tmp/quick-mbx-build/src/test-api-mbx-analytic-gradient \
  /private/tmp/quick-mbx-gradients.csv /path/to/mbx.json all hf cross
```

Arguments after the CSV are MBX JSON, case (`all`, `A_QM_B_MBX`,
`B_QM_A_MBX`, `A_QM_BC_MBX`, or `A_QM_B_MBPBE`), QM method (`hf`,
`pbe0-d3`, or `pbe-d3`), and residual mode (`cross` or `raw`). Coordinates
passed through the API are Angstrom; finite-difference steps are bohr; energy
and gradients are hartree and hartree/bohr.

QUICK checks the JSON path before entering MBX. A missing file returns error
55 collectively in serial or MPI instead of allowing MBX to use defaults.

Run the corrected analytic-force NVE smoke trajectory:

```bash
/private/tmp/quick-mbx-build/src/test-api-mbx-analytic-md-smoke \
  /private/tmp/quick-mbx-md.csv \
  /private/tmp/quick-mbx-md.xyz \
  /private/tmp/quick-mbx-md-summary.json \
  /path/to/mbx.json 20 0.05 \
  /private/tmp/quick-mbx-md-oeprop.csv
```

The final file contains on-the-fly QM ESP, EFIELD, and EFG at the moving MB
water atoms after every accepted analytic-gradient step. QUICK uses
`E_i=-dV/dC_i` and `G_ij=dE_i/dC_j`.

For MPI, configure with `-DMPI=TRUE`, build
`test-api-mbx-analytic-gradient.MPI`, and run two ranks. The MBX calls occur on
rank 0; QUICK distributes AO shell pairs and broadcasts/reduces contiguous
property and force buffers.

The tested MBX snapshot needs three isolated upstream-quality fixes:

- correct virtual-site gradient redistribution for multiple monomers;
- accept the public `h2o-pbe` identifier in MB-PBE constructors;
- disable the source-level `PRINT_INDIVIDUAL_TERMS` diagnostic so library
  energy calls do not write to `stderr`.

The first two affect correctness. The third makes the no-I/O coupling loop
usable; it does not change energies or forces.
