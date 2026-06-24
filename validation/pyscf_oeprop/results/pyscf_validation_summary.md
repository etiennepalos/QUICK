# PySCF OEPROP Validation

Python: `/opt/anaconda3/bin/python`
PySCF: `2.6.2`
Finite-difference step for PySCF EFIELD/EFG: `5.0e-04` bohr

## SCF Energies

| Case | QUICK / Eh | PySCF / Eh | PySCF - QUICK / Eh |
|---|---:|---:|---:|
| external PBE0-D3BJ/cc-pVDZ without dispersion energy | -192.933965693000 | -192.933979544092 | -1.385e-05 |
| surface HF/STO-3G | -74.947863811000 | -74.947863789510 | 2.149e-08 |

## Property Metrics

| Comparison | Values | Max abs. | RMS | Tolerance | Pass |
|---|---:|---:|---:|---:|:---:|
| external ESP_GRID acetone PBE0-D3BJ/cc-pVDZ | 663 | 4.110e-06 | 8.528e-07 | 1.0e-05 | yes |
| external EFIELD_GRID acetone PBE0-D3BJ/cc-pVDZ | 1989 | 2.400e-06 | 2.974e-07 | 5.0e-06 | yes |
| external EFG_GRID acetone PBE0-D3BJ/cc-pVDZ | 5967 | 1.510e-06 | 1.689e-07 | 1.0e-05 | yes |
| surface ESP_SURFACE H2O RHF/STO-3G | 62 | 2.582e-07 | 1.438e-07 | 5.0e-06 | yes |
| surface EFIELD_SURFACE H2O RHF/STO-3G | 186 | 3.123e-08 | 1.108e-08 | 5.0e-06 | yes |
| surface EFG_SURFACE H2O RHF/STO-3G | 558 | 4.114e-08 | 1.136e-08 | 1.0e-05 | yes |
| finite field H2O RHF/STO-3G +Fx energy | 1 | 2.158e-08 | 2.158e-08 | 1.0e-06 | yes |

## Interpretation Notes

The acetone case is a cross-code PBE0-D3BJ/cc-pVDZ comparison. The local PySCF installation does not include a D3 backend, so the PySCF reference uses the PBE0 SCF density and the QUICK energy comparison uses QUICK's printed total energy after subtracting the printed D3BJ dispersion correction. This is appropriate for ESP, EFIELD, and EFG because the D3BJ correction is a geometry-dependent post-SCF energy term and does not alter the one-particle density. The remaining differences are interpreted as cross-code DFT quadrature and SCF-density differences. The H2O RHF/STO-3G surface and finite-field checks are much stricter because the underlying SCF densities are nearly identical.

## Finite External Field

| Quantity | QUICK | PySCF | PySCF - QUICK |
|---|---:|---:|---:|
| E(+Fx) / Eh | -74.947913548000 | -74.947913526423 | 2.158e-08 |
| E(-Fx) / Eh | -74.947814117000 | -74.947814095202 | 2.180e-08 |
| E(0) / Eh | -74.947863811000 | -74.947863789510 | 2.149e-08 |
| central dE/dFx / a.u. | -0.497155000048 | -0.497156103094 | -1.103e-06 |
