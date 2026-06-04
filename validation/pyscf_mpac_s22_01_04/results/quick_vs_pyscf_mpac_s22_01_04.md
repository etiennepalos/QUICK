# QUICK MPAC/HFAC S22 01-04 Validation

Reference: PySCF conventional RHF + canonical MP2, STO-3G, no frozen core, no density fitting.

| quantity | max abs diff / Eh |
|---|---:|
| `e_hf` | 1.567123e-07 |
| `e_mp2_os` | 3.724239e-08 |
| `e_mp2_ss` | 1.870988e-08 |
| `e_mp2_corr` | 5.595127e-08 |
| `e_x_hf` | 9.312557e-07 |
| `pc` | 2.911624e-04 |
| `w_inf` | 2.718406e-04 |
| `w_half` | 1.217491e-03 |
| `w_three_quarter` | 5.496236e-06 |
| `e_spl2_corr` | 7.743177e-07 |
| `e_spl2_total` | 6.175914e-07 |
| `e_os_spl2_corr` | 7.964650e-06 |
| `e_os_spl2_total` | 7.807922e-06 |
| `e_mpac25_corr` | 1.444709e-06 |
| `e_mpac25_total` | 1.287982e-06 |
| `e_os_mpac25_corr` | 1.440023e-06 |
| `e_os_mpac25_total` | 1.283297e-06 |
| `e_hfac24_corr` | 1.193136e-08 |
| `e_hfac24_total` | 1.473147e-07 |

Full per-system values are in the CSV and JSON files.
