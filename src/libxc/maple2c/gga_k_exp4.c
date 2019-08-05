/*    
  This file was generated automatically with ./scripts/maple2c.pl.   
  Do not edit this file directly as it can be overwritten!!   
   
  This Source Code Form is subject to the terms of the Mozilla Public   
  License, v. 2.0. If a copy of the MPL was not distributed with this   
  file, You can obtain one at http://mozilla.org/MPL/2.0/.   
   
  Maple version     : Maple 2016 (X86 64 LINUX)   
  Maple source      : .//maple/gga_k_exp4.mpl   
  Type of functional: work_gga_x   
*/   
   
#ifdef DEVICE   
__device__ void xc_gga_k_exp4_enhance   
  (const void *p,  xc_gga_work_x_t *r)   
#else   
void xc_gga_k_exp4_enhance   
  (const xc_func_type *p,  xc_gga_work_x_t *r)   
#endif   
{   
  double t1, t2, t3, t4, t6, t7, t10, t12;   
  double t15, t16, t19, t21, t24, t36, t39, t55;   
   
   
  t1 = M_CBRT6;   
  t2 = 0.31415926535897932385e1 * 0.31415926535897932385e1;   
  t3 = POW_1_3(t2);   
  t4 = t3 * t3;   
  t6 = t1 / t4;   
  t7 = r->x * r->x;   
  t10 = exp(-0.83254166666666666664e1 * t6 * t7);   
  t12 = t1 * t1;   
  t15 = t12 / t3 / t2;   
  t16 = t7 * t7;   
  t19 = exp(-0.75479166666666666666e-2 * t15 * t16);   
  r->f = 0.20788e1 - 0.8524e0 * t10 - 0.12264e1 * t19;   
   
  if(r->order < 1) return;   
   
  t21 = r->x * t10;   
  t24 = t7 * r->x;   
  r->dfdx = 0.14193170333333333333e2 * t6 * t21 + 0.37027059999999999999e-1 * t15 * t24 * t19;   
   
  if(r->order < 2) return;   
   
  t36 = t2 * t2;   
  t39 = t1 / t4 / t36;   
  r->d2fdx2 = 0.14193170333333333333e2 * t6 * t10 - 0.23632811369194444443e3 * t15 * t7 * t10 + 0.11108118000000000000e0 * t15 * t7 * t19 - 0.67074519189999999998e-2 * t39 * t16 * t7 * t19;   
   
  if(r->order < 3) return;   
   
  t55 = t16 * t16;   
  r->d3fdx3 = -0.70898434107583333329e3 * t15 * t21 + 0.24238353882341514359e3 * t24 * t10 + 0.22216236000000000000e0 * t15 * r->x * t19 - 0.60367067270999999999e-1 * t39 * t16 * r->x * t19 + 0.12805511338572122736e-6 * t55 * r->x * t19;   
   
  if(r->order < 4) return;   
   
   
}   
   
#ifndef DEVICE   
#define maple2c_order 3   
#define maple2c_func  xc_gga_k_exp4_enhance   
#define kernel_id 1 
#endif   