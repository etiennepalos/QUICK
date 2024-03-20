/*
 !---------------------------------------------------------------------!
 ! Written by QUICK-GenInt code generator on 03/26/2023                !
 !                                                                     !
 ! Copyright (C) 2023-2024 Merz lab                                    !
 ! Copyright (C) 2023-2024 Götz lab                                    !
 !                                                                     !
 ! This Source Code Form is subject to the terms of the Mozilla Public !
 ! License, v. 2.0. If a copy of the MPL was not distributed with this !
 ! file, You can obtain one at http://mozilla.org/MPL/2.0/.            !
 !_____________________________________________________________________!
*/

#undef STOREDIM 
#undef VDIM3 
#undef VY 
#undef LOCSTORE 
#undef STORE_OPERATOR
#define STOREDIM STOREDIM_XL 
#define VDIM3 VDIM3_L 
#define LOCSTORE(A,i1,i2,d1,d2)  A[(i1+(i2)*(d1))*gridDim.x*blockDim.x] 
#define VY(a,b,c) LOCVY(YVerticalTemp, a, b, c, VDIM1, VDIM2, VDIM3) 
#define STORE_OPERATOR =

__device__ __inline__ void ERint_vertical_spdf_5_2(const int I, const int J, const int K, const int L, const int II, const int JJ, const int KK, const int LL, 
        const QUICKDouble Ptempx, const QUICKDouble Ptempy, const QUICKDouble Ptempz, const QUICKDouble WPtempx, const QUICKDouble WPtempy, const QUICKDouble WPtempz, 
        const QUICKDouble Qtempx, const QUICKDouble Qtempy, const QUICKDouble Qtempz, const QUICKDouble WQtempx, const QUICKDouble WQtempy, const QUICKDouble WQtempz, 
        const QUICKDouble ABCDtemp, const QUICKDouble ABtemp, const QUICKDouble CDtemp, const QUICKDouble ABcom, const QUICKDouble CDcom, 
        QUICKDouble* store, QUICKDouble* YVerticalTemp){ 
    if ((I + J) >= 4 && (K + L) >= 6)
    {
#include "iclass_gsis.h" 
    }

 } 

#undef STORE_OPERATOR
#define STORE_OPERATOR +=
