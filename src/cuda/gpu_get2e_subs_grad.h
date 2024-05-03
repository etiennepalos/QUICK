//
//  gpu_get2e_subs_grad.h
//  new_quick
//
//  Created by Yipu Miao on 1/22/14.
//
//

#include "gpu_common.h"

#undef STOREDIM

#ifdef int_sp
#undef VDIM3
#undef LOCSTORE
#undef VY
#define VDIM3 VDIM3_GRAD_T
#define STOREDIM STOREDIM_GRAD_T
#define VY(a,b,c) LOCVY(YVerticalTemp, a, b, c, VDIM1, VDIM2, VDIM3)
#define LOCSTORE(A,i1,i2,d1,d2)  A[(i1+(i2)*(d1))*gridDim.x*blockDim.x]
#elif defined int_spd
#undef VDIM3
#undef VY
#undef LOCSTORE
#define STOREDIM STOREDIM_S
#define VDIM3 VDIM3_S
#define VY(a,b,c) LOCVY(YVerticalTemp, a, b, c, VDIM1, VDIM2, VDIM3)
#define LOCSTORE(A,i1,i2,d1,d2)  A[(i1+(i2)*(d1))*gridDim.x*blockDim.x]
#elif defined int_spdf || defined int_spdf2
#undef VDIM3
#define VDIM3 VDIM3_L
#define STOREDIM STOREDIM_GRAD_S
#elif defined int_spdf3 || defined int_spdf4
#undef VDIM3
#define VDIM3 VDIM3_L
#define STOREDIM STOREDIM_XL
#else
#undef VDIM3
#define VDIM3 VDIM3_L
#define STOREDIM STOREDIM_L
#endif

#ifdef OSHELL
#ifdef int_sp
__global__ void
__launch_bounds__(SM_2X_GRAD_THREADS_PER_BLOCK, 1) getGrad_oshell_kernel_sp()
#elif defined int_spd
__global__ void
__launch_bounds__(SM_2X_GRAD_THREADS_PER_BLOCK, 1) getGrad_oshell_kernel_spd()
#elif defined int_spdf
__global__ void
__launch_bounds__(SM_2X_GRAD_THREADS_PER_BLOCK, 1) getGrad_oshell_kernel_spdf()
#elif defined int_spdf2
__global__ void
__launch_bounds__(SM_2X_GRAD_THREADS_PER_BLOCK, 1) getGrad_oshell_kernel_spdf2()
#elif defined int_spdf3
__global__ void
__launch_bounds__(SM_2X_GRAD_THREADS_PER_BLOCK, 1) getGrad_oshell_kernel_spdf3()
#elif defined int_spdf4
__global__ void
__launch_bounds__(SM_2X_GRAD_THREADS_PER_BLOCK, 1) getGrad_oshell_kernel_spdf4()
#elif defined int_spdf5
__global__ void
__launch_bounds__(SM_2X_GRAD_THREADS_PER_BLOCK, 1) getGrad_oshell_kernel_spdf5()
#elif defined int_spdf6
__global__ void
__launch_bounds__(SM_2X_GRAD_THREADS_PER_BLOCK, 1) getGrad_oshell_kernel_spdf6()
#elif defined int_spdf7
__global__ void
__launch_bounds__(SM_2X_GRAD_THREADS_PER_BLOCK, 1) getGrad_oshell_kernel_spdf7()
#elif defined int_spdf8
__global__ void
__launch_bounds__(SM_2X_GRAD_THREADS_PER_BLOCK, 1) getGrad_oshell_kernel_spdf8()
#endif
#else
#ifdef int_sp
__global__ void
__launch_bounds__(SM_2X_GRAD_THREADS_PER_BLOCK, 1) getGrad_kernel_sp()
#elif defined int_spd
__global__ void 
__launch_bounds__(SM_2X_GRAD_THREADS_PER_BLOCK, 1) getGrad_kernel_spd()
#elif defined int_spdf
__global__ void 
__launch_bounds__(SM_2X_GRAD_THREADS_PER_BLOCK, 1) getGrad_kernel_spdf()
#elif defined int_spdf2
__global__ void 
__launch_bounds__(SM_2X_GRAD_THREADS_PER_BLOCK, 1) getGrad_kernel_spdf2()
#elif defined int_spdf3
__global__ void 
__launch_bounds__(SM_2X_GRAD_THREADS_PER_BLOCK, 1) getGrad_kernel_spdf3()
#elif defined int_spdf4
__global__ void 
__launch_bounds__(SM_2X_GRAD_THREADS_PER_BLOCK, 1) getGrad_kernel_spdf4()
#elif defined int_spdf5
__global__ void 
__launch_bounds__(SM_2X_GRAD_THREADS_PER_BLOCK, 1) getGrad_kernel_spdf5()
#elif defined int_spdf6
__global__ void 
__launch_bounds__(SM_2X_GRAD_THREADS_PER_BLOCK, 1) getGrad_kernel_spdf6()
#elif defined int_spdf7
__global__ void 
__launch_bounds__(SM_2X_GRAD_THREADS_PER_BLOCK, 1) getGrad_kernel_spdf7()
#elif defined int_spdf8
__global__ void 
__launch_bounds__(SM_2X_GRAD_THREADS_PER_BLOCK, 1) getGrad_kernel_spdf8()
#endif
#endif
{

    unsigned int offside = blockIdx.x*blockDim.x+threadIdx.x;
    int totalThreads = blockDim.x*gridDim.x;
    
    
    QUICKULL jshell = (QUICKULL) devSim.sqrQshell;
    QUICKULL jshell2 = (QUICKULL) devSim.sqrQshell;
    
    for (QUICKULL i = offside; i<jshell2*jshell; i+= totalThreads) {
        
        /*
        QUICKULL a, b;
        
        // That's simply because no sqrt for ULL
        double aa = (double)((i+1)*1E-4);
        QUICKULL t = (QUICKULL)(sqrt(aa)*1E2);
        
        
        if ((i+1)==t*t) {
            t--;
        }
        
        QUICKULL k = i-t*t;
        if (k<=t) {
            a = k;
            b = t;
        }else {
            a = t;
            b = 2*t-k;
        }
        */
 
#ifdef int_spdf3
//printf("getGrad_kernel_spdf3 working.. \n");       
#endif

        QUICKULL a = (QUICKULL) i/jshell;
        QUICKULL b = (QUICKULL) (i - a*jshell);

#ifdef CUDA_MPIV
        if(devSim.mpi_bcompute[a] > 0){
#endif
                
        int II = devSim.sorted_YCutoffIJ[a].x;
        int KK = devSim.sorted_YCutoffIJ[b].x;
        
        int ii = devSim.sorted_Q[II];
        int kk = devSim.sorted_Q[KK];
        
        if (ii<=kk){
            
            int JJ = devSim.sorted_YCutoffIJ[a].y;
            int LL = devSim.sorted_YCutoffIJ[b].y;
 
            int iii = devSim.sorted_Qnumber[II];
            int jjj = devSim.sorted_Qnumber[JJ];
            int kkk = devSim.sorted_Qnumber[KK];
            int lll = devSim.sorted_Qnumber[LL];

#ifdef int_sp
            if(iii < 2 && jjj <2 && kkk < 2 && lll < 2){
#endif

#ifdef int_spd
            if(!(iii < 2 && jjj <2 && kkk < 2 && lll < 2)){
#endif
           
            int jj = devSim.sorted_Q[JJ];
            int ll = devSim.sorted_Q[LL];

            
            if ( !((devSim.katom[ii] == devSim.katom[jj]) &&
                   (devSim.katom[ii] == devSim.katom[kk]) &&
                   (devSim.katom[ii] == devSim.katom[ll]))     // In case 4 indices are in the same atom
                ) {
                
                int nshell = devSim.nshell;
                
                QUICKDouble DNMax = MAX(MAX(4.0*LOC2(devSim.cutMatrix, ii, jj, nshell, nshell), 4.0*LOC2(devSim.cutMatrix, kk, ll, nshell, nshell)),
                                        MAX(MAX(LOC2(devSim.cutMatrix, ii, ll, nshell, nshell),     LOC2(devSim.cutMatrix, ii, kk, nshell, nshell)),
                                            MAX(LOC2(devSim.cutMatrix, jj, kk, nshell, nshell),     LOC2(devSim.cutMatrix, jj, ll, nshell, nshell))));
                
                
                if ((LOC2(devSim.YCutoff, kk, ll, nshell, nshell) * LOC2(devSim.YCutoff, ii, jj, nshell, nshell))> devSim.gradCutoff && \
                    (LOC2(devSim.YCutoff, kk, ll, nshell, nshell) * LOC2(devSim.YCutoff, ii, jj, nshell, nshell) * DNMax) > devSim.gradCutoff) {
                    
#ifdef OSHELL

#ifdef int_sp
                    iclass_oshell_grad_sp(iii, jjj, kkk, lll, ii, jj, kk, ll, DNMax, devSim.YVerticalTemp+offside, devSim.store+offside, devSim.store2+offside, devSim.storeAA+offside, devSim.storeBB+offside, devSim.storeCC+offside);
#elif defined int_spd
                    iclass_oshell_grad_spd(iii, jjj, kkk, lll, ii, jj, kk, ll, DNMax, devSim.YVerticalTemp+offside, devSim.store+offside, devSim.store2+offside, devSim.storeAA+offside, devSim.storeBB+offside, devSim.storeCC+offside);
#elif defined int_spdf
                    if ( (kkk + lll) >= 4 ) {
                        iclass_oshell_grad_spdf(iii, jjj, kkk, lll, ii, jj, kk, ll, DNMax, devSim.YVerticalTemp+offside, devSim.store+offside, devSim.store2+offside, devSim.storeAA+offside, devSim.storeBB+offside, devSim.storeCC+offside);
                    }
#elif defined int_spdf2
                    if ( (iii + jjj) >= 4 ) {
                        iclass_oshell_grad_spdf2(iii, jjj, kkk, lll, ii, jj, kk, ll, DNMax, devSim.YVerticalTemp+offside, devSim.store+offside, devSim.store2+offside, devSim.storeAA+offside, devSim.storeBB+offside, devSim.storeCC+offside);
                    }
#elif defined int_spdf3
//                    iclass_oshell_grad_spdf3(iii, jjj, kkk, lll, ii, jj, kk, ll, DNMax, devSim.YVerticalTemp+offside, devSim.store+offside, devSim.store2+offside, devSim.storeAA+offside, devSim.storeBB+offside, devSim.storeCC+offside);
#elif defined int_spdf4
                    iclass_oshell_grad_spdf4(iii, jjj, kkk, lll, ii, jj, kk, ll, DNMax, devSim.YVerticalTemp+offside, devSim.store+offside, devSim.store2+offside, devSim.storeAA+offside, devSim.storeBB+offside, devSim.storeCC+offside);
#elif defined int_spdf5
                    iclass_oshell_grad_spdf5(iii, jjj, kkk, lll, ii, jj, kk, ll, DNMax, devSim.YVerticalTemp+offside, devSim.store+offside, devSim.store2+offside, devSim.storeAA+offside, devSim.storeBB+offside, devSim.storeCC+offside);
#elif defined int_spdf6
                    iclass_oshell_grad_spdf6(iii, jjj, kkk, lll, ii, jj, kk, ll, DNMax, devSim.YVerticalTemp+offside, devSim.store+offside, devSim.store2+offside, devSim.storeAA+offside, devSim.storeBB+offside, devSim.storeCC+offside);
#elif defined int_spdf7
                    iclass_oshell_grad_spdf7(iii, jjj, kkk, lll, ii, jj, kk, ll, DNMax, devSim.YVerticalTemp+offside, devSim.store+offside, devSim.store2+offside, devSim.storeAA+offside, devSim.storeBB+offside, devSim.storeCC+offside);
#elif defined int_spdf8
                    iclass_oshell_grad_spdf8(iii, jjj, kkk, lll, ii, jj, kk, ll, DNMax, devSim.YVerticalTemp+offside, devSim.store+offside, devSim.store2+offside, devSim.storeAA+offside, devSim.storeBB+offside, devSim.storeCC+offside);
#endif
#else
#ifdef int_sp
                    if( iii != 3 && jjj != 3 && kkk != 3 && lll !=3){
                    iclass_grad_sp(iii, jjj, kkk, lll, ii, jj, kk, ll, DNMax, devSim.YVerticalTemp+offside, devSim.store+offside, devSim.store2+offside, devSim.storeAA+offside, devSim.storeBB+offside, devSim.storeCC+offside);
		    }
#elif defined int_spd
                    if( iii != 3 && jjj != 3 && kkk != 3 && lll !=3){
                    iclass_grad_spd(iii, jjj, kkk, lll, ii, jj, kk, ll, DNMax, devSim.YVerticalTemp+offside, devSim.store+offside, devSim.store2+offside, devSim.storeAA+offside, devSim.storeBB+offside, devSim.storeCC+offside);
		    }
#elif defined int_spdf
                    if ( (kkk + lll) >= 4 && iii != 3 && jjj != 3 && kkk != 3 && lll != 3) {
                        iclass_grad_spdf(iii, jjj, kkk, lll, ii, jj, kk, ll, DNMax, devSim.YVerticalTemp+offside, devSim.store+offside, devSim.store2+offside, devSim.storeAA+offside, devSim.storeBB+offside, devSim.storeCC+offside);
                    }
#elif defined int_spdf2
                    if ( (iii + jjj) >= 4 && iii != 3 && jjj != 3 && kkk != 3 && lll != 3) {
                        iclass_grad_spdf2(iii, jjj, kkk, lll, ii, jj, kk, ll, DNMax, devSim.YVerticalTemp+offside, devSim.store+offside, devSim.store2+offside, devSim.storeAA+offside, devSim.storeBB+offside, devSim.storeCC+offside);
                    }
#elif defined int_spdf3
                    if( iii == 3 || jjj == 3 || kkk == 3 || lll ==3 ){
                    if(iii == 3 && jjj == 3 && kkk ==3 && lll ==3){

			}else{
                    iclass_grad_spdf3(iii, jjj, kkk, lll, ii, jj, kk, ll, DNMax, devSim.YVerticalTemp+offside, devSim.store+offside, devSim.store2+offside, devSim.storeAA+offside, devSim.storeBB+offside, devSim.storeCC+offside);
			}
                    }
#elif defined int_spdf4
		    if( iii == 3 && jjj == 3 && kkk ==3 && lll ==3){
                    iclass_grad_spdf4(iii, jjj, kkk, lll, ii, jj, kk, ll, DNMax, devSim.YVerticalTemp+offside, devSim.store+offside, devSim.store2+offside, devSim.storeAA+offside, devSim.storeBB+offside, devSim.storeCC+offside);
		    }
#elif defined int_spdf5
                    iclass_grad_spdf5(iii, jjj, kkk, lll, ii, jj, kk, ll, DNMax, devSim.YVerticalTemp+offside, devSim.store+offside, devSim.store2+offside, devSim.storeAA+offside, devSim.storeBB+offside, devSim.storeCC+offside);
#elif defined int_spdf6
                    iclass_grad_spdf6(iii, jjj, kkk, lll, ii, jj, kk, ll, DNMax, devSim.YVerticalTemp+offside, devSim.store+offside, devSim.store2+offside, devSim.storeAA+offside, devSim.storeBB+offside, devSim.storeCC+offside);
#elif defined int_spdf7
                    iclass_grad_spdf7(iii, jjj, kkk, lll, ii, jj, kk, ll, DNMax, devSim.YVerticalTemp+offside, devSim.store+offside, devSim.store2+offside, devSim.storeAA+offside, devSim.storeBB+offside, devSim.storeCC+offside);
#elif defined int_spdf8
                    iclass_grad_spdf8(iii, jjj, kkk, lll, ii, jj, kk, ll, DNMax, devSim.YVerticalTemp+offside, devSim.store+offside, devSim.store2+offside, devSim.storeAA+offside, devSim.storeBB+offside, devSim.storeCC+offside);
#endif
#endif
                    
                }
            }

#if defined int_sp || defined int_spd
        }
#endif

        }

#ifdef CUDA_MPIV
        }
#endif

    }
}


/*
 iclass subroutine is to generate 2-electron intergral using HRR and VRR method, which is the most
 performance algrithem for electron intergral evaluation. See description below for details
 */
#ifdef int_sp
#ifdef OSHELL
__device__ __forceinline__ void iclass_oshell_grad_sp
#else
__device__ __forceinline__ void iclass_grad_sp
#endif
#elif defined int_spd
#ifdef OSHELL
__device__ __forceinline__ void iclass_oshell_grad_spd
#else
__device__ __forceinline__ void iclass_grad_spd
#endif
#endif

#if defined int_sp || defined int_spd
(int I, int J, int K, int L, unsigned int II, unsigned int JJ, unsigned int KK, unsigned int LL, QUICKDouble DNMax, \
 QUICKDouble* YVerticalTemp, QUICKDouble* store, QUICKDouble* store2, QUICKDouble* storeAA, QUICKDouble* storeBB, QUICKDouble* storeCC){
    /*
     kAtom A, B, C ,D is the coresponding atom for shell ii, jj, kk, ll
     and be careful with the index difference between Fortran and C++,
     Fortran starts array index with 1 and C++ starts 0.
     
     
     RA, RB, RC, and RD are the coordinates for atom katomA, katomB, katomC and katomD,
     which means they are corrosponding coorinates for shell II, JJ, KK, and LL.
     And we don't need the coordinates now, so we will not retrieve the data now.
     */
    QUICKDouble RAx = LOC2(devSim.xyz, 0 , devSim.katom[II]-1, 3, devSim.natom);
    QUICKDouble RAy = LOC2(devSim.xyz, 1 , devSim.katom[II]-1, 3, devSim.natom);
    QUICKDouble RAz = LOC2(devSim.xyz, 2 , devSim.katom[II]-1, 3, devSim.natom);
    
    QUICKDouble RCx = LOC2(devSim.xyz, 0 , devSim.katom[KK]-1, 3, devSim.natom);
    QUICKDouble RCy = LOC2(devSim.xyz, 1 , devSim.katom[KK]-1, 3, devSim.natom);
    QUICKDouble RCz = LOC2(devSim.xyz, 2 , devSim.katom[KK]-1, 3, devSim.natom);
    
    /*
     kPrimI, J, K and L indicates the primtive gaussian function number
     kStartI, J, K, and L indicates the starting guassian function for shell I, J, K, and L.
     We retrieve from global memory and save them to register to avoid multiple retrieve.
     */
    int kPrimI = devSim.kprim[II];
    int kPrimJ = devSim.kprim[JJ];
    int kPrimK = devSim.kprim[KK];
    int kPrimL = devSim.kprim[LL];
    
    int kStartI = devSim.kstart[II]-1;
    int kStartJ = devSim.kstart[JJ]-1;
    int kStartK = devSim.kstart[KK]-1;
    int kStartL = devSim.kstart[LL]-1;
    
    
    /*
     store saves temp contracted integral as [as|bs] type. the dimension should be allocatable but because
     of cuda limitation, we can not do that now.
     
     See M.Head-Gordon and J.A.Pople, Jchem.Phys., 89, No.9 (1988) for VRR algrithem details.
     */
    /*
    QUICKDouble store[STOREDIM*STOREDIM];
    QUICKDouble storeAA[STOREDIM*STOREDIM];
    QUICKDouble storeBB[STOREDIM*STOREDIM];
    QUICKDouble storeCC[STOREDIM*STOREDIM];
    */
    
    for (int i = Sumindex[K]; i< Sumindex[K+L+3]; i++) {
        for (int j = Sumindex[I]; j< Sumindex[I+J+3]; j++) {
            if (i < STOREDIM && j < STOREDIM) {

                LOCSTORE(store2, j, i, STOREDIM, STOREDIM) = 0;

                if (j < Sumindex[I+J+2] && i < Sumindex[K+L+2]) {
                    LOCSTORE(store, j, i, STOREDIM, STOREDIM) = 0;
                }
                
                if (j >= Sumindex[I+1]) {
                    LOCSTORE(storeAA, j, i, STOREDIM, STOREDIM) = 0;
                    LOCSTORE(storeBB, j, i, STOREDIM, STOREDIM) = 0;
                }
                
                if (i >= Sumindex[K+1]) {
                    LOCSTORE(storeCC, j, i, STOREDIM, STOREDIM) = 0;
                }
            }
        }
    }
    
    
    
    for (int i = 0; i<kPrimI*kPrimJ;i++){
        int JJJ = (int) i/kPrimI;
        int III = (int) i-kPrimI*JJJ;
        /*
         In the following comments, we have I, J, K, L denote the primitive gaussian function we use, and
         for example, expo(III, ksumtype(II)) stands for the expo for the IIIth primitive guassian function for II shell,
         we use I to express the corresponding index.
         AB = expo(I)+expo(J)
         --->                --->
         ->     expo(I) * xyz (I) + expo(J) * xyz(J)
         P  = ---------------------------------------
         expo(I) + expo(J)
         Those two are pre-calculated in CPU stage.
         
         */
        int ii_start = devSim.prim_start[II];
        int jj_start = devSim.prim_start[JJ];
        
        QUICKDouble AA = LOC2(devSim.gcexpo, III , devSim.Ksumtype[II] - 1, MAXPRIM, devSim.nbasis);
        QUICKDouble BB = LOC2(devSim.gcexpo, JJJ , devSim.Ksumtype[JJ] - 1, MAXPRIM, devSim.nbasis);
        
        QUICKDouble AB = LOC2(devSim.expoSum, ii_start+III, jj_start+JJJ, devSim.prim_total, devSim.prim_total);
        QUICKDouble Px = LOC2(devSim.weightedCenterX, ii_start+III, jj_start+JJJ, devSim.prim_total, devSim.prim_total);
        QUICKDouble Py = LOC2(devSim.weightedCenterY, ii_start+III, jj_start+JJJ, devSim.prim_total, devSim.prim_total);
        QUICKDouble Pz = LOC2(devSim.weightedCenterZ, ii_start+III, jj_start+JJJ, devSim.prim_total, devSim.prim_total);
        
        /*
         X1 is the contracted coeffecient, which is pre-calcuated in CPU stage as well.
         cutoffprim is used to cut too small prim gaussian function when bring density matrix into consideration.
         */
        QUICKDouble cutoffPrim = DNMax * LOC2(devSim.cutPrim, kStartI+III, kStartJ+JJJ, devSim.jbasis, devSim.jbasis);
        QUICKDouble X1 = LOC4(devSim.Xcoeff, kStartI+III, kStartJ+JJJ, I - devSim.Qstart[II], J - devSim.Qstart[JJ], devSim.jbasis, devSim.jbasis, 2, 2);
        
        for (int j = 0; j<kPrimK*kPrimL; j++){
            int LLL = (int) j/kPrimK;
            int KKK = (int) j-kPrimK*LLL;
            
            if (cutoffPrim * LOC2(devSim.cutPrim, kStartK+KKK, kStartL+LLL, devSim.jbasis, devSim.jbasis) > devSim.primLimit) {
                
                QUICKDouble CC = LOC2(devSim.gcexpo, KKK , devSim.Ksumtype[KK] - 1, MAXPRIM, devSim.nbasis);
                /*
                 CD = expo(L)+expo(K)
                 ABCD = 1/ (AB + CD) = 1 / (expo(I)+expo(J)+expo(K)+expo(L))
                 AB * CD      (expo(I)+expo(J))*(expo(K)+expo(L))
                 Rou(Greek Letter) =   ----------- = ------------------------------------
                 AB + CD         expo(I)+expo(J)+expo(K)+expo(L)
                 
                 expo(I)+expo(J)                        expo(K)+expo(L)
                 ABcom = --------------------------------  CDcom = --------------------------------
                 expo(I)+expo(J)+expo(K)+expo(L)           expo(I)+expo(J)+expo(K)+expo(L)
                 
                 ABCDtemp = 1/2(expo(I)+expo(J)+expo(K)+expo(L))
                 */
                
                int kk_start = devSim.prim_start[KK];
                int ll_start = devSim.prim_start[LL];
                
                QUICKDouble CD = LOC2(devSim.expoSum, kk_start+KKK, ll_start+LLL, devSim.prim_total, devSim.prim_total);
                
                QUICKDouble ABCD = 1/(AB+CD);
                
                /*
                 X2 is the multiplication of four indices normalized coeffecient
                 */
                QUICKDouble X2 = sqrt(ABCD) * X1 * LOC4(devSim.Xcoeff, kStartK+KKK, kStartL+LLL, K - devSim.Qstart[KK], L - devSim.Qstart[LL], devSim.jbasis, devSim.jbasis, 2, 2);
                
                /*
                 Q' is the weighting center of K and L
                 --->           --->
                 ->  ------>       expo(K)*xyz(K)+expo(L)*xyz(L)
                 Q = P'(K,L)  = ------------------------------
                 expo(K) + expo(L)
                 
                 W' is the weight center for I, J, K, L
                 
                 --->             --->             --->            --->
                 ->     expo(I)*xyz(I) + expo(J)*xyz(J) + expo(K)*xyz(K) +expo(L)*xyz(L)
                 W = -------------------------------------------------------------------
                 expo(I) + expo(J) + expo(K) + expo(L)
                 ->  ->  2
                 RPQ =| P - Q |
                 
                 ->  -> 2
                 T = ROU * | P - Q|
                 */
                
                QUICKDouble Qx = LOC2(devSim.weightedCenterX, kk_start+KKK, ll_start+LLL, devSim.prim_total, devSim.prim_total);
                QUICKDouble Qy = LOC2(devSim.weightedCenterY, kk_start+KKK, ll_start+LLL, devSim.prim_total, devSim.prim_total);
                QUICKDouble Qz = LOC2(devSim.weightedCenterZ, kk_start+KKK, ll_start+LLL, devSim.prim_total, devSim.prim_total);
                
                //QUICKDouble T = AB * CD * ABCD * ( quick_dsqr(Px-Qx) + quick_dsqr(Py-Qy) + quick_dsqr(Pz-Qz));
                
                //QUICKDouble YVerticalTemp[VDIM1*VDIM2*VDIM3];
#ifdef int_sp
                FmT_grad_sp(I+J+K+L+1, AB * CD * ABCD * ( quick_dsqr(Px-Qx) + quick_dsqr(Py-Qy) + quick_dsqr(Pz-Qz)), YVerticalTemp);
#elif defined int_spd
                FmT_spd(I+J+K+L+1, AB * CD * ABCD * ( quick_dsqr(Px-Qx) + quick_dsqr(Py-Qy) + quick_dsqr(Pz-Qz)), YVerticalTemp);
#endif
                
                for (int i = 0; i<=I+J+K+L+1; i++) {
                    VY(0, 0, i) = VY(0, 0, i) * X2;
                }
                
                
                //QUICKDouble store2[STOREDIM*STOREDIM];
                
/*                
                vertical2(I, J + 1, K, L + 1, YVerticalTemp, store2, \
                          Px - RAx, Py - RAy, Pz - RAz, (Px*AB+Qx*CD)*ABCD - Px, (Py*AB+Qy*CD)*ABCD - Py, (Pz*AB+Qz*CD)*ABCD - Pz, \
                          Qx - RCx, Qy - RCy, Qz - RCz, (Px*AB+Qx*CD)*ABCD - Qx, (Py*AB+Qy*CD)*ABCD - Qy, (Pz*AB+Qz*CD)*ABCD - Qz, \
                          0.5 * ABCD, 0.5 / AB, 0.5 / CD, AB * ABCD, CD * ABCD);
*/

//#ifdef USE_ERI_GRAD_STOREADD
/*                for (int i = Sumindex[K]; i< Sumindex[K+L+3]; i++) {
                    for (int j = Sumindex[I]; j< Sumindex[I+J+3]; j++) {
                        if (i < STOREDIM && j < STOREDIM) {
                                LOCSTORE(store2, j, i, STOREDIM, STOREDIM) = 0;
                        }
                    }
                }
*/
//#endif
      
#ifdef int_sp 
                ERint_grad_vertical_sp(I, J+1, K, L+1, II, JJ, KK, LL, \
                         Px - RAx, Py - RAy, Pz - RAz, (Px*AB+Qx*CD)*ABCD - Px, (Py*AB+Qy*CD)*ABCD - Py, (Pz*AB+Qz*CD)*ABCD - Pz, \
                         Qx - RCx, Qy - RCy, Qz - RCz, (Px*AB+Qx*CD)*ABCD - Qx, (Py*AB+Qy*CD)*ABCD - Qy, (Pz*AB+Qz*CD)*ABCD - Qz, \
                         0.5 * ABCD, 0.5 / AB, 0.5 / CD, AB * ABCD, CD * ABCD, store2, YVerticalTemp);
#elif defined int_spd          
                ERint_grad_vertical_spd(I, J+1, K, L+1, II, JJ, KK, LL, \
                         Px - RAx, Py - RAy, Pz - RAz, (Px*AB+Qx*CD)*ABCD - Px, (Py*AB+Qy*CD)*ABCD - Py, (Pz*AB+Qz*CD)*ABCD - Pz, \
                         Qx - RCx, Qy - RCy, Qz - RCz, (Px*AB+Qx*CD)*ABCD - Qx, (Py*AB+Qy*CD)*ABCD - Qy, (Pz*AB+Qz*CD)*ABCD - Qz, \
                         0.5 * ABCD, 0.5 / AB, 0.5 / CD, AB * ABCD, CD * ABCD, store2, YVerticalTemp);
#endif                
                
                for (int i = Sumindex[K]; i< Sumindex[K+L+2]; i++) {
                    for (int j = Sumindex[I]; j< Sumindex[I+J+2]; j++) {
                        if (i < STOREDIM && j < STOREDIM) {
                            LOCSTORE(store, j, i , STOREDIM, STOREDIM) += LOCSTORE(store2, j, i, STOREDIM, STOREDIM);
                        }
                    }
                }
                
                
                for (int i = Sumindex[K]; i< Sumindex[K+L+2]; i++) {
                    for (int j = Sumindex[I+1]; j< Sumindex[I+J+3]; j++) {
                        if (i < STOREDIM && j < STOREDIM) {
                            LOCSTORE(storeAA, j, i, STOREDIM, STOREDIM) += LOCSTORE(store2, j, i, STOREDIM, STOREDIM) * AA * 2 ;
                            LOCSTORE(storeBB, j, i, STOREDIM, STOREDIM) += LOCSTORE(store2, j, i, STOREDIM, STOREDIM) * BB * 2 ;
                        }
                        
                    }
                }
                
                for (int i = Sumindex[K+1]; i< Sumindex[K+L+3]; i++) {
                    for (int j = Sumindex[I]; j< Sumindex[I+J+2]; j++) {
                        if (i < STOREDIM && j < STOREDIM) {
                            LOCSTORE(storeCC, j, i, STOREDIM, STOREDIM) += LOCSTORE(store2, j, i, STOREDIM, STOREDIM) * CC * 2 ;

                        }
                    }
                }

                
            }
        }
    }
    
/*

                for (int i = Sumindex[K]; i< Sumindex[K+L+2]; i++) {
                    for (int j = Sumindex[I]; j< Sumindex[I+J+2]; j++) {
                        if (i < STOREDIM && j < STOREDIM) {
                            printf("STORE   %d   %d   %d   %d   %d   %d   %d   %d   %d   %d   %.9f \n",II, JJ, KK, LL, I, J, K, L, j, i, LOCSTORE(store, j, i , STOREDIM, STOREDIM));
                        }
                    }
                }
   
*/ 
    QUICKDouble AGradx = 0.0;
    QUICKDouble AGrady = 0.0;
    QUICKDouble AGradz = 0.0;
    QUICKDouble BGradx = 0.0;
    QUICKDouble BGrady = 0.0;
    QUICKDouble BGradz = 0.0;
    QUICKDouble CGradx = 0.0;
    QUICKDouble CGrady = 0.0;
    QUICKDouble CGradz = 0.0;
    
    int         AStart = (devSim.katom[II]-1) * 3;
    int         BStart = (devSim.katom[JJ]-1) * 3;
    int         CStart = (devSim.katom[KK]-1) * 3;
    int         DStart = (devSim.katom[LL]-1) * 3;
    
    
    QUICKDouble RBx, RBy, RBz;
    QUICKDouble RDx, RDy, RDz;
    
    RBx = LOC2(devSim.xyz, 0 , devSim.katom[JJ]-1, 3, devSim.natom);
    RBy = LOC2(devSim.xyz, 1 , devSim.katom[JJ]-1, 3, devSim.natom);
    RBz = LOC2(devSim.xyz, 2 , devSim.katom[JJ]-1, 3, devSim.natom);
    
    
    RDx = LOC2(devSim.xyz, 0 , devSim.katom[LL]-1, 3, devSim.natom);
    RDy = LOC2(devSim.xyz, 1 , devSim.katom[LL]-1, 3, devSim.natom);
    RDz = LOC2(devSim.xyz, 2 , devSim.katom[LL]-1, 3, devSim.natom);
    
    int III1 = LOC2(devSim.Qsbasis, II, I, devSim.nshell, 4);
    int III2 = LOC2(devSim.Qfbasis, II, I, devSim.nshell, 4);
    int JJJ1 = LOC2(devSim.Qsbasis, JJ, J, devSim.nshell, 4);
    int JJJ2 = LOC2(devSim.Qfbasis, JJ, J, devSim.nshell, 4);
    int KKK1 = LOC2(devSim.Qsbasis, KK, K, devSim.nshell, 4);
    int KKK2 = LOC2(devSim.Qfbasis, KK, K, devSim.nshell, 4);
    int LLL1 = LOC2(devSim.Qsbasis, LL, L, devSim.nshell, 4);
    int LLL2 = LOC2(devSim.Qfbasis, LL, L, devSim.nshell, 4);
    
    
    int  IJKLTYPE = 999;
    
    int  nbasis = devSim.nbasis;

    for (int III = III1; III <= III2; III++) {
        for (int JJJ = MAX(III,JJJ1); JJJ <= JJJ2; JJJ++) {
            for (int KKK = MAX(III,KKK1); KKK <= KKK2; KKK++) {
                for (int LLL = MAX(KKK,LLL1); LLL <= LLL2; LLL++) {
                    
                    if (III < KKK ||
                        ((III == JJJ) && (III == LLL)) ||
                        ((III == JJJ) && (III  < LLL)) ||
                        ((JJJ == LLL) && (III  < JJJ)) ||
                        ((III == KKK) && (III  < JJJ)  && (JJJ < LLL))) {
                        
                        QUICKDouble Yaax, Yaay, Yaaz;
                        QUICKDouble Ybbx, Ybby, Ybbz;
                        QUICKDouble Yccx, Yccy, Yccz;
                       
#ifdef int_sp
                        hrrwholegrad_sp
#elif defined int_spd  
                        hrrwholegrad
#endif
				     (&Yaax, &Yaay, &Yaaz, \
                                     &Ybbx, &Ybby, &Ybbz, \
                                     &Yccx, &Yccy, &Yccz, \
                                     I, J, K, L,\
                                     III, JJJ, KKK, LLL, IJKLTYPE, \
                                     store, storeAA, storeBB, storeCC, \
                                     RAx, RAy, RAz, RBx, RBy, RBz, \
                                     RCx, RCy, RCz, RDx, RDy, RDz);
  
//printf("Y   %d   %d   %d   %d   %d   %d   %d   %d   %d   %d   %d   %d   %.9f   %.9f   %.9f   %.9f   %.9f   %.9f   %.9f   %.9f   %.9f\n",II, JJ, KK, LL, I, J, K, L, III, JJJ, KKK, LLL, Yaax, Yaay, Yaaz, Ybbx, Ybby, Ybbz, Yccx, Yccy, Yccz);
                      
                        QUICKDouble constant = 0.0 ;
                        
#ifdef OSHELL
                        QUICKDouble DENSELJ = (QUICKDouble) (LOC2(devSim.dense, LLL-1, JJJ-1, nbasis, nbasis)+LOC2(devSim.denseb, LLL-1, JJJ-1, nbasis, nbasis));
                        QUICKDouble DENSELI = (QUICKDouble) (LOC2(devSim.dense, LLL-1, III-1, nbasis, nbasis)+LOC2(devSim.denseb, LLL-1, III-1, nbasis, nbasis));
                        QUICKDouble DENSELK = (QUICKDouble) (LOC2(devSim.dense, LLL-1, KKK-1, nbasis, nbasis)+LOC2(devSim.denseb, LLL-1, KKK-1, nbasis, nbasis));
                        QUICKDouble DENSEJI = (QUICKDouble) (LOC2(devSim.dense, JJJ-1, III-1, nbasis, nbasis)+LOC2(devSim.denseb, JJJ-1, III-1, nbasis, nbasis));

                        QUICKDouble DENSEKIA = (QUICKDouble) LOC2(devSim.dense, KKK-1, III-1, nbasis, nbasis);
                        QUICKDouble DENSEKJA = (QUICKDouble) LOC2(devSim.dense, KKK-1, JJJ-1, nbasis, nbasis);
                        QUICKDouble DENSELJA = (QUICKDouble) LOC2(devSim.dense, LLL-1, JJJ-1, nbasis, nbasis);
                        QUICKDouble DENSELIA = (QUICKDouble) LOC2(devSim.dense, LLL-1, III-1, nbasis, nbasis);
                        QUICKDouble DENSEJIA = (QUICKDouble) LOC2(devSim.dense, JJJ-1, III-1, nbasis, nbasis);

                        QUICKDouble DENSEKIB = (QUICKDouble) LOC2(devSim.denseb, KKK-1, III-1, nbasis, nbasis);
                        QUICKDouble DENSEKJB = (QUICKDouble) LOC2(devSim.denseb, KKK-1, JJJ-1, nbasis, nbasis);
                        QUICKDouble DENSELJB = (QUICKDouble) LOC2(devSim.denseb, LLL-1, JJJ-1, nbasis, nbasis);
                        QUICKDouble DENSELIB = (QUICKDouble) LOC2(devSim.denseb, LLL-1, III-1, nbasis, nbasis);
                        QUICKDouble DENSEJIB = (QUICKDouble) LOC2(devSim.denseb, JJJ-1, III-1, nbasis, nbasis);
#else
                        QUICKDouble DENSEKI = (QUICKDouble) LOC2(devSim.dense, KKK-1, III-1, nbasis, nbasis);
                        QUICKDouble DENSEKJ = (QUICKDouble) LOC2(devSim.dense, KKK-1, JJJ-1, nbasis, nbasis);
                        QUICKDouble DENSELJ = (QUICKDouble) LOC2(devSim.dense, LLL-1, JJJ-1, nbasis, nbasis);
                        QUICKDouble DENSELI = (QUICKDouble) LOC2(devSim.dense, LLL-1, III-1, nbasis, nbasis);
                        QUICKDouble DENSELK = (QUICKDouble) LOC2(devSim.dense, LLL-1, KKK-1, nbasis, nbasis);
                        QUICKDouble DENSEJI = (QUICKDouble) LOC2(devSim.dense, JJJ-1, III-1, nbasis, nbasis);
#endif
                        
                        if (II < JJ && II < KK && KK < LL ||
                            ( III < KKK && III < JJJ && KKK < LLL)) {
#ifdef OSHELL
                            constant = ( 4.0 * DENSEJI * DENSELK - 2.0 * devSim.hyb_coeff * DENSEKIA * DENSELJA - 2.0 * devSim.hyb_coeff * DENSELIA * DENSEKJA
                                       - 2.0 * devSim.hyb_coeff * DENSEKIB * DENSELJB - 2.0 * devSim.hyb_coeff * DENSELIB * DENSEKJB);
#else
                            constant = ( 4.0 * DENSEJI * DENSELK - devSim.hyb_coeff * DENSEKI * DENSELJ - devSim.hyb_coeff * DENSELI * DENSEKJ);
#endif
                        }else{
                            if (III < KKK) {
                                if( III == JJJ && KKK == LLL){
#ifdef OSHELL
                                    constant = (DENSEJI * DENSELK - devSim.hyb_coeff * DENSEKIA * DENSEKIA - devSim.hyb_coeff * DENSEKIB * DENSEKIB);
#else
                                    constant = (DENSEJI * DENSELK - 0.5 * devSim.hyb_coeff * DENSEKI * DENSEKI);
#endif
                                }else if (JJJ == KKK && JJJ == LLL){
#ifdef OSHELL
                                    constant = 2.0 * DENSELJ * DENSEJI - 2.0 * devSim.hyb_coeff * DENSELJA * DENSEJIA - 2.0 * devSim.hyb_coeff * DENSELJB * DENSEJIB;
#else
                                    constant = 2.0 * DENSELJ * DENSEJI - devSim.hyb_coeff * DENSELJ * DENSEJI;
#endif
                                }else if (KKK == LLL && III < JJJ && JJJ != KKK){
#ifdef OSHELL
                                    constant = (2.0* DENSEJI * DENSELK - 2.0 * devSim.hyb_coeff * DENSEKIA * DENSEKJA - 2.0 * devSim.hyb_coeff * DENSEKIB * DENSEKJB);
#else
                                    constant = (2.0* DENSEJI * DENSELK - devSim.hyb_coeff * DENSEKI * DENSEKJ);
#endif
                                }else if ( III == JJJ && KKK < LLL){
#ifdef OSHELL
                                    constant = (2.0* DENSELK * DENSEJI - 2.0 * devSim.hyb_coeff * DENSEKIA * DENSELIA - 2.0 * devSim.hyb_coeff * DENSEKIB * DENSELIB);
#else
                                    constant = (2.0* DENSELK * DENSEJI - devSim.hyb_coeff * DENSEKI * DENSELI);
#endif
                                }
                            }
                            else{
                                if (JJJ <= LLL) {
                                    if (III == JJJ && III == KKK && III == LLL) {
                                        // Do nothing
                                    }else if (III==JJJ && III==KKK && III < LLL){
#ifdef OSHELL
                                        constant = 2.0 * DENSELI * DENSEJI - 2.0 * devSim.hyb_coeff * DENSELIA * DENSEJIA - 2.0 * devSim.hyb_coeff * DENSELIB * DENSEJIB;
#else
					constant = 2.0 * DENSELI * DENSEJI - devSim.hyb_coeff * DENSELI * DENSEJI;
#endif
                                    }else if (III==KKK && JJJ==LLL && III < JJJ){
#ifdef OSHELL
                                        constant = (2.0 * DENSEJI * DENSEJI - devSim.hyb_coeff * DENSEJIA * DENSEJIA - devSim.hyb_coeff * DENSELJA * DENSEKIA
                                                   - devSim.hyb_coeff * DENSEJIB * DENSEJIB - devSim.hyb_coeff * DENSELJB * DENSEKIB);
#else
					constant = (2.0 * DENSEJI * DENSEJI - 0.5 * devSim.hyb_coeff * DENSEJI * DENSEJI - 0.5 * devSim.hyb_coeff * DENSELJ * DENSEKI);
#endif
                                    }else if (III== KKK && III < JJJ && JJJ < LLL){
#ifdef OSHELL
                                        constant = (4.0 * DENSEJI * DENSELI - 2.0 * devSim.hyb_coeff * DENSEJIA * DENSELIA - 2.0 * devSim.hyb_coeff * DENSELJA * DENSEKIA
                                                   - 2.0 * devSim.hyb_coeff * DENSEJIB * DENSELIB - 2.0 * devSim.hyb_coeff * DENSELJB * DENSEKIB);
#else
					constant = (4.0 * DENSEJI * DENSELI - devSim.hyb_coeff * DENSEJI * DENSELI - devSim.hyb_coeff * DENSELJ * DENSEKI);
#endif
                                    }
                                }
                            }
                        }
                        
                        
                        AGradx += constant * Yaax;
                        AGrady += constant * Yaay;
                        AGradz += constant * Yaaz;
                        
                        BGradx += constant * Ybbx;
                        BGrady += constant * Ybby;
                        BGradz += constant * Ybbz;
                        
                        CGradx += constant * Yccx;
                        CGrady += constant * Yccy;
                        CGradz += constant * Yccz;
                    }
                }
            }
        }
    }
    
    
   
#ifdef USE_LEGACY_ATOMICS 
    GRADADD(devSim.gradULL[AStart], AGradx);
    GRADADD(devSim.gradULL[AStart + 1], AGrady);
    GRADADD(devSim.gradULL[AStart + 2], AGradz);
    
    
    GRADADD(devSim.gradULL[BStart], BGradx);
    GRADADD(devSim.gradULL[BStart + 1], BGrady);
    GRADADD(devSim.gradULL[BStart + 2], BGradz);
    
    
    GRADADD(devSim.gradULL[CStart], CGradx);
    GRADADD(devSim.gradULL[CStart + 1], CGrady);
    GRADADD(devSim.gradULL[CStart + 2], CGradz);
    
    
    GRADADD(devSim.gradULL[DStart], (-AGradx-BGradx-CGradx));
    GRADADD(devSim.gradULL[DStart + 1], (-AGrady-BGrady-CGrady));
    GRADADD(devSim.gradULL[DStart + 2], (-AGradz-BGradz-CGradz));
#else    
    atomicAdd(&devSim.grad[AStart], AGradx);
    atomicAdd(&devSim.grad[AStart + 1], AGrady);
    atomicAdd(&devSim.grad[AStart + 2], AGradz);


    atomicAdd(&devSim.grad[BStart], BGradx);
    atomicAdd(&devSim.grad[BStart + 1], BGrady);
    atomicAdd(&devSim.grad[BStart + 2], BGradz);


    atomicAdd(&devSim.grad[CStart], CGradx);
    atomicAdd(&devSim.grad[CStart + 1], CGrady);
    atomicAdd(&devSim.grad[CStart + 2], CGradz);


    atomicAdd(&devSim.grad[DStart], (-AGradx-BGradx-CGradx));
    atomicAdd(&devSim.grad[DStart + 1], (-AGrady-BGrady-CGrady));
    atomicAdd(&devSim.grad[DStart + 2], (-AGradz-BGradz-CGradz));
#endif
    return;
}
#else


/*
 iclass subroutine is to generate 2-electron intergral using HRR and VRR method, which is the most
 performance algrithem for electron intergral evaluation. See description below for details
 */
#ifdef OSHELL
#ifdef int_spdf
__device__ __forceinline__ void iclass_oshell_grad_spdf
#elif defined int_spdf2
__device__ __forceinline__ void iclass_oshell_grad_spdf2
#elif defined int_spdf3
__device__ __forceinline__ void iclass_oshell_grad_spdf3
#elif defined int_spdf4
__device__ __forceinline__ void iclass_oshell_grad_spdf4
#elif defined int_spdf5
__device__ __forceinline__ void iclass_oshell_grad_spdf5
#elif defined int_spdf6
__device__ __forceinline__ void iclass_oshell_grad_spdf6
#elif defined int_spdf7
__device__ __forceinline__ void iclass_oshell_grad_spdf7
#elif defined int_spdf8
__device__ __forceinline__ void iclass_oshell_grad_spdf8
#endif
#else
#ifdef int_spdf
__device__ __forceinline__ void iclass_grad_spdf
#elif defined int_spdf2
__device__ __forceinline__ void iclass_grad_spdf2
#elif defined int_spdf3
__device__ __forceinline__ void iclass_grad_spdf3
#elif defined int_spdf4
__device__ __forceinline__ void iclass_grad_spdf4
#elif defined int_spdf5
__device__ __forceinline__ void iclass_grad_spdf5
#elif defined int_spdf6
__device__ __forceinline__ void iclass_grad_spdf6
#elif defined int_spdf7
__device__ __forceinline__ void iclass_grad_spdf7
#elif defined int_spdf8
__device__ __forceinline__ void iclass_grad_spdf8
#endif
#endif
(int I, int J, int K, int L, unsigned int II, unsigned int JJ, unsigned int KK, unsigned int LL, QUICKDouble DNMax, \
QUICKDouble* YVerticalTemp, QUICKDouble* store, QUICKDouble* store2, QUICKDouble* storeAA, QUICKDouble* storeBB, QUICKDouble* storeCC){
    /*
     kAtom A, B, C ,D is the coresponding atom for shell ii, jj, kk, ll
     and be careful with the index difference between Fortran and C++,
     Fortran starts array index with 1 and C++ starts 0.
     
     
     RA, RB, RC, and RD are the coordinates for atom katomA, katomB, katomC and katomD,
     which means they are corrosponding coorinates for shell II, JJ, KK, and LL.
     And we don't need the coordinates now, so we will not retrieve the data now.
     */
    QUICKDouble RAx = LOC2(devSim.xyz, 0 , devSim.katom[II]-1, 3, devSim.natom);
    QUICKDouble RAy = LOC2(devSim.xyz, 1 , devSim.katom[II]-1, 3, devSim.natom);
    QUICKDouble RAz = LOC2(devSim.xyz, 2 , devSim.katom[II]-1, 3, devSim.natom);
    
    QUICKDouble RCx = LOC2(devSim.xyz, 0 , devSim.katom[KK]-1, 3, devSim.natom);
    QUICKDouble RCy = LOC2(devSim.xyz, 1 , devSim.katom[KK]-1, 3, devSim.natom);
    QUICKDouble RCz = LOC2(devSim.xyz, 2 , devSim.katom[KK]-1, 3, devSim.natom);
    
    /*
     kPrimI, J, K and L indicates the primtive gaussian function number
     kStartI, J, K, and L indicates the starting guassian function for shell I, J, K, and L.
     We retrieve from global memory and save them to register to avoid multiple retrieve.
     */
    int kPrimI = devSim.kprim[II];
    int kPrimJ = devSim.kprim[JJ];
    int kPrimK = devSim.kprim[KK];
    int kPrimL = devSim.kprim[LL];
    
    int kStartI = devSim.kstart[II]-1;
    int kStartJ = devSim.kstart[JJ]-1;
    int kStartK = devSim.kstart[KK]-1;
    int kStartL = devSim.kstart[LL]-1;
    
/*    
    QUICKDouble AGradx = 0.0;
    QUICKDouble AGrady = 0.0;
    QUICKDouble AGradz = 0.0;
    QUICKDouble BGradx = 0.0;
    QUICKDouble BGrady = 0.0;
    QUICKDouble BGradz = 0.0;
    QUICKDouble CGradx = 0.0;
    QUICKDouble CGrady = 0.0;
    QUICKDouble CGradz = 0.0;
    
    int         AStart = (devSim.katom[II]-1) * 3;
    int         BStart = (devSim.katom[JJ]-1) * 3;
    int         CStart = (devSim.katom[KK]-1) * 3;
    int         DStart = (devSim.katom[LL]-1) * 3;
*/    
    /*
     store saves temp contracted integral as [as|bs] type. the dimension should be allocatable but because
     of cuda limitation, we can not do that now.
     
     See M.Head-Gordon and J.A.Pople, Jchem.Phys., 89, No.9 (1988) for VRR algrithem details.
     */

    for (int i = Sumindex[K]; i< Sumindex[K+L+3]; i++) {
        for (int j = Sumindex[I]; j< Sumindex[I+J+3]; j++) {
            if (i < STOREDIM && j < STOREDIM) {

                LOCSTORE(store2, j, i, STOREDIM, STOREDIM) = 0;

                if (j < Sumindex[I+J+2] && i < Sumindex[K+L+2]) {
                    LOCSTORE(store, j, i, STOREDIM, STOREDIM) = 0;
                }

                if (j >= Sumindex[I+1]) {
                    LOCSTORE(storeAA, j, i, STOREDIM, STOREDIM) = 0;
                    LOCSTORE(storeBB, j, i, STOREDIM, STOREDIM) = 0;
                }

                if (i >= Sumindex[K+1]) {
                    LOCSTORE(storeCC, j, i, STOREDIM, STOREDIM) = 0;
                }
            }
        }
    }    

    
    for (int i = 0; i<kPrimI*kPrimJ;i++){
        int JJJ = (int) i/kPrimI;
        int III = (int) i-kPrimI*JJJ;
        /*
         In the following comments, we have I, J, K, L denote the primitive gaussian function we use, and
         for example, expo(III, ksumtype(II)) stands for the expo for the IIIth primitive guassian function for II shell,
         we use I to express the corresponding index.
         AB = expo(I)+expo(J)
         --->                --->
         ->     expo(I) * xyz (I) + expo(J) * xyz(J)
         P  = ---------------------------------------
         expo(I) + expo(J)
         Those two are pre-calculated in CPU stage.
         
         */
        int ii_start = devSim.prim_start[II];
        int jj_start = devSim.prim_start[JJ];
        
        QUICKDouble AA = LOC2(devSim.gcexpo, III , devSim.Ksumtype[II] - 1, MAXPRIM, devSim.nbasis);
        QUICKDouble BB = LOC2(devSim.gcexpo, JJJ , devSim.Ksumtype[JJ] - 1, MAXPRIM, devSim.nbasis);
        
        QUICKDouble AB = LOC2(devSim.expoSum, ii_start+III, jj_start+JJJ, devSim.prim_total, devSim.prim_total);
        QUICKDouble Px = LOC2(devSim.weightedCenterX, ii_start+III, jj_start+JJJ, devSim.prim_total, devSim.prim_total);
        QUICKDouble Py = LOC2(devSim.weightedCenterY, ii_start+III, jj_start+JJJ, devSim.prim_total, devSim.prim_total);
        QUICKDouble Pz = LOC2(devSim.weightedCenterZ, ii_start+III, jj_start+JJJ, devSim.prim_total, devSim.prim_total);
        
        /*
         X1 is the contracted coeffecient, which is pre-calcuated in CPU stage as well.
         cutoffprim is used to cut too small prim gaussian function when bring density matrix into consideration.
         */
        QUICKDouble cutoffPrim = DNMax * LOC2(devSim.cutPrim, kStartI+III, kStartJ+JJJ, devSim.jbasis, devSim.jbasis);
        QUICKDouble X1 = LOC4(devSim.Xcoeff, kStartI+III, kStartJ+JJJ, I - devSim.Qstart[II], J - devSim.Qstart[JJ], devSim.jbasis, devSim.jbasis, 2, 2);
        
        
        for (int j = 0; j<kPrimK*kPrimL; j++){
            int LLL = (int) j/kPrimK;
            int KKK = (int) j-kPrimK*LLL;
            
            if (cutoffPrim * LOC2(devSim.cutPrim, kStartK+KKK, kStartL+LLL, devSim.jbasis, devSim.jbasis) > devSim.primLimit) {
                
                QUICKDouble CC = LOC2(devSim.gcexpo, KKK , devSim.Ksumtype[KK] - 1, MAXPRIM, devSim.nbasis);
                /*
                 CD = expo(L)+expo(K)
                 ABCD = 1/ (AB + CD) = 1 / (expo(I)+expo(J)+expo(K)+expo(L))
                 AB * CD      (expo(I)+expo(J))*(expo(K)+expo(L))
                 Rou(Greek Letter) =   ----------- = ------------------------------------
                 AB + CD         expo(I)+expo(J)+expo(K)+expo(L)
                 
                 expo(I)+expo(J)                        expo(K)+expo(L)
                 ABcom = --------------------------------  CDcom = --------------------------------
                 expo(I)+expo(J)+expo(K)+expo(L)           expo(I)+expo(J)+expo(K)+expo(L)
                 
                 ABCDtemp = 1/2(expo(I)+expo(J)+expo(K)+expo(L))
                 */
                
                int kk_start = devSim.prim_start[KK];
                int ll_start = devSim.prim_start[LL];
                
                QUICKDouble CD = LOC2(devSim.expoSum, kk_start+KKK, ll_start+LLL, devSim.prim_total, devSim.prim_total);
                
                QUICKDouble ABCD = 1/(AB+CD);
                
                /*
                 X2 is the multiplication of four indices normalized coeffecient
                 */
                QUICKDouble X2 = sqrt(ABCD) * X1 * LOC4(devSim.Xcoeff, kStartK+KKK, kStartL+LLL, K - devSim.Qstart[KK], L - devSim.Qstart[LL], devSim.jbasis, devSim.jbasis, 2, 2);
                
                /*
                 Q' is the weighting center of K and L
                 --->           --->
                 ->  ------>       expo(K)*xyz(K)+expo(L)*xyz(L)
                 Q = P'(K,L)  = ------------------------------
                 expo(K) + expo(L)
                 
                 W' is the weight center for I, J, K, L
                 
                 --->             --->             --->            --->
                 ->     expo(I)*xyz(I) + expo(J)*xyz(J) + expo(K)*xyz(K) +expo(L)*xyz(L)
                 W = -------------------------------------------------------------------
                 expo(I) + expo(J) + expo(K) + expo(L)
                 ->  ->  2
                 RPQ =| P - Q |
                 
                 ->  -> 2
                 T = ROU * | P - Q|
                 */
                
                QUICKDouble Qx = LOC2(devSim.weightedCenterX, kk_start+KKK, ll_start+LLL, devSim.prim_total, devSim.prim_total);
                QUICKDouble Qy = LOC2(devSim.weightedCenterY, kk_start+KKK, ll_start+LLL, devSim.prim_total, devSim.prim_total);
                QUICKDouble Qz = LOC2(devSim.weightedCenterZ, kk_start+KKK, ll_start+LLL, devSim.prim_total, devSim.prim_total);
                
                //QUICKDouble T = AB * CD * ABCD * ( quick_dsqr(Px-Qx) + quick_dsqr(Py-Qy) + quick_dsqr(Pz-Qz));
                
                //QUICKDouble YVerticalTemp[VDIM1*VDIM2*VDIM3];
                FmT(I+J+K+L+2, AB * CD * ABCD * ( quick_dsqr(Px-Qx) + quick_dsqr(Py-Qy) + quick_dsqr(Pz-Qz)), YVerticalTemp);
                
                for (int i = 0; i<=I+J+K+L+2; i++) {
                    VY(0, 0, i) = VY(0, 0, i) * X2;
                }
                
                //QUICKDouble store2[STOREDIM*STOREDIM];
                
/*                
                for (int i = Sumindex[K]; i< Sumindex[K+L+3]; i++) {
                    for (int j = Sumindex[I]; j< Sumindex[I+J+3]; j++) {
                        if (i < STOREDIM && j < STOREDIM ) {
                            LOCSTORE(store2, j, i, STOREDIM, STOREDIM) = 0;
                        }
                    }
                }
*/                
                
#ifdef int_spdf
                ERint_grad_vertical_dddd_1(I, J+1, K, L+1, II, JJ, KK, LL, \
                         Px - RAx, Py - RAy, Pz - RAz, (Px*AB+Qx*CD)*ABCD - Px, (Py*AB+Qy*CD)*ABCD - Py, (Pz*AB+Qz*CD)*ABCD - Pz, \
                         Qx - RCx, Qy - RCy, Qz - RCz, (Px*AB+Qx*CD)*ABCD - Qx, (Py*AB+Qy*CD)*ABCD - Qy, (Pz*AB+Qz*CD)*ABCD - Qz, \
                         0.5 * ABCD, 0.5 / AB, 0.5 / CD, AB * ABCD, CD * ABCD, store2, YVerticalTemp);
#elif defined int_spdf2
                ERint_grad_vertical_dddd_2(I, J+1, K, L+1, II, JJ, KK, LL, \
                         Px - RAx, Py - RAy, Pz - RAz, (Px*AB+Qx*CD)*ABCD - Px, (Py*AB+Qy*CD)*ABCD - Py, (Pz*AB+Qz*CD)*ABCD - Pz, \
                         Qx - RCx, Qy - RCy, Qz - RCz, (Px*AB+Qx*CD)*ABCD - Qx, (Py*AB+Qy*CD)*ABCD - Qy, (Pz*AB+Qz*CD)*ABCD - Qz, \
                         0.5 * ABCD, 0.5 / AB, 0.5 / CD, AB * ABCD, CD * ABCD, store2, YVerticalTemp);
#elif defined int_spdf3

/*                for (int i = Sumindex[K]; i< Sumindex[K+L+3]; i++) {
                    for (int j = Sumindex[I]; j< Sumindex[I+J+3]; j++) {
                        if (i < STOREDIM && j < STOREDIM ) {
                            LOCSTORE(store2, j, i, STOREDIM, STOREDIM) = 0;
                        }
                    }
                }
*/

                ERint_vertical_spdf_1_2(I, J+1, K, L+1, II, JJ, KK, LL, \
                         Px - RAx, Py - RAy, Pz - RAz, (Px*AB+Qx*CD)*ABCD - Px, (Py*AB+Qy*CD)*ABCD - Py, (Pz*AB+Qz*CD)*ABCD - Pz, \
                         Qx - RCx, Qy - RCy, Qz - RCz, (Px*AB+Qx*CD)*ABCD - Qx, (Py*AB+Qy*CD)*ABCD - Qy, (Pz*AB+Qz*CD)*ABCD - Qz, \
                         0.5 * ABCD, 0.5 / AB, 0.5 / CD, AB * ABCD, CD * ABCD, store2, YVerticalTemp);
         
                ERint_vertical_spdf_2_2(I, J+1, K, L+1, II, JJ, KK, LL, \
                         Px - RAx, Py - RAy, Pz - RAz, (Px*AB+Qx*CD)*ABCD - Px, (Py*AB+Qy*CD)*ABCD - Py, (Pz*AB+Qz*CD)*ABCD - Pz, \
                         Qx - RCx, Qy - RCy, Qz - RCz, (Px*AB+Qx*CD)*ABCD - Qx, (Py*AB+Qy*CD)*ABCD - Qy, (Pz*AB+Qz*CD)*ABCD - Qz, \
                         0.5 * ABCD, 0.5 / AB, 0.5 / CD, AB * ABCD, CD * ABCD, store2, YVerticalTemp);

                ERint_vertical_spdf_3_2(I, J+1, K, L+1, II, JJ, KK, LL, \
                         Px - RAx, Py - RAy, Pz - RAz, (Px*AB+Qx*CD)*ABCD - Px, (Py*AB+Qy*CD)*ABCD - Py, (Pz*AB+Qz*CD)*ABCD - Pz, \
                         Qx - RCx, Qy - RCy, Qz - RCz, (Px*AB+Qx*CD)*ABCD - Qx, (Py*AB+Qy*CD)*ABCD - Qy, (Pz*AB+Qz*CD)*ABCD - Qz, \
                         0.5 * ABCD, 0.5 / AB, 0.5 / CD, AB * ABCD, CD * ABCD, store2, YVerticalTemp);

                ERint_vertical_spdf_4_2(I, J+1, K, L+1, II, JJ, KK, LL, \
                         Px - RAx, Py - RAy, Pz - RAz, (Px*AB+Qx*CD)*ABCD - Px, (Py*AB+Qy*CD)*ABCD - Py, (Pz*AB+Qz*CD)*ABCD - Pz, \
                         Qx - RCx, Qy - RCy, Qz - RCz, (Px*AB+Qx*CD)*ABCD - Qx, (Py*AB+Qy*CD)*ABCD - Qy, (Pz*AB+Qz*CD)*ABCD - Qz, \
                         0.5 * ABCD, 0.5 / AB, 0.5 / CD, AB * ABCD, CD * ABCD, store2, YVerticalTemp);

                ERint_vertical_spdf_5_2(I, J+1, K, L+1, II, JJ, KK, LL, \
                         Px - RAx, Py - RAy, Pz - RAz, (Px*AB+Qx*CD)*ABCD - Px, (Py*AB+Qy*CD)*ABCD - Py, (Pz*AB+Qz*CD)*ABCD - Pz, \
                         Qx - RCx, Qy - RCy, Qz - RCz, (Px*AB+Qx*CD)*ABCD - Qx, (Py*AB+Qy*CD)*ABCD - Qy, (Pz*AB+Qz*CD)*ABCD - Qz, \
                         0.5 * ABCD, 0.5 / AB, 0.5 / CD, AB * ABCD, CD * ABCD, store2, YVerticalTemp);

                ERint_vertical_spdf_6_2(I, J+1, K, L+1, II, JJ, KK, LL, \
                         Px - RAx, Py - RAy, Pz - RAz, (Px*AB+Qx*CD)*ABCD - Px, (Py*AB+Qy*CD)*ABCD - Py, (Pz*AB+Qz*CD)*ABCD - Pz, \
                         Qx - RCx, Qy - RCy, Qz - RCz, (Px*AB+Qx*CD)*ABCD - Qx, (Py*AB+Qy*CD)*ABCD - Qy, (Pz*AB+Qz*CD)*ABCD - Qz, \
                         0.5 * ABCD, 0.5 / AB, 0.5 / CD, AB * ABCD, CD * ABCD, store2, YVerticalTemp);

                ERint_vertical_spdf_7_2(I, J+1, K, L+1, II, JJ, KK, LL, \
                         Px - RAx, Py - RAy, Pz - RAz, (Px*AB+Qx*CD)*ABCD - Px, (Py*AB+Qy*CD)*ABCD - Py, (Pz*AB+Qz*CD)*ABCD - Pz, \
                         Qx - RCx, Qy - RCy, Qz - RCz, (Px*AB+Qx*CD)*ABCD - Qx, (Py*AB+Qy*CD)*ABCD - Qy, (Pz*AB+Qz*CD)*ABCD - Qz, \
                         0.5 * ABCD, 0.5 / AB, 0.5 / CD, AB * ABCD, CD * ABCD, store2, YVerticalTemp);

                ERint_vertical_spdf_8_2(I, J+1, K, L+1, II, JJ, KK, LL, \
                         Px - RAx, Py - RAy, Pz - RAz, (Px*AB+Qx*CD)*ABCD - Px, (Py*AB+Qy*CD)*ABCD - Py, (Pz*AB+Qz*CD)*ABCD - Pz, \
                         Qx - RCx, Qy - RCy, Qz - RCz, (Px*AB+Qx*CD)*ABCD - Qx, (Py*AB+Qy*CD)*ABCD - Qy, (Pz*AB+Qz*CD)*ABCD - Qz, \
                         0.5 * ABCD, 0.5 / AB, 0.5 / CD, AB * ABCD, CD * ABCD, store2, YVerticalTemp);

                ERint_grad_vertical_spd_2(I, J+1, K, L+1, II, JJ, KK, LL, \
                         Px - RAx, Py - RAy, Pz - RAz, (Px*AB+Qx*CD)*ABCD - Px, (Py*AB+Qy*CD)*ABCD - Py, (Pz*AB+Qz*CD)*ABCD - Pz, \
                         Qx - RCx, Qy - RCy, Qz - RCz, (Px*AB+Qx*CD)*ABCD - Qx, (Py*AB+Qy*CD)*ABCD - Qy, (Pz*AB+Qz*CD)*ABCD - Qz, \
                         0.5 * ABCD, 0.5 / AB, 0.5 / CD, AB * ABCD, CD * ABCD, store2, YVerticalTemp);

                ERint_grad_vertical_spdf_1(I, J+1, K, L+1, II, JJ, KK, LL, \
                         Px - RAx, Py - RAy, Pz - RAz, (Px*AB+Qx*CD)*ABCD - Px, (Py*AB+Qy*CD)*ABCD - Py, (Pz*AB+Qz*CD)*ABCD - Pz, \
                         Qx - RCx, Qy - RCy, Qz - RCz, (Px*AB+Qx*CD)*ABCD - Qx, (Py*AB+Qy*CD)*ABCD - Qy, (Pz*AB+Qz*CD)*ABCD - Qz, \
                         0.5 * ABCD, 0.5 / AB, 0.5 / CD, AB * ABCD, CD * ABCD, store2, YVerticalTemp);

                ERint_grad_vertical_spdf_2(I, J+1, K, L+1, II, JJ, KK, LL, \
                         Px - RAx, Py - RAy, Pz - RAz, (Px*AB+Qx*CD)*ABCD - Px, (Py*AB+Qy*CD)*ABCD - Py, (Pz*AB+Qz*CD)*ABCD - Pz, \
                         Qx - RCx, Qy - RCy, Qz - RCz, (Px*AB+Qx*CD)*ABCD - Qx, (Py*AB+Qy*CD)*ABCD - Qy, (Pz*AB+Qz*CD)*ABCD - Qz, \
                         0.5 * ABCD, 0.5 / AB, 0.5 / CD, AB * ABCD, CD * ABCD, store2, YVerticalTemp);

                ERint_grad_vertical_spdf_3(I, J+1, K, L+1, II, JJ, KK, LL, \
                         Px - RAx, Py - RAy, Pz - RAz, (Px*AB+Qx*CD)*ABCD - Px, (Py*AB+Qy*CD)*ABCD - Py, (Pz*AB+Qz*CD)*ABCD - Pz, \
                         Qx - RCx, Qy - RCy, Qz - RCz, (Px*AB+Qx*CD)*ABCD - Qx, (Py*AB+Qy*CD)*ABCD - Qy, (Pz*AB+Qz*CD)*ABCD - Qz, \
                         0.5 * ABCD, 0.5 / AB, 0.5 / CD, AB * ABCD, CD * ABCD, store2, YVerticalTemp);

                ERint_grad_vertical_spdf_4(I, J+1, K, L+1, II, JJ, KK, LL, \
                         Px - RAx, Py - RAy, Pz - RAz, (Px*AB+Qx*CD)*ABCD - Px, (Py*AB+Qy*CD)*ABCD - Py, (Pz*AB+Qz*CD)*ABCD - Pz, \
                         Qx - RCx, Qy - RCy, Qz - RCz, (Px*AB+Qx*CD)*ABCD - Qx, (Py*AB+Qy*CD)*ABCD - Qy, (Pz*AB+Qz*CD)*ABCD - Qz, \
                         0.5 * ABCD, 0.5 / AB, 0.5 / CD, AB * ABCD, CD * ABCD, store2, YVerticalTemp);

                ERint_grad_vertical_spdf_5(I, J+1, K, L+1, II, JJ, KK, LL, \
                         Px - RAx, Py - RAy, Pz - RAz, (Px*AB+Qx*CD)*ABCD - Px, (Py*AB+Qy*CD)*ABCD - Py, (Pz*AB+Qz*CD)*ABCD - Pz, \
                         Qx - RCx, Qy - RCy, Qz - RCz, (Px*AB+Qx*CD)*ABCD - Qx, (Py*AB+Qy*CD)*ABCD - Qy, (Pz*AB+Qz*CD)*ABCD - Qz, \
                         0.5 * ABCD, 0.5 / AB, 0.5 / CD, AB * ABCD, CD * ABCD, store2, YVerticalTemp);

                ERint_grad_vertical_spdf_6(I, J+1, K, L+1, II, JJ, KK, LL, \
                         Px - RAx, Py - RAy, Pz - RAz, (Px*AB+Qx*CD)*ABCD - Px, (Py*AB+Qy*CD)*ABCD - Py, (Pz*AB+Qz*CD)*ABCD - Pz, \
                         Qx - RCx, Qy - RCy, Qz - RCz, (Px*AB+Qx*CD)*ABCD - Qx, (Py*AB+Qy*CD)*ABCD - Qy, (Pz*AB+Qz*CD)*ABCD - Qz, \
                         0.5 * ABCD, 0.5 / AB, 0.5 / CD, AB * ABCD, CD * ABCD, store2, YVerticalTemp);
/*
                ERint_grad_vertical_spdf_7_1(I, J+1, K, L+1, II, JJ, KK, LL, \
                         Px - RAx, Py - RAy, Pz - RAz, (Px*AB+Qx*CD)*ABCD - Px, (Py*AB+Qy*CD)*ABCD - Py, (Pz*AB+Qz*CD)*ABCD - Pz, \
                         Qx - RCx, Qy - RCy, Qz - RCz, (Px*AB+Qx*CD)*ABCD - Qx, (Py*AB+Qy*CD)*ABCD - Qy, (Pz*AB+Qz*CD)*ABCD - Qz, \
                         0.5 * ABCD, 0.5 / AB, 0.5 / CD, AB * ABCD, CD * ABCD, store2, YVerticalTemp);

                ERint_grad_vertical_spdf_7_2(I, J+1, K, L+1, II, JJ, KK, LL, \
                         Px - RAx, Py - RAy, Pz - RAz, (Px*AB+Qx*CD)*ABCD - Px, (Py*AB+Qy*CD)*ABCD - Py, (Pz*AB+Qz*CD)*ABCD - Pz, \
                         Qx - RCx, Qy - RCy, Qz - RCz, (Px*AB+Qx*CD)*ABCD - Qx, (Py*AB+Qy*CD)*ABCD - Qy, (Pz*AB+Qz*CD)*ABCD - Qz, \
                         0.5 * ABCD, 0.5 / AB, 0.5 / CD, AB * ABCD, CD * ABCD, store2, YVerticalTemp);

                ERint_grad_vertical_spdf_7_3(I, J+1, K, L+1, II, JJ, KK, LL, \
                         Px - RAx, Py - RAy, Pz - RAz, (Px*AB+Qx*CD)*ABCD - Px, (Py*AB+Qy*CD)*ABCD - Py, (Pz*AB+Qz*CD)*ABCD - Pz, \
                         Qx - RCx, Qy - RCy, Qz - RCz, (Px*AB+Qx*CD)*ABCD - Qx, (Py*AB+Qy*CD)*ABCD - Qy, (Pz*AB+Qz*CD)*ABCD - Qz, \
                         0.5 * ABCD, 0.5 / AB, 0.5 / CD, AB * ABCD, CD * ABCD, store2, YVerticalTemp);

                vertical2_spdf3(I, J + 1, K, L + 1, YVerticalTemp, store2, \
                                Px - RAx, Py - RAy, Pz - RAz, (Px*AB+Qx*CD)*ABCD - Px, (Py*AB+Qy*CD)*ABCD - Py, (Pz*AB+Qz*CD)*ABCD - Pz, \
                                Qx - RCx, Qy - RCy, Qz - RCz, (Px*AB+Qx*CD)*ABCD - Qx, (Py*AB+Qy*CD)*ABCD - Qy, (Pz*AB+Qz*CD)*ABCD - Qz, \

                                0.5 * ABCD, 0.5 / AB, 0.5 / CD, AB * ABCD, CD * ABCD);
*/
#elif defined int_spdf4

		ERint_grad_vrr_ffff_1(I, J, K, L, II, JJ, KK, LL, 
                    Px - RAx, Py - RAy, Pz - RAz, (Px*AB+Qx*CD)*ABCD - Px, (Py*AB+Qy*CD)*ABCD - Py, (Pz*AB+Qz*CD)*ABCD - Pz, 
                    Qx - RCx, Qy - RCy, Qz - RCz, (Px*AB+Qx*CD)*ABCD - Qx, (Py*AB+Qy*CD)*ABCD - Qy, (Pz*AB+Qz*CD)*ABCD - Qz, 
                    0.5 * ABCD, 0.5 / AB, 0.5 / CD, AB * ABCD, CD * ABCD, store2, YVerticalTemp);


                ERint_grad_vrr_ffff_2(I, J, K, L, II, JJ, KK, LL, 
                    Px - RAx, Py - RAy, Pz - RAz, (Px*AB+Qx*CD)*ABCD - Px, (Py*AB+Qy*CD)*ABCD - Py, (Pz*AB+Qz*CD)*ABCD - Pz, 
                    Qx - RCx, Qy - RCy, Qz - RCz, (Px*AB+Qx*CD)*ABCD - Qx, (Py*AB+Qy*CD)*ABCD - Qy, (Pz*AB+Qz*CD)*ABCD - Qz, 
                    0.5 * ABCD, 0.5 / AB, 0.5 / CD, AB * ABCD, CD * ABCD, store2, YVerticalTemp);


                ERint_grad_vrr_ffff_3(I, J, K, L, II, JJ, KK, LL, 
                    Px - RAx, Py - RAy, Pz - RAz, (Px*AB+Qx*CD)*ABCD - Px, (Py*AB+Qy*CD)*ABCD - Py, (Pz*AB+Qz*CD)*ABCD - Pz, 
                    Qx - RCx, Qy - RCy, Qz - RCz, (Px*AB+Qx*CD)*ABCD - Qx, (Py*AB+Qy*CD)*ABCD - Qy, (Pz*AB+Qz*CD)*ABCD - Qz, 
                    0.5 * ABCD, 0.5 / AB, 0.5 / CD, AB * ABCD, CD * ABCD, store2, YVerticalTemp);


                ERint_grad_vrr_ffff_4(I, J, K, L, II, JJ, KK, LL, 
                    Px - RAx, Py - RAy, Pz - RAz, (Px*AB+Qx*CD)*ABCD - Px, (Py*AB+Qy*CD)*ABCD - Py, (Pz*AB+Qz*CD)*ABCD - Pz, 
                    Qx - RCx, Qy - RCy, Qz - RCz, (Px*AB+Qx*CD)*ABCD - Qx, (Py*AB+Qy*CD)*ABCD - Qy, (Pz*AB+Qz*CD)*ABCD - Qz, 
                    0.5 * ABCD, 0.5 / AB, 0.5 / CD, AB * ABCD, CD * ABCD, store2, YVerticalTemp);


                ERint_grad_vrr_ffff_5(I, J, K, L, II, JJ, KK, LL, 
                    Px - RAx, Py - RAy, Pz - RAz, (Px*AB+Qx*CD)*ABCD - Px, (Py*AB+Qy*CD)*ABCD - Py, (Pz*AB+Qz*CD)*ABCD - Pz, 
                    Qx - RCx, Qy - RCy, Qz - RCz, (Px*AB+Qx*CD)*ABCD - Qx, (Py*AB+Qy*CD)*ABCD - Qy, (Pz*AB+Qz*CD)*ABCD - Qz, 
                    0.5 * ABCD, 0.5 / AB, 0.5 / CD, AB * ABCD, CD * ABCD, store2, YVerticalTemp);


                ERint_grad_vrr_ffff_6(I, J, K, L, II, JJ, KK, LL, 
                    Px - RAx, Py - RAy, Pz - RAz, (Px*AB+Qx*CD)*ABCD - Px, (Py*AB+Qy*CD)*ABCD - Py, (Pz*AB+Qz*CD)*ABCD - Pz, 
                    Qx - RCx, Qy - RCy, Qz - RCz, (Px*AB+Qx*CD)*ABCD - Qx, (Py*AB+Qy*CD)*ABCD - Qy, (Pz*AB+Qz*CD)*ABCD - Qz, 
                    0.5 * ABCD, 0.5 / AB, 0.5 / CD, AB * ABCD, CD * ABCD, store2, YVerticalTemp);


                ERint_grad_vrr_ffff_7(I, J, K, L, II, JJ, KK, LL, 
                    Px - RAx, Py - RAy, Pz - RAz, (Px*AB+Qx*CD)*ABCD - Px, (Py*AB+Qy*CD)*ABCD - Py, (Pz*AB+Qz*CD)*ABCD - Pz, 
                    Qx - RCx, Qy - RCy, Qz - RCz, (Px*AB+Qx*CD)*ABCD - Qx, (Py*AB+Qy*CD)*ABCD - Qy, (Pz*AB+Qz*CD)*ABCD - Qz, 
                    0.5 * ABCD, 0.5 / AB, 0.5 / CD, AB * ABCD, CD * ABCD, store2, YVerticalTemp);


                ERint_grad_vrr_ffff_8(I, J, K, L, II, JJ, KK, LL, 
                    Px - RAx, Py - RAy, Pz - RAz, (Px*AB+Qx*CD)*ABCD - Px, (Py*AB+Qy*CD)*ABCD - Py, (Pz*AB+Qz*CD)*ABCD - Pz, 
                    Qx - RCx, Qy - RCy, Qz - RCz, (Px*AB+Qx*CD)*ABCD - Qx, (Py*AB+Qy*CD)*ABCD - Qy, (Pz*AB+Qz*CD)*ABCD - Qz, 
                    0.5 * ABCD, 0.5 / AB, 0.5 / CD, AB * ABCD, CD * ABCD, store2, YVerticalTemp);


                ERint_grad_vrr_ffff_9(I, J, K, L, II, JJ, KK, LL, 
                    Px - RAx, Py - RAy, Pz - RAz, (Px*AB+Qx*CD)*ABCD - Px, (Py*AB+Qy*CD)*ABCD - Py, (Pz*AB+Qz*CD)*ABCD - Pz, 
                    Qx - RCx, Qy - RCy, Qz - RCz, (Px*AB+Qx*CD)*ABCD - Qx, (Py*AB+Qy*CD)*ABCD - Qy, (Pz*AB+Qz*CD)*ABCD - Qz, 
                    0.5 * ABCD, 0.5 / AB, 0.5 / CD, AB * ABCD, CD * ABCD, store2, YVerticalTemp);


                ERint_grad_vrr_ffff_10(I, J, K, L, II, JJ, KK, LL, 
                    Px - RAx, Py - RAy, Pz - RAz, (Px*AB+Qx*CD)*ABCD - Px, (Py*AB+Qy*CD)*ABCD - Py, (Pz*AB+Qz*CD)*ABCD - Pz, 
                    Qx - RCx, Qy - RCy, Qz - RCz, (Px*AB+Qx*CD)*ABCD - Qx, (Py*AB+Qy*CD)*ABCD - Qy, (Pz*AB+Qz*CD)*ABCD - Qz, 
                    0.5 * ABCD, 0.5 / AB, 0.5 / CD, AB * ABCD, CD * ABCD, store2, YVerticalTemp);


                ERint_grad_vrr_ffff_11(I, J, K, L, II, JJ, KK, LL, 
                    Px - RAx, Py - RAy, Pz - RAz, (Px*AB+Qx*CD)*ABCD - Px, (Py*AB+Qy*CD)*ABCD - Py, (Pz*AB+Qz*CD)*ABCD - Pz, 
                    Qx - RCx, Qy - RCy, Qz - RCz, (Px*AB+Qx*CD)*ABCD - Qx, (Py*AB+Qy*CD)*ABCD - Qy, (Pz*AB+Qz*CD)*ABCD - Qz, 
                    0.5 * ABCD, 0.5 / AB, 0.5 / CD, AB * ABCD, CD * ABCD, store2, YVerticalTemp);


                ERint_grad_vrr_ffff_12(I, J, K, L, II, JJ, KK, LL, 
                    Px - RAx, Py - RAy, Pz - RAz, (Px*AB+Qx*CD)*ABCD - Px, (Py*AB+Qy*CD)*ABCD - Py, (Pz*AB+Qz*CD)*ABCD - Pz, 
                    Qx - RCx, Qy - RCy, Qz - RCz, (Px*AB+Qx*CD)*ABCD - Qx, (Py*AB+Qy*CD)*ABCD - Qy, (Pz*AB+Qz*CD)*ABCD - Qz, 
                    0.5 * ABCD, 0.5 / AB, 0.5 / CD, AB * ABCD, CD * ABCD, store2, YVerticalTemp);


                ERint_grad_vrr_ffff_13(I, J, K, L, II, JJ, KK, LL, 
                    Px - RAx, Py - RAy, Pz - RAz, (Px*AB+Qx*CD)*ABCD - Px, (Py*AB+Qy*CD)*ABCD - Py, (Pz*AB+Qz*CD)*ABCD - Pz, 
                    Qx - RCx, Qy - RCy, Qz - RCz, (Px*AB+Qx*CD)*ABCD - Qx, (Py*AB+Qy*CD)*ABCD - Qy, (Pz*AB+Qz*CD)*ABCD - Qz, 
                    0.5 * ABCD, 0.5 / AB, 0.5 / CD, AB * ABCD, CD * ABCD, store2, YVerticalTemp);


                ERint_grad_vrr_ffff_14(I, J, K, L, II, JJ, KK, LL, 
                    Px - RAx, Py - RAy, Pz - RAz, (Px*AB+Qx*CD)*ABCD - Px, (Py*AB+Qy*CD)*ABCD - Py, (Pz*AB+Qz*CD)*ABCD - Pz, 
                    Qx - RCx, Qy - RCy, Qz - RCz, (Px*AB+Qx*CD)*ABCD - Qx, (Py*AB+Qy*CD)*ABCD - Qy, (Pz*AB+Qz*CD)*ABCD - Qz, 
                    0.5 * ABCD, 0.5 / AB, 0.5 / CD, AB * ABCD, CD * ABCD, store2, YVerticalTemp);


                ERint_grad_vrr_ffff_15(I, J, K, L, II, JJ, KK, LL, 
                    Px - RAx, Py - RAy, Pz - RAz, (Px*AB+Qx*CD)*ABCD - Px, (Py*AB+Qy*CD)*ABCD - Py, (Pz*AB+Qz*CD)*ABCD - Pz, 
                    Qx - RCx, Qy - RCy, Qz - RCz, (Px*AB+Qx*CD)*ABCD - Qx, (Py*AB+Qy*CD)*ABCD - Qy, (Pz*AB+Qz*CD)*ABCD - Qz, 
                    0.5 * ABCD, 0.5 / AB, 0.5 / CD, AB * ABCD, CD * ABCD, store2, YVerticalTemp);


                ERint_grad_vrr_ffff_16(I, J, K, L, II, JJ, KK, LL, 
                    Px - RAx, Py - RAy, Pz - RAz, (Px*AB+Qx*CD)*ABCD - Px, (Py*AB+Qy*CD)*ABCD - Py, (Pz*AB+Qz*CD)*ABCD - Pz, 
                    Qx - RCx, Qy - RCy, Qz - RCz, (Px*AB+Qx*CD)*ABCD - Qx, (Py*AB+Qy*CD)*ABCD - Qy, (Pz*AB+Qz*CD)*ABCD - Qz, 
                    0.5 * ABCD, 0.5 / AB, 0.5 / CD, AB * ABCD, CD * ABCD, store2, YVerticalTemp);


                ERint_grad_vrr_ffff_17(I, J, K, L, II, JJ, KK, LL, 
                    Px - RAx, Py - RAy, Pz - RAz, (Px*AB+Qx*CD)*ABCD - Px, (Py*AB+Qy*CD)*ABCD - Py, (Pz*AB+Qz*CD)*ABCD - Pz, 
                    Qx - RCx, Qy - RCy, Qz - RCz, (Px*AB+Qx*CD)*ABCD - Qx, (Py*AB+Qy*CD)*ABCD - Qy, (Pz*AB+Qz*CD)*ABCD - Qz, 
                    0.5 * ABCD, 0.5 / AB, 0.5 / CD, AB * ABCD, CD * ABCD, store2, YVerticalTemp);


                ERint_grad_vrr_ffff_18(I, J, K, L, II, JJ, KK, LL, 
                    Px - RAx, Py - RAy, Pz - RAz, (Px*AB+Qx*CD)*ABCD - Px, (Py*AB+Qy*CD)*ABCD - Py, (Pz*AB+Qz*CD)*ABCD - Pz, 
                    Qx - RCx, Qy - RCy, Qz - RCz, (Px*AB+Qx*CD)*ABCD - Qx, (Py*AB+Qy*CD)*ABCD - Qy, (Pz*AB+Qz*CD)*ABCD - Qz, 
                    0.5 * ABCD, 0.5 / AB, 0.5 / CD, AB * ABCD, CD * ABCD, store2, YVerticalTemp);


                ERint_grad_vrr_ffff_19(I, J, K, L, II, JJ, KK, LL, 
                    Px - RAx, Py - RAy, Pz - RAz, (Px*AB+Qx*CD)*ABCD - Px, (Py*AB+Qy*CD)*ABCD - Py, (Pz*AB+Qz*CD)*ABCD - Pz, 
                    Qx - RCx, Qy - RCy, Qz - RCz, (Px*AB+Qx*CD)*ABCD - Qx, (Py*AB+Qy*CD)*ABCD - Qy, (Pz*AB+Qz*CD)*ABCD - Qz, 
                    0.5 * ABCD, 0.5 / AB, 0.5 / CD, AB * ABCD, CD * ABCD, store2, YVerticalTemp);


                ERint_grad_vrr_ffff_20(I, J, K, L, II, JJ, KK, LL, 
                    Px - RAx, Py - RAy, Pz - RAz, (Px*AB+Qx*CD)*ABCD - Px, (Py*AB+Qy*CD)*ABCD - Py, (Pz*AB+Qz*CD)*ABCD - Pz, 
                    Qx - RCx, Qy - RCy, Qz - RCz, (Px*AB+Qx*CD)*ABCD - Qx, (Py*AB+Qy*CD)*ABCD - Qy, (Pz*AB+Qz*CD)*ABCD - Qz, 
                    0.5 * ABCD, 0.5 / AB, 0.5 / CD, AB * ABCD, CD * ABCD, store2, YVerticalTemp);


                ERint_grad_vrr_ffff_21(I, J, K, L, II, JJ, KK, LL, 
                    Px - RAx, Py - RAy, Pz - RAz, (Px*AB+Qx*CD)*ABCD - Px, (Py*AB+Qy*CD)*ABCD - Py, (Pz*AB+Qz*CD)*ABCD - Pz, 
                    Qx - RCx, Qy - RCy, Qz - RCz, (Px*AB+Qx*CD)*ABCD - Qx, (Py*AB+Qy*CD)*ABCD - Qy, (Pz*AB+Qz*CD)*ABCD - Qz, 
                    0.5 * ABCD, 0.5 / AB, 0.5 / CD, AB * ABCD, CD * ABCD, store2, YVerticalTemp);


                ERint_grad_vrr_ffff_22(I, J, K, L, II, JJ, KK, LL, 
                    Px - RAx, Py - RAy, Pz - RAz, (Px*AB+Qx*CD)*ABCD - Px, (Py*AB+Qy*CD)*ABCD - Py, (Pz*AB+Qz*CD)*ABCD - Pz, 
                    Qx - RCx, Qy - RCy, Qz - RCz, (Px*AB+Qx*CD)*ABCD - Qx, (Py*AB+Qy*CD)*ABCD - Qy, (Pz*AB+Qz*CD)*ABCD - Qz, 
                    0.5 * ABCD, 0.5 / AB, 0.5 / CD, AB * ABCD, CD * ABCD, store2, YVerticalTemp);


                ERint_grad_vrr_ffff_23(I, J, K, L, II, JJ, KK, LL, 
                    Px - RAx, Py - RAy, Pz - RAz, (Px*AB+Qx*CD)*ABCD - Px, (Py*AB+Qy*CD)*ABCD - Py, (Pz*AB+Qz*CD)*ABCD - Pz, 
                    Qx - RCx, Qy - RCy, Qz - RCz, (Px*AB+Qx*CD)*ABCD - Qx, (Py*AB+Qy*CD)*ABCD - Qy, (Pz*AB+Qz*CD)*ABCD - Qz, 
                    0.5 * ABCD, 0.5 / AB, 0.5 / CD, AB * ABCD, CD * ABCD, store2, YVerticalTemp);


                ERint_grad_vrr_ffff_24(I, J, K, L, II, JJ, KK, LL, 
                    Px - RAx, Py - RAy, Pz - RAz, (Px*AB+Qx*CD)*ABCD - Px, (Py*AB+Qy*CD)*ABCD - Py, (Pz*AB+Qz*CD)*ABCD - Pz, 
                    Qx - RCx, Qy - RCy, Qz - RCz, (Px*AB+Qx*CD)*ABCD - Qx, (Py*AB+Qy*CD)*ABCD - Qy, (Pz*AB+Qz*CD)*ABCD - Qz, 
                    0.5 * ABCD, 0.5 / AB, 0.5 / CD, AB * ABCD, CD * ABCD, store2, YVerticalTemp);


                ERint_grad_vrr_ffff_25(I, J, K, L, II, JJ, KK, LL, 
                    Px - RAx, Py - RAy, Pz - RAz, (Px*AB+Qx*CD)*ABCD - Px, (Py*AB+Qy*CD)*ABCD - Py, (Pz*AB+Qz*CD)*ABCD - Pz, 
                    Qx - RCx, Qy - RCy, Qz - RCz, (Px*AB+Qx*CD)*ABCD - Qx, (Py*AB+Qy*CD)*ABCD - Qy, (Pz*AB+Qz*CD)*ABCD - Qz, 
                    0.5 * ABCD, 0.5 / AB, 0.5 / CD, AB * ABCD, CD * ABCD, store2, YVerticalTemp);


                ERint_grad_vrr_ffff_26(I, J, K, L, II, JJ, KK, LL, 
                    Px - RAx, Py - RAy, Pz - RAz, (Px*AB+Qx*CD)*ABCD - Px, (Py*AB+Qy*CD)*ABCD - Py, (Pz*AB+Qz*CD)*ABCD - Pz, 
                    Qx - RCx, Qy - RCy, Qz - RCz, (Px*AB+Qx*CD)*ABCD - Qx, (Py*AB+Qy*CD)*ABCD - Qy, (Pz*AB+Qz*CD)*ABCD - Qz, 
                    0.5 * ABCD, 0.5 / AB, 0.5 / CD, AB * ABCD, CD * ABCD, store2, YVerticalTemp);


                ERint_grad_vrr_ffff_27(I, J, K, L, II, JJ, KK, LL, 
                    Px - RAx, Py - RAy, Pz - RAz, (Px*AB+Qx*CD)*ABCD - Px, (Py*AB+Qy*CD)*ABCD - Py, (Pz*AB+Qz*CD)*ABCD - Pz, 
                    Qx - RCx, Qy - RCy, Qz - RCz, (Px*AB+Qx*CD)*ABCD - Qx, (Py*AB+Qy*CD)*ABCD - Qy, (Pz*AB+Qz*CD)*ABCD - Qz, 
                    0.5 * ABCD, 0.5 / AB, 0.5 / CD, AB * ABCD, CD * ABCD, store2, YVerticalTemp);


                ERint_grad_vrr_ffff_28(I, J, K, L, II, JJ, KK, LL, 
                    Px - RAx, Py - RAy, Pz - RAz, (Px*AB+Qx*CD)*ABCD - Px, (Py*AB+Qy*CD)*ABCD - Py, (Pz*AB+Qz*CD)*ABCD - Pz, 
                    Qx - RCx, Qy - RCy, Qz - RCz, (Px*AB+Qx*CD)*ABCD - Qx, (Py*AB+Qy*CD)*ABCD - Qy, (Pz*AB+Qz*CD)*ABCD - Qz, 
                    0.5 * ABCD, 0.5 / AB, 0.5 / CD, AB * ABCD, CD * ABCD, store2, YVerticalTemp);


                ERint_grad_vrr_ffff_29(I, J, K, L, II, JJ, KK, LL, 
                    Px - RAx, Py - RAy, Pz - RAz, (Px*AB+Qx*CD)*ABCD - Px, (Py*AB+Qy*CD)*ABCD - Py, (Pz*AB+Qz*CD)*ABCD - Pz, 
                    Qx - RCx, Qy - RCy, Qz - RCz, (Px*AB+Qx*CD)*ABCD - Qx, (Py*AB+Qy*CD)*ABCD - Qy, (Pz*AB+Qz*CD)*ABCD - Qz, 
                    0.5 * ABCD, 0.5 / AB, 0.5 / CD, AB * ABCD, CD * ABCD, store2, YVerticalTemp);


                ERint_grad_vrr_ffff_30(I, J, K, L, II, JJ, KK, LL, 
                    Px - RAx, Py - RAy, Pz - RAz, (Px*AB+Qx*CD)*ABCD - Px, (Py*AB+Qy*CD)*ABCD - Py, (Pz*AB+Qz*CD)*ABCD - Pz, 
                    Qx - RCx, Qy - RCy, Qz - RCz, (Px*AB+Qx*CD)*ABCD - Qx, (Py*AB+Qy*CD)*ABCD - Qy, (Pz*AB+Qz*CD)*ABCD - Qz, 
                    0.5 * ABCD, 0.5 / AB, 0.5 / CD, AB * ABCD, CD * ABCD, store2, YVerticalTemp);


                ERint_grad_vrr_ffff_31(I, J, K, L, II, JJ, KK, LL, 
                    Px - RAx, Py - RAy, Pz - RAz, (Px*AB+Qx*CD)*ABCD - Px, (Py*AB+Qy*CD)*ABCD - Py, (Pz*AB+Qz*CD)*ABCD - Pz, 
                    Qx - RCx, Qy - RCy, Qz - RCz, (Px*AB+Qx*CD)*ABCD - Qx, (Py*AB+Qy*CD)*ABCD - Qy, (Pz*AB+Qz*CD)*ABCD - Qz, 
                    0.5 * ABCD, 0.5 / AB, 0.5 / CD, AB * ABCD, CD * ABCD, store2, YVerticalTemp);


                ERint_grad_vrr_ffff_32(I, J, K, L, II, JJ, KK, LL, 
                    Px - RAx, Py - RAy, Pz - RAz, (Px*AB+Qx*CD)*ABCD - Px, (Py*AB+Qy*CD)*ABCD - Py, (Pz*AB+Qz*CD)*ABCD - Pz, 
                    Qx - RCx, Qy - RCy, Qz - RCz, (Px*AB+Qx*CD)*ABCD - Qx, (Py*AB+Qy*CD)*ABCD - Qy, (Pz*AB+Qz*CD)*ABCD - Qz, 
                    0.5 * ABCD, 0.5 / AB, 0.5 / CD, AB * ABCD, CD * ABCD, store2, YVerticalTemp);



/*
                ERint_grad_vertical_spdf_5(I, J+1, K, L+1, II, JJ, KK, LL, \
                         Px - RAx, Py - RAy, Pz - RAz, (Px*AB+Qx*CD)*ABCD - Px, (Py*AB+Qy*CD)*ABCD - Py, (Pz*AB+Qz*CD)*ABCD - Pz, \
                         Qx - RCx, Qy - RCy, Qz - RCz, (Px*AB+Qx*CD)*ABCD - Qx, (Py*AB+Qy*CD)*ABCD - Qy, (Pz*AB+Qz*CD)*ABCD - Qz, \
                         0.5 * ABCD, 0.5 / AB, 0.5 / CD, AB * ABCD, CD * ABCD, store2, YVerticalTemp);

                ERint_vertical_spdf_7_2(I, J+1, K, L+1, II, JJ, KK, LL, \
                         Px - RAx, Py - RAy, Pz - RAz, (Px*AB+Qx*CD)*ABCD - Px, (Py*AB+Qy*CD)*ABCD - Py, (Pz*AB+Qz*CD)*ABCD - Pz, \
                         Qx - RCx, Qy - RCy, Qz - RCz, (Px*AB+Qx*CD)*ABCD - Qx, (Py*AB+Qy*CD)*ABCD - Qy, (Pz*AB+Qz*CD)*ABCD - Qz, \
                         0.5 * ABCD, 0.5 / AB, 0.5 / CD, AB * ABCD, CD * ABCD, store2, YVerticalTemp);

                ERint_grad_vertical_spdf_7_1(I, J+1, K, L+1, II, JJ, KK, LL, \
                         Px - RAx, Py - RAy, Pz - RAz, (Px*AB+Qx*CD)*ABCD - Px, (Py*AB+Qy*CD)*ABCD - Py, (Pz*AB+Qz*CD)*ABCD - Pz, \
                         Qx - RCx, Qy - RCy, Qz - RCz, (Px*AB+Qx*CD)*ABCD - Qx, (Py*AB+Qy*CD)*ABCD - Qy, (Pz*AB+Qz*CD)*ABCD - Qz, \
                         0.5 * ABCD, 0.5 / AB, 0.5 / CD, AB * ABCD, CD * ABCD, store2, YVerticalTemp);

                ERint_grad_vertical_spdf_7_2(I, J+1, K, L+1, II, JJ, KK, LL, \
                         Px - RAx, Py - RAy, Pz - RAz, (Px*AB+Qx*CD)*ABCD - Px, (Py*AB+Qy*CD)*ABCD - Py, (Pz*AB+Qz*CD)*ABCD - Pz, \
                         Qx - RCx, Qy - RCy, Qz - RCz, (Px*AB+Qx*CD)*ABCD - Qx, (Py*AB+Qy*CD)*ABCD - Qy, (Pz*AB+Qz*CD)*ABCD - Qz, \
                         0.5 * ABCD, 0.5 / AB, 0.5 / CD, AB * ABCD, CD * ABCD, store2, YVerticalTemp);

                ERint_grad_vertical_spdf_7_3(I, J+1, K, L+1, II, JJ, KK, LL, \
                         Px - RAx, Py - RAy, Pz - RAz, (Px*AB+Qx*CD)*ABCD - Px, (Py*AB+Qy*CD)*ABCD - Py, (Pz*AB+Qz*CD)*ABCD - Pz, \
                         Qx - RCx, Qy - RCy, Qz - RCz, (Px*AB+Qx*CD)*ABCD - Qx, (Py*AB+Qy*CD)*ABCD - Qy, (Pz*AB+Qz*CD)*ABCD - Qz, \
                         0.5 * ABCD, 0.5 / AB, 0.5 / CD, AB * ABCD, CD * ABCD, store2, YVerticalTemp);


                vertical2_spdf4(I, J + 1, K, L + 1, YVerticalTemp, store2, \
                                Px - RAx, Py - RAy, Pz - RAz, (Px*AB+Qx*CD)*ABCD - Px, (Py*AB+Qy*CD)*ABCD - Py, (Pz*AB+Qz*CD)*ABCD - Pz, \
                                Qx - RCx, Qy - RCy, Qz - RCz, (Px*AB+Qx*CD)*ABCD - Qx, (Py*AB+Qy*CD)*ABCD - Qy, (Pz*AB+Qz*CD)*ABCD - Qz, \
                                0.5 * ABCD, 0.5 / AB, 0.5 / CD, AB * ABCD, CD * ABCD);
#elif defined int_spdf5
                vertical2_spdf5(I, J + 1, K, L + 1, YVerticalTemp, store2, \
                                Px - RAx, Py - RAy, Pz - RAz, (Px*AB+Qx*CD)*ABCD - Px, (Py*AB+Qy*CD)*ABCD - Py, (Pz*AB+Qz*CD)*ABCD - Pz, \
                                Qx - RCx, Qy - RCy, Qz - RCz, (Px*AB+Qx*CD)*ABCD - Qx, (Py*AB+Qy*CD)*ABCD - Qy, (Pz*AB+Qz*CD)*ABCD - Qz, \
                                0.5 * ABCD, 0.5 / AB, 0.5 / CD, AB * ABCD, CD * ABCD);
#elif defined int_spdf6
                vertical2_spdf6(I, J + 1, K, L + 1, YVerticalTemp, store2, \
                                Px - RAx, Py - RAy, Pz - RAz, (Px*AB+Qx*CD)*ABCD - Px, (Py*AB+Qy*CD)*ABCD - Py, (Pz*AB+Qz*CD)*ABCD - Pz, \
                                Qx - RCx, Qy - RCy, Qz - RCz, (Px*AB+Qx*CD)*ABCD - Qx, (Py*AB+Qy*CD)*ABCD - Qy, (Pz*AB+Qz*CD)*ABCD - Qz, \
                                0.5 * ABCD, 0.5 / AB, 0.5 / CD, AB * ABCD, CD * ABCD);
#elif defined int_spdf7
                vertical2_spdf7(I, J + 1, K, L + 1, YVerticalTemp, store2, \
                                Px - RAx, Py - RAy, Pz - RAz, (Px*AB+Qx*CD)*ABCD - Px, (Py*AB+Qy*CD)*ABCD - Py, (Pz*AB+Qz*CD)*ABCD - Pz, \
                                Qx - RCx, Qy - RCy, Qz - RCz, (Px*AB+Qx*CD)*ABCD - Qx, (Py*AB+Qy*CD)*ABCD - Qy, (Pz*AB+Qz*CD)*ABCD - Qz, \
                                0.5 * ABCD, 0.5 / AB, 0.5 / CD, AB * ABCD, CD * ABCD);
#elif defined int_spdf8
                vertical2_spdf8(I, J + 1, K, L + 1, YVerticalTemp, store2, \
                                Px - RAx, Py - RAy, Pz - RAz, (Px*AB+Qx*CD)*ABCD - Px, (Py*AB+Qy*CD)*ABCD - Py, (Pz*AB+Qz*CD)*ABCD - Pz, \
                                Qx - RCx, Qy - RCy, Qz - RCz, (Px*AB+Qx*CD)*ABCD - Qx, (Py*AB+Qy*CD)*ABCD - Qy, (Pz*AB+Qz*CD)*ABCD - Qz, \
                                0.5 * ABCD, 0.5 / AB, 0.5 / CD, AB * ABCD, CD * ABCD);
*/
#endif
  

                for (int i = Sumindex[K]; i< Sumindex[K+L+2]; i++) {
                    for (int j = Sumindex[I]; j< Sumindex[I+J+2]; j++) {
                        if (i < STOREDIM && j < STOREDIM) {
                            LOCSTORE(store, j, i , STOREDIM, STOREDIM) += LOCSTORE(store2, j, i, STOREDIM, STOREDIM);
                        }
                    }
                }


                for (int i = Sumindex[K]; i< Sumindex[K+L+2]; i++) {
                    for (int j = Sumindex[I+1]; j< Sumindex[I+J+3]; j++) {
                        if (i < STOREDIM && j < STOREDIM) {
                            LOCSTORE(storeAA, j, i, STOREDIM, STOREDIM) += LOCSTORE(store2, j, i, STOREDIM, STOREDIM) * AA * 2 ;
                            LOCSTORE(storeBB, j, i, STOREDIM, STOREDIM) += LOCSTORE(store2, j, i, STOREDIM, STOREDIM) * BB * 2 ;
                        }

                    }
                }

                for (int i = Sumindex[K+1]; i< Sumindex[K+L+3]; i++) {
                    for (int j = Sumindex[I]; j< Sumindex[I+J+2]; j++) {
                        if (i < STOREDIM && j < STOREDIM) {
                            LOCSTORE(storeCC, j, i, STOREDIM, STOREDIM) += LOCSTORE(store2, j, i, STOREDIM, STOREDIM) * CC * 2 ;

                        }
                    }
                }              
                
                
            }
        }
    }
 
/*
                for (int i = Sumindex[K]; i< Sumindex[K+L+2]; i++) {
                    for (int j = Sumindex[I]; j< Sumindex[I+J+2]; j++) {
                        if (i < STOREDIM && j < STOREDIM) {
                            printf("STORE   %d   %d   %d   %d   %d   %d   %d   %d   %d   %d   %.9f \n",II, JJ, KK, LL, I, J, K, L, j, i, LOCSTORE(store, j, i , STOREDIM, STOREDIM));
                        }
                    }
                }    
*/
    QUICKDouble AGradx = 0.0;
    QUICKDouble AGrady = 0.0;
    QUICKDouble AGradz = 0.0;
    QUICKDouble BGradx = 0.0;
    QUICKDouble BGrady = 0.0;
    QUICKDouble BGradz = 0.0;
    QUICKDouble CGradx = 0.0;
    QUICKDouble CGrady = 0.0;
    QUICKDouble CGradz = 0.0;

    int         AStart = (devSim.katom[II]-1) * 3;
    int         BStart = (devSim.katom[JJ]-1) * 3;
    int         CStart = (devSim.katom[KK]-1) * 3;
    int         DStart = (devSim.katom[LL]-1) * 3;
           
                QUICKDouble RBx, RBy, RBz;
                QUICKDouble RDx, RDy, RDz;
                
                RBx = LOC2(devSim.xyz, 0 , devSim.katom[JJ]-1, 3, devSim.natom);
                RBy = LOC2(devSim.xyz, 1 , devSim.katom[JJ]-1, 3, devSim.natom);
                RBz = LOC2(devSim.xyz, 2 , devSim.katom[JJ]-1, 3, devSim.natom);
                
                
                RDx = LOC2(devSim.xyz, 0 , devSim.katom[LL]-1, 3, devSim.natom);
                RDy = LOC2(devSim.xyz, 1 , devSim.katom[LL]-1, 3, devSim.natom);
                RDz = LOC2(devSim.xyz, 2 , devSim.katom[LL]-1, 3, devSim.natom);
                
                int III1 = LOC2(devSim.Qsbasis, II, I, devSim.nshell, 4);
                int III2 = LOC2(devSim.Qfbasis, II, I, devSim.nshell, 4);
                int JJJ1 = LOC2(devSim.Qsbasis, JJ, J, devSim.nshell, 4);
                int JJJ2 = LOC2(devSim.Qfbasis, JJ, J, devSim.nshell, 4);
                int KKK1 = LOC2(devSim.Qsbasis, KK, K, devSim.nshell, 4);
                int KKK2 = LOC2(devSim.Qfbasis, KK, K, devSim.nshell, 4);
                int LLL1 = LOC2(devSim.Qsbasis, LL, L, devSim.nshell, 4);
                int LLL2 = LOC2(devSim.Qfbasis, LL, L, devSim.nshell, 4);
                
                
                int  IJKLTYPE = 999;
                
                int  nbasis = devSim.nbasis;
                
                for (int III = III1; III <= III2; III++) {
                    for (int JJJ = MAX(III,JJJ1); JJJ <= JJJ2; JJJ++) {
                        for (int KKK = MAX(III,KKK1); KKK <= KKK2; KKK++) {
                            for (int LLL = MAX(KKK,LLL1); LLL <= LLL2; LLL++) {
                                
                                if (III < KKK ||
                                    ((III == JJJ) && (III == LLL)) ||
                                    ((III == JJJ) && (III  < LLL)) ||
                                    ((JJJ == LLL) && (III  < JJJ)) ||
                                    ((III == KKK) && (III  < JJJ)  && (JJJ < LLL))) {
                                    
                                    QUICKDouble Yaax, Yaay, Yaaz;
                                    QUICKDouble Ybbx, Ybby, Ybbz;
                                    QUICKDouble Yccx, Yccy, Yccz;


#ifdef  int_spdf
                                    hrrwholegrad2_1
#elif defined int_spdf2

                                    hrrwholegrad2_2
#else


                                    hrrwholegrad2
#endif
                                    (&Yaax, &Yaay, &Yaaz, \
                                                  &Ybbx, &Ybby, &Ybbz, \
                                                  &Yccx, &Yccy, &Yccz, \
                                                  I, J, K, L,\
                                                  III, JJJ, KKK, LLL, IJKLTYPE, \
                                                  store, storeAA, storeBB, storeCC, \
                                                  RAx, RAy, RAz, RBx, RBy, RBz, \
                                                  RCx, RCy, RCz, RDx, RDy, RDz);

                                    QUICKDouble constant = 0.0 ;
                                    
#ifdef OSHELL
                                    QUICKDouble DENSELJ = (QUICKDouble) (LOC2(devSim.dense, LLL-1, JJJ-1, nbasis, nbasis)+LOC2(devSim.denseb, LLL-1, JJJ-1, nbasis, nbasis));
                                    QUICKDouble DENSELI = (QUICKDouble) (LOC2(devSim.dense, LLL-1, III-1, nbasis, nbasis)+LOC2(devSim.denseb, LLL-1, III-1, nbasis, nbasis));
                                    QUICKDouble DENSELK = (QUICKDouble) (LOC2(devSim.dense, LLL-1, KKK-1, nbasis, nbasis)+LOC2(devSim.denseb, LLL-1, KKK-1, nbasis, nbasis));
                                    QUICKDouble DENSEJI = (QUICKDouble) (LOC2(devSim.dense, JJJ-1, III-1, nbasis, nbasis)+LOC2(devSim.denseb, JJJ-1, III-1, nbasis, nbasis));

                                    QUICKDouble DENSEKIA = (QUICKDouble) LOC2(devSim.dense, KKK-1, III-1, nbasis, nbasis);
                                    QUICKDouble DENSEKJA = (QUICKDouble) LOC2(devSim.dense, KKK-1, JJJ-1, nbasis, nbasis);
                                    QUICKDouble DENSELJA = (QUICKDouble) LOC2(devSim.dense, LLL-1, JJJ-1, nbasis, nbasis);
                                    QUICKDouble DENSELIA = (QUICKDouble) LOC2(devSim.dense, LLL-1, III-1, nbasis, nbasis);
                                    QUICKDouble DENSEJIA = (QUICKDouble) LOC2(devSim.dense, JJJ-1, III-1, nbasis, nbasis);

                                    QUICKDouble DENSEKIB = (QUICKDouble) LOC2(devSim.denseb, KKK-1, III-1, nbasis, nbasis);
                                    QUICKDouble DENSEKJB = (QUICKDouble) LOC2(devSim.denseb, KKK-1, JJJ-1, nbasis, nbasis);
                                    QUICKDouble DENSELJB = (QUICKDouble) LOC2(devSim.denseb, LLL-1, JJJ-1, nbasis, nbasis);
                                    QUICKDouble DENSELIB = (QUICKDouble) LOC2(devSim.denseb, LLL-1, III-1, nbasis, nbasis);
                                    QUICKDouble DENSEJIB = (QUICKDouble) LOC2(devSim.denseb, JJJ-1, III-1, nbasis, nbasis);

#else
                                    QUICKDouble DENSEKI = (QUICKDouble) LOC2(devSim.dense, KKK-1, III-1, nbasis, nbasis);
                                    QUICKDouble DENSEKJ = (QUICKDouble) LOC2(devSim.dense, KKK-1, JJJ-1, nbasis, nbasis);
                                    QUICKDouble DENSELJ = (QUICKDouble) LOC2(devSim.dense, LLL-1, JJJ-1, nbasis, nbasis);
                                    QUICKDouble DENSELI = (QUICKDouble) LOC2(devSim.dense, LLL-1, III-1, nbasis, nbasis);
                                    QUICKDouble DENSELK = (QUICKDouble) LOC2(devSim.dense, LLL-1, KKK-1, nbasis, nbasis);
                                    QUICKDouble DENSEJI = (QUICKDouble) LOC2(devSim.dense, JJJ-1, III-1, nbasis, nbasis);
#endif
                                    
                                    if (II < JJ && II < KK && KK < LL ||
                                        ( III < KKK && III < JJJ && KKK < LLL)) {
                                        //constant = ( 4.0 * DENSEJI * DENSELK - DENSEKI * DENSELJ - DENSELI * DENSEKJ);
#ifdef OSHELL
					constant = ( 4.0 * DENSEJI * DENSELK - 2.0 * devSim.hyb_coeff * DENSEKIA * DENSELJA - 2.0 * devSim.hyb_coeff * DENSELIA * DENSEKJA
                                                   - 2.0 * devSim.hyb_coeff * DENSEKIB * DENSELJB - 2.0 * devSim.hyb_coeff * DENSELIB * DENSEKJB);
#else
                                        constant = ( 4.0 * DENSEJI * DENSELK - devSim.hyb_coeff * DENSEKI * DENSELJ - devSim.hyb_coeff * DENSELI * DENSEKJ);
#endif
                                    }else{
                                        if (III < KKK) {
                                            if( III == JJJ && KKK == LLL){
                                                //constant = (DENSEJI * DENSELK - 0.5 * DENSEKI * DENSEKI);
#ifdef OSHELL
                                                constant = (DENSEJI * DENSELK - devSim.hyb_coeff * DENSEKIA * DENSEKIA - devSim.hyb_coeff * DENSEKIB * DENSEKIB);
#else
                                                constant = (DENSEJI * DENSELK - 0.5 * devSim.hyb_coeff * DENSEKI * DENSEKI);
#endif

                                            }else if (JJJ == KKK && JJJ == LLL){
                                                //constant = DENSELJ * DENSEJI;
#ifdef OSHELL
                                                constant = 2.0 * DENSELJ * DENSEJI - 2.0 * devSim.hyb_coeff * DENSELJA * DENSEJIA - 2.0 * devSim.hyb_coeff * DENSELJB * DENSEJIB;
#else
                                                constant = 2.0 * DENSELJ * DENSEJI - devSim.hyb_coeff * DENSELJ * DENSEJI;
#endif
                                            }else if (KKK == LLL && III < JJJ && JJJ != KKK){
                                                //constant = (2.0* DENSEJI * DENSELK - DENSEKI * DENSEKJ);
#ifdef OSHELL
                                                constant = (2.0* DENSEJI * DENSELK - 2.0 * devSim.hyb_coeff * DENSEKIA * DENSEKJA - 2.0 * devSim.hyb_coeff * DENSEKIB * DENSEKJB);
#else
                                                constant = (2.0* DENSEJI * DENSELK - devSim.hyb_coeff * DENSEKI * DENSEKJ);
#endif
                                            }else if ( III == JJJ && KKK < LLL){
                                                //constant = (2.0* DENSELK * DENSEJI - DENSEKI * DENSELI);
#ifdef OSHELL
                                                constant = (2.0* DENSELK * DENSEJI - 2.0 * devSim.hyb_coeff * DENSEKIA * DENSELIA - 2.0 * devSim.hyb_coeff * DENSEKIB * DENSELIB);
#else
                                                constant = (2.0* DENSELK * DENSEJI - devSim.hyb_coeff * DENSEKI * DENSELI);
#endif
                                            }
                                        }
                                        else{
                                            if (JJJ <= LLL) {
                                                if (III == JJJ && III == KKK && III == LLL) {
                                                    // Do nothing
                                                }else if (III==JJJ && III==KKK && III < LLL){
                                                    //constant = DENSELI * DENSEJI;
#ifdef OSHELL
                                                    constant = 2.0 * DENSELI * DENSEJI - 2.0 * devSim.hyb_coeff * DENSELIA * DENSEJIA - 2.0 * devSim.hyb_coeff * DENSELIB * DENSEJIB;
#else
						    constant = 2.0 * DENSELI * DENSEJI - devSim.hyb_coeff * DENSELI * DENSEJI;
#endif
                                                }else if (III==KKK && JJJ==LLL && III < JJJ){
                                                    //constant = (1.5 * DENSEJI * DENSEJI - 0.5 * DENSELJ * DENSEKI);
#ifdef OSHELL
                                                    constant = (2.0 * DENSEJI * DENSEJI - devSim.hyb_coeff * DENSEJIA * DENSEJIA - devSim.hyb_coeff * DENSELJA * DENSEKIA
                                                               - devSim.hyb_coeff * DENSEJIB * DENSEJIB - devSim.hyb_coeff * DENSELJB * DENSEKIB);
#else
                                                    constant = (2.0 * DENSEJI * DENSEJI - 0.5 * devSim.hyb_coeff * DENSEJI * DENSEJI - 0.5 * devSim.hyb_coeff * DENSELJ * DENSEKI);
#endif

                                                }else if (III== KKK && III < JJJ && JJJ < LLL){
                                                    //constant = (3.0 * DENSEJI * DENSELI - DENSELJ * DENSEKI);
#ifdef OSHELL
                                                    constant = (4.0 * DENSEJI * DENSELI - 2.0 * devSim.hyb_coeff * DENSEJIA * DENSELIA - 2.0 * devSim.hyb_coeff * DENSELJA * DENSEKIA
                                                               - 2.0 * devSim.hyb_coeff * DENSEJIB * DENSELIB - 2.0 * devSim.hyb_coeff * DENSELJB * DENSEKIB);
#else
                                                    constant = (4.0 * DENSEJI * DENSELI - devSim.hyb_coeff * DENSEJI * DENSELI - devSim.hyb_coeff * DENSELJ * DENSEKI);
#endif
                                                }
                                            }
                                        }
                                    }
                                    
                                    AGradx += constant * Yaax;
                                    AGrady += constant * Yaay;
                                    AGradz += constant * Yaaz;
                                    
                                    BGradx += constant * Ybbx;
                                    BGrady += constant * Ybby;
                                    BGradz += constant * Ybbz;
                                    
                                    CGradx += constant * Yccx;
                                    CGrady += constant * Yccy;
                                    CGradz += constant * Yccz;
                                    
                                    
                                    
                                }
                            }
                        }
                    }
                }
                
                
                
                /*
                if ( abs(AGradx) > 0 || abs(AGrady) > 0 || abs(AGradz) > 0 ||
                    abs(BGradx) > 0 || abs(BGrady) > 0 || abs(BGradz) > 0 ||
                    abs(CGradx) > 0 || abs(CGrady) > 0 || abs(CGradz) > 0) {
                    
                    printf("%i %i %i %i %i %i %i %i %20.10e %20.10e %20.10e %20.10e %20.10e %20.10e %20.10e %20.10e %20.10e \n", II, JJ, KK, LL, \
                           I, J, K, L, AGradx, AGrady, AGradz, BGradx, BGrady, BGradz, CGradx, CGrady, CGradz);
                }*/
                
/*                
                
            }
        }
    }
*/    
    
#ifdef DEBUG
    //printf("FILE: %s, LINE: %d, FUNCTION: %s, devSim.hyb_coeff \n", __FILE__, __LINE__, __func__);
#endif    
   
#ifdef USE_LEGACY_ATOMICS 
    GRADADD(devSim.gradULL[AStart], AGradx);
    GRADADD(devSim.gradULL[AStart + 1], AGrady);
    GRADADD(devSim.gradULL[AStart + 2], AGradz);
    
    
    GRADADD(devSim.gradULL[BStart], BGradx);
    GRADADD(devSim.gradULL[BStart + 1], BGrady);
    GRADADD(devSim.gradULL[BStart + 2], BGradz);
    
    
    GRADADD(devSim.gradULL[CStart], CGradx);
    GRADADD(devSim.gradULL[CStart + 1], CGrady);
    GRADADD(devSim.gradULL[CStart + 2], CGradz);
    
    
    GRADADD(devSim.gradULL[DStart], (-AGradx-BGradx-CGradx));
    GRADADD(devSim.gradULL[DStart + 1], (-AGrady-BGrady-CGrady));
    GRADADD(devSim.gradULL[DStart + 2], (-AGradz-BGradz-CGradz));
#else 
    atomicAdd(&devSim.grad[AStart], AGradx);
    atomicAdd(&devSim.grad[AStart + 1], AGrady);
    atomicAdd(&devSim.grad[AStart + 2], AGradz);


    atomicAdd(&devSim.grad[BStart], BGradx);
    atomicAdd(&devSim.grad[BStart + 1], BGrady);
    atomicAdd(&devSim.grad[BStart + 2], BGradz);


    atomicAdd(&devSim.grad[CStart], CGradx);
    atomicAdd(&devSim.grad[CStart + 1], CGrady);
    atomicAdd(&devSim.grad[CStart + 2], CGradz);


    atomicAdd(&devSim.grad[DStart], (-AGradx-BGradx-CGradx));
    atomicAdd(&devSim.grad[DStart + 1], (-AGrady-BGrady-CGrady));
    atomicAdd(&devSim.grad[DStart + 2], (-AGradz-BGradz-CGradz));   
#endif

    return;
}

#endif


#ifndef new_quick_2_gpu_get2e_subs_grad_h
#define new_quick_2_gpu_get2e_subs_grad_h

#undef STOREDIM
#define STOREDIM STOREDIM_GRAD_T

__device__ __forceinline__ void hrrwholegrad_sp(QUICKDouble* Yaax, QUICKDouble* Yaay, QUICKDouble* Yaaz, \
                                             QUICKDouble* Ybbx, QUICKDouble* Ybby, QUICKDouble* Ybbz, \
                                             QUICKDouble* Yccx, QUICKDouble* Yccy, QUICKDouble* Yccz, \
                                             const int I, const int J, const int K, const int L, \
                                             const int III, int JJJ, const int KKK, const int LLL, const int IJKLTYPE,
                                             QUICKDouble* store, QUICKDouble* storeAA, QUICKDouble* storeBB, QUICKDouble* storeCC, \
                                             const QUICKDouble RAx, const QUICKDouble RAy, const QUICKDouble RAz, \
                                             const QUICKDouble RBx, const QUICKDouble RBy, const QUICKDouble RBz, \
                                             const QUICKDouble RCx, const QUICKDouble RCy, const QUICKDouble RCz, \
                                             const QUICKDouble RDx, const QUICKDouble RDy, const QUICKDouble RDz)
{
    unsigned char angularL[4], angularR[4];
    QUICKDouble coefAngularL[4], coefAngularR[4];

    *Yaax = 0.0;
    *Yaay = 0.0;
    *Yaaz = 0.0;
    *Ybbx = 0.0;
    *Ybby = 0.0;
    *Ybbz = 0.0;
    *Yccx = 0.0;
    *Yccy = 0.0;
    *Yccz = 0.0;

    QUICKDouble constant = devSim.cons[III-1] * devSim.cons[JJJ-1] * devSim.cons[KKK-1] * devSim.cons[LLL-1];
    int numAngularL, numAngularR;

    numAngularR = lefthrr_sp(RCx, RCy, RCz, RDx, RDy, RDz, \
                          LOC2(devSim.KLMN,0,KKK-1,3,devSim.nbasis), LOC2(devSim.KLMN,1,KKK-1,3,devSim.nbasis), LOC2(devSim.KLMN,2,KKK-1,3,devSim.nbasis), \
                          LOC2(devSim.KLMN,0,LLL-1,3,devSim.nbasis), LOC2(devSim.KLMN,1,LLL-1,3,devSim.nbasis), LOC2(devSim.KLMN,2,LLL-1,3,devSim.nbasis), \
                          L, coefAngularR, angularR);


    //  Part A - x
    numAngularL = lefthrr_sp(RAx, RAy, RAz, RBx, RBy, RBz, \
                          LOC2(devSim.KLMN,0,III-1,3,devSim.nbasis) + 1, LOC2(devSim.KLMN,1,III-1,3,devSim.nbasis), LOC2(devSim.KLMN,2,III-1,3,devSim.nbasis), \
                          LOC2(devSim.KLMN,0,JJJ-1,3,devSim.nbasis), LOC2(devSim.KLMN,1,JJJ-1,3,devSim.nbasis), LOC2(devSim.KLMN,2,JJJ-1,3,devSim.nbasis), \
                          J, coefAngularL, angularL);
    for (int i = 0; i<numAngularL; i++) {
        for (int j = 0; j<numAngularR; j++) {
            if (angularL[i] <= STOREDIM && angularR[j] <= STOREDIM) {
                *Yaax = *Yaax + coefAngularL[i] * coefAngularR[j] * LOCSTORE(storeAA, angularL[i]-1, angularR[j]-1, STOREDIM, STOREDIM);
            }
        }
    }
    
    numAngularL = lefthrr_sp(RAx, RAy, RAz, RBx, RBy, RBz, \
                          LOC2(devSim.KLMN,0,III-1,3,devSim.nbasis), LOC2(devSim.KLMN,1,III-1,3,devSim.nbasis) + 1, LOC2(devSim.KLMN,2,III-1,3,devSim.nbasis), \
                          LOC2(devSim.KLMN,0,JJJ-1,3,devSim.nbasis), LOC2(devSim.KLMN,1,JJJ-1,3,devSim.nbasis), LOC2(devSim.KLMN,2,JJJ-1,3,devSim.nbasis), \
                          J, coefAngularL, angularL);
    for (int i = 0; i<numAngularL; i++) {
        for (int j = 0; j<numAngularR; j++) {
            if (angularL[i] <= STOREDIM && angularR[j] <= STOREDIM) {
                *Yaay = *Yaay + coefAngularL[i] * coefAngularR[j] * LOCSTORE(storeAA, angularL[i]-1, angularR[j]-1, STOREDIM, STOREDIM);
            }
        }
    }
    
    numAngularL = lefthrr_sp(RAx, RAy, RAz, RBx, RBy, RBz, \
                          LOC2(devSim.KLMN,0,III-1,3,devSim.nbasis), LOC2(devSim.KLMN,1,III-1,3,devSim.nbasis), LOC2(devSim.KLMN,2,III-1,3,devSim.nbasis) + 1, \
                          LOC2(devSim.KLMN,0,JJJ-1,3,devSim.nbasis), LOC2(devSim.KLMN,1,JJJ-1,3,devSim.nbasis), LOC2(devSim.KLMN,2,JJJ-1,3,devSim.nbasis), \
                          J, coefAngularL, angularL);
    for (int i = 0; i<numAngularL; i++) {
        for (int j = 0; j<numAngularR; j++) {
            if (angularL[i] <= STOREDIM && angularR[j] <= STOREDIM) {
                *Yaaz = *Yaaz + coefAngularL[i] * coefAngularR[j] * LOCSTORE(storeAA, angularL[i]-1, angularR[j]-1, STOREDIM, STOREDIM);
            }
        }
    }
    
    numAngularL = lefthrr_spd(RAx, RAy, RAz, RBx, RBy, RBz, \
                          LOC2(devSim.KLMN,0,III-1,3,devSim.nbasis), LOC2(devSim.KLMN,1,III-1,3,devSim.nbasis), LOC2(devSim.KLMN,2,III-1,3,devSim.nbasis), \
                          LOC2(devSim.KLMN,0,JJJ-1,3,devSim.nbasis) + 1, LOC2(devSim.KLMN,1,JJJ-1,3,devSim.nbasis), LOC2(devSim.KLMN,2,JJJ-1,3,devSim.nbasis), \
                          J + 1, coefAngularL, angularL);
    for (int i = 0; i<numAngularL; i++) {
        for (int j = 0; j<numAngularR; j++) {
            if (angularL[i] <= STOREDIM && angularR[j] <= STOREDIM) {
                *Ybbx = *Ybbx + coefAngularL[i] * coefAngularR[j] * LOCSTORE(storeBB, angularL[i]-1, angularR[j]-1, STOREDIM, STOREDIM);
            }
        }
    }
    
    numAngularL = lefthrr_spd(RAx, RAy, RAz, RBx, RBy, RBz, \
                          LOC2(devSim.KLMN,0,III-1,3,devSim.nbasis), LOC2(devSim.KLMN,1,III-1,3,devSim.nbasis), LOC2(devSim.KLMN,2,III-1,3,devSim.nbasis), \
                          LOC2(devSim.KLMN,0,JJJ-1,3,devSim.nbasis), LOC2(devSim.KLMN,1,JJJ-1,3,devSim.nbasis) + 1, LOC2(devSim.KLMN,2,JJJ-1,3,devSim.nbasis), \
                          J + 1, coefAngularL, angularL);
    for (int i = 0; i<numAngularL; i++) {
        for (int j = 0; j<numAngularR; j++) {
            if (angularL[i] <= STOREDIM && angularR[j] <= STOREDIM) {
                *Ybby = *Ybby + coefAngularL[i] * coefAngularR[j] * LOCSTORE(storeBB, angularL[i]-1, angularR[j]-1, STOREDIM, STOREDIM);
            }
        }
    }
    
    numAngularL = lefthrr_spd(RAx, RAy, RAz, RBx, RBy, RBz, \
                          LOC2(devSim.KLMN,0,III-1,3,devSim.nbasis), LOC2(devSim.KLMN,1,III-1,3,devSim.nbasis), LOC2(devSim.KLMN,2,III-1,3,devSim.nbasis), \
                          LOC2(devSim.KLMN,0,JJJ-1,3,devSim.nbasis), LOC2(devSim.KLMN,1,JJJ-1,3,devSim.nbasis), LOC2(devSim.KLMN,2,JJJ-1,3,devSim.nbasis) + 1, \
                          J + 1, coefAngularL, angularL);
    for (int i = 0; i<numAngularL; i++) {
        for (int j = 0; j<numAngularR; j++) {
            if (angularL[i] <= STOREDIM && angularR[j] <= STOREDIM) {
                *Ybbz = *Ybbz + coefAngularL[i] * coefAngularR[j] * LOCSTORE(storeBB, angularL[i]-1, angularR[j]-1, STOREDIM, STOREDIM);
            }
        }
    }
    
    
    
    
    if (LOC2(devSim.KLMN,0,III-1,3,devSim.nbasis) >= 1) {
        
        numAngularL = lefthrr_sp(RAx, RAy, RAz, RBx, RBy, RBz,
                              LOC2(devSim.KLMN,0,III-1,3,devSim.nbasis) - 1, LOC2(devSim.KLMN,1,III-1,3,devSim.nbasis), LOC2(devSim.KLMN,2,III-1,3,devSim.nbasis),
                              LOC2(devSim.KLMN,0,JJJ-1,3,devSim.nbasis), LOC2(devSim.KLMN,1,JJJ-1,3,devSim.nbasis), LOC2(devSim.KLMN,2,JJJ-1,3,devSim.nbasis),
                              J, coefAngularL, angularL);
        for (int i = 0; i<numAngularL; i++) {
            for (int j = 0; j<numAngularR; j++) {
                
                if (angularL[i] <= STOREDIM && angularR[j] <= STOREDIM) {
                    *Yaax = *Yaax - LOC2(devSim.KLMN,0,III-1,3,devSim.nbasis) * coefAngularL[i] * coefAngularR[j] * LOCSTORE(store, angularL[i]-1, angularR[j]-1, STOREDIM, STOREDIM);
                }
            }
        }
        
    }
    
    if (LOC2(devSim.KLMN,1,III-1,3,devSim.nbasis) >= 1) {
        
        numAngularL = lefthrr_sp(RAx, RAy, RAz, RBx, RBy, RBz,
                              LOC2(devSim.KLMN,0,III-1,3,devSim.nbasis), LOC2(devSim.KLMN,1,III-1,3,devSim.nbasis) - 1, LOC2(devSim.KLMN,2,III-1,3,devSim.nbasis),
                              LOC2(devSim.KLMN,0,JJJ-1,3,devSim.nbasis), LOC2(devSim.KLMN,1,JJJ-1,3,devSim.nbasis), LOC2(devSim.KLMN,2,JJJ-1,3,devSim.nbasis),
                              J, coefAngularL, angularL);
        for (int i = 0; i<numAngularL; i++) {
            for (int j = 0; j<numAngularR; j++) {
                
                if (angularL[i] <= STOREDIM && angularR[j] <= STOREDIM) {
                    *Yaay = *Yaay - LOC2(devSim.KLMN,1,III-1,3,devSim.nbasis) * coefAngularL[i] * coefAngularR[j] * LOCSTORE(store, angularL[i]-1, angularR[j]-1, STOREDIM, STOREDIM);
                }
            }
        }
    }
    
    if (LOC2(devSim.KLMN,2,III-1,3,devSim.nbasis) >= 1) {
        
        numAngularL = lefthrr_sp(RAx, RAy, RAz, RBx, RBy, RBz,
                              LOC2(devSim.KLMN,0,III-1,3,devSim.nbasis), LOC2(devSim.KLMN,1,III-1,3,devSim.nbasis), LOC2(devSim.KLMN,2,III-1,3,devSim.nbasis) - 1,
                              LOC2(devSim.KLMN,0,JJJ-1,3,devSim.nbasis), LOC2(devSim.KLMN,1,JJJ-1,3,devSim.nbasis), LOC2(devSim.KLMN,2,JJJ-1,3,devSim.nbasis),
                              J, coefAngularL, angularL);
        for (int i = 0; i<numAngularL; i++) {
            for (int j = 0; j<numAngularR; j++) {
                
                if (angularL[i] <= STOREDIM && angularR[j] <= STOREDIM) {
                    *Yaaz = *Yaaz - LOC2(devSim.KLMN,2,III-1,3,devSim.nbasis) * coefAngularL[i] * coefAngularR[j] * LOCSTORE(store, angularL[i]-1, angularR[j]-1, STOREDIM, STOREDIM);
                }
            }
        }
    }
    
    
    
    
    if (LOC2(devSim.KLMN,0,JJJ-1,3,devSim.nbasis) >= 1) {
        
        numAngularL = lefthrr_sp(RAx, RAy, RAz, RBx, RBy, RBz,
                              LOC2(devSim.KLMN,0,III-1,3,devSim.nbasis), LOC2(devSim.KLMN,1,III-1,3,devSim.nbasis), LOC2(devSim.KLMN,2,III-1,3,devSim.nbasis),
                              LOC2(devSim.KLMN,0,JJJ-1,3,devSim.nbasis) - 1, LOC2(devSim.KLMN,1,JJJ-1,3,devSim.nbasis), LOC2(devSim.KLMN,2,JJJ-1,3,devSim.nbasis),
                              J - 1, coefAngularL, angularL);
        for (int i = 0; i<numAngularL; i++) {
            for (int j = 0; j<numAngularR; j++) {
                
                if (angularL[i] <= STOREDIM && angularR[j] <= STOREDIM) {
                    *Ybbx = *Ybbx - LOC2(devSim.KLMN,0,JJJ-1,3,devSim.nbasis) * coefAngularL[i] * coefAngularR[j] * LOCSTORE(store, angularL[i]-1, angularR[j]-1, STOREDIM, STOREDIM);
                }
            }
        }
    }
    
    
    
    if (LOC2(devSim.KLMN,1,JJJ-1,3,devSim.nbasis) >= 1) {
        
        numAngularL = lefthrr_sp(RAx, RAy, RAz, RBx, RBy, RBz,
                              LOC2(devSim.KLMN,0,III-1,3,devSim.nbasis), LOC2(devSim.KLMN,1,III-1,3,devSim.nbasis), LOC2(devSim.KLMN,2,III-1,3,devSim.nbasis),
                              LOC2(devSim.KLMN,0,JJJ-1,3,devSim.nbasis), LOC2(devSim.KLMN,1,JJJ-1,3,devSim.nbasis) - 1, LOC2(devSim.KLMN,2,JJJ-1,3,devSim.nbasis),
                              J - 1, coefAngularL, angularL);
        for (int i = 0; i<numAngularL; i++) {
            for (int j = 0; j<numAngularR; j++) {
                
                if (angularL[i] <= STOREDIM && angularR[j] <= STOREDIM) {
                    *Ybby = *Ybby - LOC2(devSim.KLMN,1,JJJ-1,3,devSim.nbasis) * coefAngularL[i] * coefAngularR[j] * LOCSTORE(store, angularL[i]-1, angularR[j]-1, STOREDIM, STOREDIM);
                }
            }
        }
    }
    
    
    if (LOC2(devSim.KLMN,2,JJJ-1,3,devSim.nbasis) >= 1) {
        
        numAngularL = lefthrr_sp(RAx, RAy, RAz, RBx, RBy, RBz,
                              LOC2(devSim.KLMN,0,III-1,3,devSim.nbasis), LOC2(devSim.KLMN,1,III-1,3,devSim.nbasis), LOC2(devSim.KLMN,2,III-1,3,devSim.nbasis),
                              LOC2(devSim.KLMN,0,JJJ-1,3,devSim.nbasis), LOC2(devSim.KLMN,1,JJJ-1,3,devSim.nbasis), LOC2(devSim.KLMN,2,JJJ-1,3,devSim.nbasis) - 1,
                              J - 1, coefAngularL, angularL);
        for (int i = 0; i<numAngularL; i++) {
            for (int j = 0; j<numAngularR; j++) {
                
                if (angularL[i] <= STOREDIM && angularR[j] <= STOREDIM) {
                    *Ybbz = *Ybbz - LOC2(devSim.KLMN,2,JJJ-1,3,devSim.nbasis) * coefAngularL[i] * coefAngularR[j] * LOCSTORE(store, angularL[i]-1, angularR[j]-1, STOREDIM, STOREDIM);
                }
            }
        }
    }
    
    
    
    // KET PART =====================================
    
    // Part C - x
    
    
    numAngularL = lefthrr_sp(RAx, RAy, RAz, RBx, RBy, RBz,
                          LOC2(devSim.KLMN,0,III-1,3,devSim.nbasis), LOC2(devSim.KLMN,1,III-1,3,devSim.nbasis), LOC2(devSim.KLMN,2,III-1,3,devSim.nbasis),
                          LOC2(devSim.KLMN,0,JJJ-1,3,devSim.nbasis), LOC2(devSim.KLMN,1,JJJ-1,3,devSim.nbasis), LOC2(devSim.KLMN,2,JJJ-1,3,devSim.nbasis),
                          J, coefAngularL, angularL);
    
    numAngularR = lefthrr_sp(RCx, RCy, RCz, RDx, RDy, RDz,
                          LOC2(devSim.KLMN,0,KKK-1,3,devSim.nbasis) + 1, LOC2(devSim.KLMN,1,KKK-1,3,devSim.nbasis), LOC2(devSim.KLMN,2,KKK-1,3,devSim.nbasis),
                          LOC2(devSim.KLMN,0,LLL-1,3,devSim.nbasis), LOC2(devSim.KLMN,1,LLL-1,3,devSim.nbasis), LOC2(devSim.KLMN,2,LLL-1,3,devSim.nbasis),
                          L, coefAngularR, angularR);
    
    
    for (int i = 0; i<numAngularL; i++) {
        for (int j = 0; j<numAngularR; j++) {
            
            if (angularL[i] <= STOREDIM && angularR[j] <= STOREDIM) {
                *Yccx = *Yccx + coefAngularL[i] * coefAngularR[j] * LOCSTORE(storeCC, angularL[i]-1, angularR[j]-1, STOREDIM, STOREDIM);
            }
        }
    }
    
    if (LOC2(devSim.KLMN,0,KKK-1,3,devSim.nbasis) >= 1) {
        
        numAngularR = lefthrr_sp(RCx, RCy, RCz, RDx, RDy, RDz,
                              LOC2(devSim.KLMN,0,KKK-1,3,devSim.nbasis) - 1, LOC2(devSim.KLMN,1,KKK-1,3,devSim.nbasis), LOC2(devSim.KLMN,2,KKK-1,3,devSim.nbasis),
                              LOC2(devSim.KLMN,0,LLL-1,3,devSim.nbasis), LOC2(devSim.KLMN,1,LLL-1,3,devSim.nbasis), LOC2(devSim.KLMN,2,LLL-1,3,devSim.nbasis),
                              L, coefAngularR, angularR);
        
        for (int i = 0; i<numAngularL; i++) {
            for (int j = 0; j<numAngularR; j++) {
                if (angularL[i] <= STOREDIM && angularR[j] <= STOREDIM) {
                    *Yccx = *Yccx - LOC2(devSim.KLMN,0,KKK-1,3,devSim.nbasis) * coefAngularL[i] * coefAngularR[j] * LOCSTORE(store, angularL[i]-1, angularR[j]-1, STOREDIM, STOREDIM);
                }
            }
        }
    }
    
    
    // Part C - y
    
    numAngularR = lefthrr_sp(RCx, RCy, RCz, RDx, RDy, RDz,
                          LOC2(devSim.KLMN,0,KKK-1,3,devSim.nbasis), LOC2(devSim.KLMN,1,KKK-1,3,devSim.nbasis) + 1, LOC2(devSim.KLMN,2,KKK-1,3,devSim.nbasis),
                          LOC2(devSim.KLMN,0,LLL-1,3,devSim.nbasis), LOC2(devSim.KLMN,1,LLL-1,3,devSim.nbasis), LOC2(devSim.KLMN,2,LLL-1,3,devSim.nbasis),
                          L, coefAngularR, angularR);
    
    
    for (int i = 0; i<numAngularL; i++) {
        for (int j = 0; j<numAngularR; j++) {
            
            if (angularL[i] <= STOREDIM && angularR[j] <= STOREDIM) {
                *Yccy = *Yccy + coefAngularL[i] * coefAngularR[j] * LOCSTORE(storeCC, angularL[i]-1, angularR[j]-1, STOREDIM, STOREDIM);
            }
        }
    }
    
    if (LOC2(devSim.KLMN,1,KKK-1,3,devSim.nbasis) >= 1) {
        
        numAngularR = lefthrr_sp(RCx, RCy, RCz, RDx, RDy, RDz,
                              LOC2(devSim.KLMN,0,KKK-1,3,devSim.nbasis), LOC2(devSim.KLMN,1,KKK-1,3,devSim.nbasis) - 1, LOC2(devSim.KLMN,2,KKK-1,3,devSim.nbasis),
                              LOC2(devSim.KLMN,0,LLL-1,3,devSim.nbasis), LOC2(devSim.KLMN,1,LLL-1,3,devSim.nbasis), LOC2(devSim.KLMN,2,LLL-1,3,devSim.nbasis),
                              L, coefAngularR, angularR);
        
        for (int i = 0; i<numAngularL; i++) {
            for (int j = 0; j<numAngularR; j++) {
                
                if (angularL[i] <= STOREDIM && angularR[j] <= STOREDIM) {
                    *Yccy = *Yccy - LOC2(devSim.KLMN,1,KKK-1,3,devSim.nbasis) * coefAngularL[i] * coefAngularR[j] * LOCSTORE(store, angularL[i]-1, angularR[j]-1, STOREDIM, STOREDIM);
                }
            }
        }
    }
    
    // Part C - z
    
    numAngularR = lefthrr_sp(RCx, RCy, RCz, RDx, RDy, RDz,
                          LOC2(devSim.KLMN,0,KKK-1,3,devSim.nbasis), LOC2(devSim.KLMN,1,KKK-1,3,devSim.nbasis), LOC2(devSim.KLMN,2,KKK-1,3,devSim.nbasis) + 1,
                          LOC2(devSim.KLMN,0,LLL-1,3,devSim.nbasis), LOC2(devSim.KLMN,1,LLL-1,3,devSim.nbasis), LOC2(devSim.KLMN,2,LLL-1,3,devSim.nbasis),
                          L, coefAngularR, angularR);
    
    
    for (int i = 0; i<numAngularL; i++) {
        for (int j = 0; j<numAngularR; j++) {
            
            if (angularL[i] <= STOREDIM && angularR[j] <= STOREDIM) {
                *Yccz = *Yccz + coefAngularL[i] * coefAngularR[j] * LOCSTORE(storeCC, angularL[i]-1, angularR[j]-1, STOREDIM, STOREDIM);
            }
        }
    }
    
    if (LOC2(devSim.KLMN,2,KKK-1,3,devSim.nbasis) >= 1) {
        
        numAngularR = lefthrr_sp(RCx, RCy, RCz, RDx, RDy, RDz,
                              LOC2(devSim.KLMN,0,KKK-1,3,devSim.nbasis), LOC2(devSim.KLMN,1,KKK-1,3,devSim.nbasis), LOC2(devSim.KLMN,2,KKK-1,3,devSim.nbasis) - 1,
                              LOC2(devSim.KLMN,0,LLL-1,3,devSim.nbasis), LOC2(devSim.KLMN,1,LLL-1,3,devSim.nbasis), LOC2(devSim.KLMN,2,LLL-1,3,devSim.nbasis),
                              L, coefAngularR, angularR);
        
        for (int i = 0; i<numAngularL; i++) {
            for (int j = 0; j<numAngularR; j++) {
                
                if (angularL[i] <= STOREDIM && angularR[j] <= STOREDIM) {
                    *Yccz = *Yccz - LOC2(devSim.KLMN,2,KKK-1,3,devSim.nbasis) * coefAngularL[i] * coefAngularR[j] * LOCSTORE(store, angularL[i]-1, angularR[j]-1, STOREDIM, STOREDIM);
                }
            }
        }
    }
    
    
    *Yaax = *Yaax * constant;
    *Yaay = *Yaay * constant;
    *Yaaz = *Yaaz * constant;
    
    
    *Ybbx = *Ybbx * constant;
    *Ybby = *Ybby * constant;
    *Ybbz = *Ybbz * constant;
    
    
    *Yccx = *Yccx * constant;
    *Yccy = *Yccy * constant;
    *Yccz = *Yccz * constant;
    
    
    
    return;
    
}





#undef STOREDIM
#define STOREDIM STOREDIM_S

__device__ __forceinline__ void hrrwholegrad(QUICKDouble* Yaax, QUICKDouble* Yaay, QUICKDouble* Yaaz, \
                                             QUICKDouble* Ybbx, QUICKDouble* Ybby, QUICKDouble* Ybbz, \
                                             QUICKDouble* Yccx, QUICKDouble* Yccy, QUICKDouble* Yccz, \
                                             const int I, const int J, const int K, const int L, \
                                             const int III, int JJJ, const int KKK, const int LLL, const int IJKLTYPE,
                                             QUICKDouble* store, QUICKDouble* storeAA, QUICKDouble* storeBB, QUICKDouble* storeCC, \
                                             const QUICKDouble RAx, const QUICKDouble RAy, const QUICKDouble RAz, \
                                             const QUICKDouble RBx, const QUICKDouble RBy, const QUICKDouble RBz, \
                                             const QUICKDouble RCx, const QUICKDouble RCy, const QUICKDouble RCz, \
                                             const QUICKDouble RDx, const QUICKDouble RDy, const QUICKDouble RDz)
{
    unsigned char angularL[8], angularR[8];
    QUICKDouble coefAngularL[8], coefAngularR[8];
    
    *Yaax = 0.0;
    *Yaay = 0.0;
    *Yaaz = 0.0;
    *Ybbx = 0.0;
    *Ybby = 0.0;
    *Ybbz = 0.0;
    *Yccx = 0.0;
    *Yccy = 0.0;
    *Yccz = 0.0;
    
    QUICKDouble constant = devSim.cons[III-1] * devSim.cons[JJJ-1] * devSim.cons[KKK-1] * devSim.cons[LLL-1];
    int numAngularL, numAngularR;
    
    numAngularR = lefthrr_spd(RCx, RCy, RCz, RDx, RDy, RDz, \
                          LOC2(devSim.KLMN,0,KKK-1,3,devSim.nbasis), LOC2(devSim.KLMN,1,KKK-1,3,devSim.nbasis), LOC2(devSim.KLMN,2,KKK-1,3,devSim.nbasis), \
                          LOC2(devSim.KLMN,0,LLL-1,3,devSim.nbasis), LOC2(devSim.KLMN,1,LLL-1,3,devSim.nbasis), LOC2(devSim.KLMN,2,LLL-1,3,devSim.nbasis), \
                          L, coefAngularR, angularR);
    
    
    //  Part A - x
    
    numAngularL = lefthrr_spd(RAx, RAy, RAz, RBx, RBy, RBz, \
                          LOC2(devSim.KLMN,0,III-1,3,devSim.nbasis) + 1, LOC2(devSim.KLMN,1,III-1,3,devSim.nbasis), LOC2(devSim.KLMN,2,III-1,3,devSim.nbasis), \
                          LOC2(devSim.KLMN,0,JJJ-1,3,devSim.nbasis), LOC2(devSim.KLMN,1,JJJ-1,3,devSim.nbasis), LOC2(devSim.KLMN,2,JJJ-1,3,devSim.nbasis), \
                          J, coefAngularL, angularL);
    for (int i = 0; i<numAngularL; i++) {
        for (int j = 0; j<numAngularR; j++) {
            if (angularL[i] <= STOREDIM && angularR[j] <= STOREDIM) {
                *Yaax = *Yaax + coefAngularL[i] * coefAngularR[j] * LOCSTORE(storeAA, angularL[i]-1, angularR[j]-1, STOREDIM, STOREDIM);
            }
        }
    }
    
    numAngularL = lefthrr_spd(RAx, RAy, RAz, RBx, RBy, RBz, \
                          LOC2(devSim.KLMN,0,III-1,3,devSim.nbasis), LOC2(devSim.KLMN,1,III-1,3,devSim.nbasis) + 1, LOC2(devSim.KLMN,2,III-1,3,devSim.nbasis), \
                          LOC2(devSim.KLMN,0,JJJ-1,3,devSim.nbasis), LOC2(devSim.KLMN,1,JJJ-1,3,devSim.nbasis), LOC2(devSim.KLMN,2,JJJ-1,3,devSim.nbasis), \
                          J, coefAngularL, angularL);
    for (int i = 0; i<numAngularL; i++) {
        for (int j = 0; j<numAngularR; j++) {
            if (angularL[i] <= STOREDIM && angularR[j] <= STOREDIM) {
                *Yaay = *Yaay + coefAngularL[i] * coefAngularR[j] * LOCSTORE(storeAA, angularL[i]-1, angularR[j]-1, STOREDIM, STOREDIM);
            }
        }
    }
    
    numAngularL = lefthrr_spd(RAx, RAy, RAz, RBx, RBy, RBz, \
                          LOC2(devSim.KLMN,0,III-1,3,devSim.nbasis), LOC2(devSim.KLMN,1,III-1,3,devSim.nbasis), LOC2(devSim.KLMN,2,III-1,3,devSim.nbasis) + 1, \
                          LOC2(devSim.KLMN,0,JJJ-1,3,devSim.nbasis), LOC2(devSim.KLMN,1,JJJ-1,3,devSim.nbasis), LOC2(devSim.KLMN,2,JJJ-1,3,devSim.nbasis), \
                          J, coefAngularL, angularL);
    for (int i = 0; i<numAngularL; i++) {
        for (int j = 0; j<numAngularR; j++) {
            if (angularL[i] <= STOREDIM && angularR[j] <= STOREDIM) {
                *Yaaz = *Yaaz + coefAngularL[i] * coefAngularR[j] * LOCSTORE(storeAA, angularL[i]-1, angularR[j]-1, STOREDIM, STOREDIM);
            }
        }
    }
    
    numAngularL = lefthrr(RAx, RAy, RAz, RBx, RBy, RBz, \
                          LOC2(devSim.KLMN,0,III-1,3,devSim.nbasis), LOC2(devSim.KLMN,1,III-1,3,devSim.nbasis), LOC2(devSim.KLMN,2,III-1,3,devSim.nbasis), \
                          LOC2(devSim.KLMN,0,JJJ-1,3,devSim.nbasis) + 1, LOC2(devSim.KLMN,1,JJJ-1,3,devSim.nbasis), LOC2(devSim.KLMN,2,JJJ-1,3,devSim.nbasis), \
                          J + 1, coefAngularL, angularL);
    for (int i = 0; i<numAngularL; i++) {
        for (int j = 0; j<numAngularR; j++) {
            if (angularL[i] <= STOREDIM && angularR[j] <= STOREDIM) {
                *Ybbx = *Ybbx + coefAngularL[i] * coefAngularR[j] * LOCSTORE(storeBB, angularL[i]-1, angularR[j]-1, STOREDIM, STOREDIM);
            }
        }
    }
    
    numAngularL = lefthrr(RAx, RAy, RAz, RBx, RBy, RBz, \
                          LOC2(devSim.KLMN,0,III-1,3,devSim.nbasis), LOC2(devSim.KLMN,1,III-1,3,devSim.nbasis), LOC2(devSim.KLMN,2,III-1,3,devSim.nbasis), \
                          LOC2(devSim.KLMN,0,JJJ-1,3,devSim.nbasis), LOC2(devSim.KLMN,1,JJJ-1,3,devSim.nbasis) + 1, LOC2(devSim.KLMN,2,JJJ-1,3,devSim.nbasis), \
                          J + 1, coefAngularL, angularL);
    for (int i = 0; i<numAngularL; i++) {
        for (int j = 0; j<numAngularR; j++) {
            if (angularL[i] <= STOREDIM && angularR[j] <= STOREDIM) {
                *Ybby = *Ybby + coefAngularL[i] * coefAngularR[j] * LOCSTORE(storeBB, angularL[i]-1, angularR[j]-1, STOREDIM, STOREDIM);
            }
        }
    }
    
    numAngularL = lefthrr(RAx, RAy, RAz, RBx, RBy, RBz, \
                          LOC2(devSim.KLMN,0,III-1,3,devSim.nbasis), LOC2(devSim.KLMN,1,III-1,3,devSim.nbasis), LOC2(devSim.KLMN,2,III-1,3,devSim.nbasis), \
                          LOC2(devSim.KLMN,0,JJJ-1,3,devSim.nbasis), LOC2(devSim.KLMN,1,JJJ-1,3,devSim.nbasis), LOC2(devSim.KLMN,2,JJJ-1,3,devSim.nbasis) + 1, \
                          J + 1, coefAngularL, angularL);
    for (int i = 0; i<numAngularL; i++) {
        for (int j = 0; j<numAngularR; j++) {
            if (angularL[i] <= STOREDIM && angularR[j] <= STOREDIM) {
                *Ybbz = *Ybbz + coefAngularL[i] * coefAngularR[j] * LOCSTORE(storeBB, angularL[i]-1, angularR[j]-1, STOREDIM, STOREDIM);
            }
        }
    }
    
    
    
    
    if (LOC2(devSim.KLMN,0,III-1,3,devSim.nbasis) >= 1) {
        
        numAngularL = lefthrr_spd(RAx, RAy, RAz, RBx, RBy, RBz,
                              LOC2(devSim.KLMN,0,III-1,3,devSim.nbasis) - 1, LOC2(devSim.KLMN,1,III-1,3,devSim.nbasis), LOC2(devSim.KLMN,2,III-1,3,devSim.nbasis),
                              LOC2(devSim.KLMN,0,JJJ-1,3,devSim.nbasis), LOC2(devSim.KLMN,1,JJJ-1,3,devSim.nbasis), LOC2(devSim.KLMN,2,JJJ-1,3,devSim.nbasis),
                              J, coefAngularL, angularL);
        for (int i = 0; i<numAngularL; i++) {
            for (int j = 0; j<numAngularR; j++) {
                
                if (angularL[i] <= STOREDIM && angularR[j] <= STOREDIM) {
                    *Yaax = *Yaax - LOC2(devSim.KLMN,0,III-1,3,devSim.nbasis) * coefAngularL[i] * coefAngularR[j] * LOCSTORE(store, angularL[i]-1, angularR[j]-1, STOREDIM, STOREDIM);
                }
            }
        }
        
    }
    
    if (LOC2(devSim.KLMN,1,III-1,3,devSim.nbasis) >= 1) {
        
        numAngularL = lefthrr_spd(RAx, RAy, RAz, RBx, RBy, RBz,
                              LOC2(devSim.KLMN,0,III-1,3,devSim.nbasis), LOC2(devSim.KLMN,1,III-1,3,devSim.nbasis) - 1, LOC2(devSim.KLMN,2,III-1,3,devSim.nbasis),
                              LOC2(devSim.KLMN,0,JJJ-1,3,devSim.nbasis), LOC2(devSim.KLMN,1,JJJ-1,3,devSim.nbasis), LOC2(devSim.KLMN,2,JJJ-1,3,devSim.nbasis),
                              J, coefAngularL, angularL);
        for (int i = 0; i<numAngularL; i++) {
            for (int j = 0; j<numAngularR; j++) {
                
                if (angularL[i] <= STOREDIM && angularR[j] <= STOREDIM) {
                    *Yaay = *Yaay - LOC2(devSim.KLMN,1,III-1,3,devSim.nbasis) * coefAngularL[i] * coefAngularR[j] * LOCSTORE(store, angularL[i]-1, angularR[j]-1, STOREDIM, STOREDIM);
                }
            }
        }
    }
    
    if (LOC2(devSim.KLMN,2,III-1,3,devSim.nbasis) >= 1) {
        
        numAngularL = lefthrr_spd(RAx, RAy, RAz, RBx, RBy, RBz,
                              LOC2(devSim.KLMN,0,III-1,3,devSim.nbasis), LOC2(devSim.KLMN,1,III-1,3,devSim.nbasis), LOC2(devSim.KLMN,2,III-1,3,devSim.nbasis) - 1,
                              LOC2(devSim.KLMN,0,JJJ-1,3,devSim.nbasis), LOC2(devSim.KLMN,1,JJJ-1,3,devSim.nbasis), LOC2(devSim.KLMN,2,JJJ-1,3,devSim.nbasis),
                              J, coefAngularL, angularL);
        for (int i = 0; i<numAngularL; i++) {
            for (int j = 0; j<numAngularR; j++) {
                
                if (angularL[i] <= STOREDIM && angularR[j] <= STOREDIM) {
                    *Yaaz = *Yaaz - LOC2(devSim.KLMN,2,III-1,3,devSim.nbasis) * coefAngularL[i] * coefAngularR[j] * LOCSTORE(store, angularL[i]-1, angularR[j]-1, STOREDIM, STOREDIM);
                }
            }
        }
    }
    
    
    
    
    if (LOC2(devSim.KLMN,0,JJJ-1,3,devSim.nbasis) >= 1) {
        
        numAngularL = lefthrr_spd(RAx, RAy, RAz, RBx, RBy, RBz,
                              LOC2(devSim.KLMN,0,III-1,3,devSim.nbasis), LOC2(devSim.KLMN,1,III-1,3,devSim.nbasis), LOC2(devSim.KLMN,2,III-1,3,devSim.nbasis),
                              LOC2(devSim.KLMN,0,JJJ-1,3,devSim.nbasis) - 1, LOC2(devSim.KLMN,1,JJJ-1,3,devSim.nbasis), LOC2(devSim.KLMN,2,JJJ-1,3,devSim.nbasis),
                              J - 1, coefAngularL, angularL);
        for (int i = 0; i<numAngularL; i++) {
            for (int j = 0; j<numAngularR; j++) {
                
                if (angularL[i] <= STOREDIM && angularR[j] <= STOREDIM) {
                    *Ybbx = *Ybbx - LOC2(devSim.KLMN,0,JJJ-1,3,devSim.nbasis) * coefAngularL[i] * coefAngularR[j] * LOCSTORE(store, angularL[i]-1, angularR[j]-1, STOREDIM, STOREDIM);
                }
            }
        }
    }
    
    
    
    if (LOC2(devSim.KLMN,1,JJJ-1,3,devSim.nbasis) >= 1) {
        
        numAngularL = lefthrr_spd(RAx, RAy, RAz, RBx, RBy, RBz,
                              LOC2(devSim.KLMN,0,III-1,3,devSim.nbasis), LOC2(devSim.KLMN,1,III-1,3,devSim.nbasis), LOC2(devSim.KLMN,2,III-1,3,devSim.nbasis),
                              LOC2(devSim.KLMN,0,JJJ-1,3,devSim.nbasis), LOC2(devSim.KLMN,1,JJJ-1,3,devSim.nbasis) - 1, LOC2(devSim.KLMN,2,JJJ-1,3,devSim.nbasis),
                              J - 1, coefAngularL, angularL);
        for (int i = 0; i<numAngularL; i++) {
            for (int j = 0; j<numAngularR; j++) {
                
                if (angularL[i] <= STOREDIM && angularR[j] <= STOREDIM) {
                    *Ybby = *Ybby - LOC2(devSim.KLMN,1,JJJ-1,3,devSim.nbasis) * coefAngularL[i] * coefAngularR[j] * LOCSTORE(store, angularL[i]-1, angularR[j]-1, STOREDIM, STOREDIM);
                }
            }
        }
    }
    
    
    if (LOC2(devSim.KLMN,2,JJJ-1,3,devSim.nbasis) >= 1) {
        
        numAngularL = lefthrr_spd(RAx, RAy, RAz, RBx, RBy, RBz,
                              LOC2(devSim.KLMN,0,III-1,3,devSim.nbasis), LOC2(devSim.KLMN,1,III-1,3,devSim.nbasis), LOC2(devSim.KLMN,2,III-1,3,devSim.nbasis),
                              LOC2(devSim.KLMN,0,JJJ-1,3,devSim.nbasis), LOC2(devSim.KLMN,1,JJJ-1,3,devSim.nbasis), LOC2(devSim.KLMN,2,JJJ-1,3,devSim.nbasis) - 1,
                              J - 1, coefAngularL, angularL);
        for (int i = 0; i<numAngularL; i++) {
            for (int j = 0; j<numAngularR; j++) {
                
                if (angularL[i] <= STOREDIM && angularR[j] <= STOREDIM) {
                    *Ybbz = *Ybbz - LOC2(devSim.KLMN,2,JJJ-1,3,devSim.nbasis) * coefAngularL[i] * coefAngularR[j] * LOCSTORE(store, angularL[i]-1, angularR[j]-1, STOREDIM, STOREDIM);
                }
            }
        }
    }
    
    
    
    // KET PART =====================================
    
    // Part C - x
    
    
    numAngularL = lefthrr_spd(RAx, RAy, RAz, RBx, RBy, RBz,
                          LOC2(devSim.KLMN,0,III-1,3,devSim.nbasis), LOC2(devSim.KLMN,1,III-1,3,devSim.nbasis), LOC2(devSim.KLMN,2,III-1,3,devSim.nbasis),
                          LOC2(devSim.KLMN,0,JJJ-1,3,devSim.nbasis), LOC2(devSim.KLMN,1,JJJ-1,3,devSim.nbasis), LOC2(devSim.KLMN,2,JJJ-1,3,devSim.nbasis),
                          J, coefAngularL, angularL);
    
    numAngularR = lefthrr_spd(RCx, RCy, RCz, RDx, RDy, RDz,
                          LOC2(devSim.KLMN,0,KKK-1,3,devSim.nbasis) + 1, LOC2(devSim.KLMN,1,KKK-1,3,devSim.nbasis), LOC2(devSim.KLMN,2,KKK-1,3,devSim.nbasis),
                          LOC2(devSim.KLMN,0,LLL-1,3,devSim.nbasis), LOC2(devSim.KLMN,1,LLL-1,3,devSim.nbasis), LOC2(devSim.KLMN,2,LLL-1,3,devSim.nbasis),
                          L, coefAngularR, angularR);
    
    
    for (int i = 0; i<numAngularL; i++) {
        for (int j = 0; j<numAngularR; j++) {
            
            if (angularL[i] <= STOREDIM && angularR[j] <= STOREDIM) {
                *Yccx = *Yccx + coefAngularL[i] * coefAngularR[j] * LOCSTORE(storeCC, angularL[i]-1, angularR[j]-1, STOREDIM, STOREDIM);
            }
        }
    }
    
    if (LOC2(devSim.KLMN,0,KKK-1,3,devSim.nbasis) >= 1) {
        
        numAngularR = lefthrr_spd(RCx, RCy, RCz, RDx, RDy, RDz,
                              LOC2(devSim.KLMN,0,KKK-1,3,devSim.nbasis) - 1, LOC2(devSim.KLMN,1,KKK-1,3,devSim.nbasis), LOC2(devSim.KLMN,2,KKK-1,3,devSim.nbasis),
                              LOC2(devSim.KLMN,0,LLL-1,3,devSim.nbasis), LOC2(devSim.KLMN,1,LLL-1,3,devSim.nbasis), LOC2(devSim.KLMN,2,LLL-1,3,devSim.nbasis),
                              L, coefAngularR, angularR);
        
        for (int i = 0; i<numAngularL; i++) {
            for (int j = 0; j<numAngularR; j++) {
                if (angularL[i] <= STOREDIM && angularR[j] <= STOREDIM) {
                    *Yccx = *Yccx - LOC2(devSim.KLMN,0,KKK-1,3,devSim.nbasis) * coefAngularL[i] * coefAngularR[j] * LOCSTORE(store, angularL[i]-1, angularR[j]-1, STOREDIM, STOREDIM);
                }
            }
        }
    }
    
    
    // Part C - y
    
    numAngularR = lefthrr_spd(RCx, RCy, RCz, RDx, RDy, RDz,
                          LOC2(devSim.KLMN,0,KKK-1,3,devSim.nbasis), LOC2(devSim.KLMN,1,KKK-1,3,devSim.nbasis) + 1, LOC2(devSim.KLMN,2,KKK-1,3,devSim.nbasis),
                          LOC2(devSim.KLMN,0,LLL-1,3,devSim.nbasis), LOC2(devSim.KLMN,1,LLL-1,3,devSim.nbasis), LOC2(devSim.KLMN,2,LLL-1,3,devSim.nbasis),
                          L, coefAngularR, angularR);
    
    
    for (int i = 0; i<numAngularL; i++) {
        for (int j = 0; j<numAngularR; j++) {
            
            if (angularL[i] <= STOREDIM && angularR[j] <= STOREDIM) {
                *Yccy = *Yccy + coefAngularL[i] * coefAngularR[j] * LOCSTORE(storeCC, angularL[i]-1, angularR[j]-1, STOREDIM, STOREDIM);
            }
        }
    }
    
    if (LOC2(devSim.KLMN,1,KKK-1,3,devSim.nbasis) >= 1) {
        
        numAngularR = lefthrr_spd(RCx, RCy, RCz, RDx, RDy, RDz,
                              LOC2(devSim.KLMN,0,KKK-1,3,devSim.nbasis), LOC2(devSim.KLMN,1,KKK-1,3,devSim.nbasis) - 1, LOC2(devSim.KLMN,2,KKK-1,3,devSim.nbasis),
                              LOC2(devSim.KLMN,0,LLL-1,3,devSim.nbasis), LOC2(devSim.KLMN,1,LLL-1,3,devSim.nbasis), LOC2(devSim.KLMN,2,LLL-1,3,devSim.nbasis),
                              L, coefAngularR, angularR);
        
        for (int i = 0; i<numAngularL; i++) {
            for (int j = 0; j<numAngularR; j++) {
                
                if (angularL[i] <= STOREDIM && angularR[j] <= STOREDIM) {
                    *Yccy = *Yccy - LOC2(devSim.KLMN,1,KKK-1,3,devSim.nbasis) * coefAngularL[i] * coefAngularR[j] * LOCSTORE(store, angularL[i]-1, angularR[j]-1, STOREDIM, STOREDIM);
                }
            }
        }
    }
    
    // Part C - z
    
    numAngularR = lefthrr_spd(RCx, RCy, RCz, RDx, RDy, RDz,
                          LOC2(devSim.KLMN,0,KKK-1,3,devSim.nbasis), LOC2(devSim.KLMN,1,KKK-1,3,devSim.nbasis), LOC2(devSim.KLMN,2,KKK-1,3,devSim.nbasis) + 1,
                          LOC2(devSim.KLMN,0,LLL-1,3,devSim.nbasis), LOC2(devSim.KLMN,1,LLL-1,3,devSim.nbasis), LOC2(devSim.KLMN,2,LLL-1,3,devSim.nbasis),
                          L, coefAngularR, angularR);
    
    
    for (int i = 0; i<numAngularL; i++) {
        for (int j = 0; j<numAngularR; j++) {
            
            if (angularL[i] <= STOREDIM && angularR[j] <= STOREDIM) {
                *Yccz = *Yccz + coefAngularL[i] * coefAngularR[j] * LOCSTORE(storeCC, angularL[i]-1, angularR[j]-1, STOREDIM, STOREDIM);
            }
        }
    }
    
    if (LOC2(devSim.KLMN,2,KKK-1,3,devSim.nbasis) >= 1) {
        
        numAngularR = lefthrr_spd(RCx, RCy, RCz, RDx, RDy, RDz,
                              LOC2(devSim.KLMN,0,KKK-1,3,devSim.nbasis), LOC2(devSim.KLMN,1,KKK-1,3,devSim.nbasis), LOC2(devSim.KLMN,2,KKK-1,3,devSim.nbasis) - 1,
                              LOC2(devSim.KLMN,0,LLL-1,3,devSim.nbasis), LOC2(devSim.KLMN,1,LLL-1,3,devSim.nbasis), LOC2(devSim.KLMN,2,LLL-1,3,devSim.nbasis),
                              L, coefAngularR, angularR);
        
        for (int i = 0; i<numAngularL; i++) {
            for (int j = 0; j<numAngularR; j++) {
                
                if (angularL[i] <= STOREDIM && angularR[j] <= STOREDIM) {
                    *Yccz = *Yccz - LOC2(devSim.KLMN,2,KKK-1,3,devSim.nbasis) * coefAngularL[i] * coefAngularR[j] * LOCSTORE(store, angularL[i]-1, angularR[j]-1, STOREDIM, STOREDIM);
                }
            }
        }
    }
    
    
    *Yaax = *Yaax * constant;
    *Yaay = *Yaay * constant;
    *Yaaz = *Yaaz * constant;
    
    
    *Ybbx = *Ybbx * constant;
    *Ybby = *Ybby * constant;
    *Ybbz = *Ybbz * constant;
    
    
    *Yccx = *Yccx * constant;
    *Yccy = *Yccy * constant;
    *Yccz = *Yccz * constant;
    
    
    
    return;
    
}

#undef STOREDIM
#define STOREDIM STOREDIM_XL

__device__ __forceinline__ void hrrwholegrad2(QUICKDouble* Yaax, QUICKDouble* Yaay, QUICKDouble* Yaaz, \
                                             QUICKDouble* Ybbx, QUICKDouble* Ybby, QUICKDouble* Ybbz, \
                                             QUICKDouble* Yccx, QUICKDouble* Yccy, QUICKDouble* Yccz, \
                                             int I, int J, int K, int L, \
                                             int III, int JJJ, int KKK, int LLL, int IJKLTYPE,
                                             QUICKDouble* store, QUICKDouble* storeAA, QUICKDouble* storeBB, QUICKDouble* storeCC, \
                                             QUICKDouble RAx,QUICKDouble RAy,QUICKDouble RAz, \
                                             QUICKDouble RBx,QUICKDouble RBy,QUICKDouble RBz, \
                                             QUICKDouble RCx,QUICKDouble RCy,QUICKDouble RCz, \
                                             QUICKDouble RDx,QUICKDouble RDy,QUICKDouble RDz)
{
    unsigned char angularL[12], angularR[12];
    QUICKDouble coefAngularL[12], coefAngularR[12];
    
    *Yaax = 0.0;
    *Yaay = 0.0;
    *Yaaz = 0.0;
    *Ybbx = 0.0;
    *Ybby = 0.0;
    *Ybbz = 0.0;
    *Yccx = 0.0;
    *Yccy = 0.0;
    *Yccz = 0.0;
    
    QUICKDouble constant = devSim.cons[III-1] * devSim.cons[JJJ-1] * devSim.cons[KKK-1] * devSim.cons[LLL-1];
    int numAngularL, numAngularR;
    
    numAngularR = lefthrr(RCx, RCy, RCz, RDx, RDy, RDz, \
                          LOC2(devSim.KLMN,0,KKK-1,3,devSim.nbasis), LOC2(devSim.KLMN,1,KKK-1,3,devSim.nbasis), LOC2(devSim.KLMN,2,KKK-1,3,devSim.nbasis), \
                          LOC2(devSim.KLMN,0,LLL-1,3,devSim.nbasis), LOC2(devSim.KLMN,1,LLL-1,3,devSim.nbasis), LOC2(devSim.KLMN,2,LLL-1,3,devSim.nbasis), \
                          L, coefAngularR, angularR);
    
    
    //  Part A - x
    
    numAngularL = lefthrr(RAx, RAy, RAz, RBx, RBy, RBz, \
                          LOC2(devSim.KLMN,0,III-1,3,devSim.nbasis) + 1, LOC2(devSim.KLMN,1,III-1,3,devSim.nbasis), LOC2(devSim.KLMN,2,III-1,3,devSim.nbasis), \
                          LOC2(devSim.KLMN,0,JJJ-1,3,devSim.nbasis), LOC2(devSim.KLMN,1,JJJ-1,3,devSim.nbasis), LOC2(devSim.KLMN,2,JJJ-1,3,devSim.nbasis), \
                          J, coefAngularL, angularL);


    for (int i = 0; i<numAngularL; i++) {
        for (int j = 0; j<numAngularR; j++) {
            if (angularL[i] <= STOREDIM && angularR[j] <= STOREDIM) {
                *Yaax = *Yaax + coefAngularL[i] * coefAngularR[j] * LOCSTORE(storeAA, angularL[i]-1, angularR[j]-1, STOREDIM, STOREDIM);
            }
        }
    }
   
 
    numAngularL = lefthrr(RAx, RAy, RAz, RBx, RBy, RBz, \
                          LOC2(devSim.KLMN,0,III-1,3,devSim.nbasis), LOC2(devSim.KLMN,1,III-1,3,devSim.nbasis) + 1, LOC2(devSim.KLMN,2,III-1,3,devSim.nbasis), \
                          LOC2(devSim.KLMN,0,JJJ-1,3,devSim.nbasis), LOC2(devSim.KLMN,1,JJJ-1,3,devSim.nbasis), LOC2(devSim.KLMN,2,JJJ-1,3,devSim.nbasis), \
                          J, coefAngularL, angularL);
    for (int i = 0; i<numAngularL; i++) {
        for (int j = 0; j<numAngularR; j++) {
            if (angularL[i] <= STOREDIM && angularR[j] <= STOREDIM) {
                *Yaay = *Yaay + coefAngularL[i] * coefAngularR[j] * LOCSTORE(storeAA, angularL[i]-1, angularR[j]-1, STOREDIM, STOREDIM);
            }
        }
    }
    
    numAngularL = lefthrr(RAx, RAy, RAz, RBx, RBy, RBz, \
                          LOC2(devSim.KLMN,0,III-1,3,devSim.nbasis), LOC2(devSim.KLMN,1,III-1,3,devSim.nbasis), LOC2(devSim.KLMN,2,III-1,3,devSim.nbasis) + 1, \
                          LOC2(devSim.KLMN,0,JJJ-1,3,devSim.nbasis), LOC2(devSim.KLMN,1,JJJ-1,3,devSim.nbasis), LOC2(devSim.KLMN,2,JJJ-1,3,devSim.nbasis), \
                          J, coefAngularL, angularL);
    for (int i = 0; i<numAngularL; i++) {
        for (int j = 0; j<numAngularR; j++) {
            if (angularL[i] <= STOREDIM && angularR[j] <= STOREDIM) {
                *Yaaz = *Yaaz + coefAngularL[i] * coefAngularR[j] * LOCSTORE(storeAA, angularL[i]-1, angularR[j]-1, STOREDIM, STOREDIM);
            }
        }
    }
    
    numAngularL = lefthrr(RAx, RAy, RAz, RBx, RBy, RBz, \
                          LOC2(devSim.KLMN,0,III-1,3,devSim.nbasis), LOC2(devSim.KLMN,1,III-1,3,devSim.nbasis), LOC2(devSim.KLMN,2,III-1,3,devSim.nbasis), \
                          LOC2(devSim.KLMN,0,JJJ-1,3,devSim.nbasis) + 1, LOC2(devSim.KLMN,1,JJJ-1,3,devSim.nbasis), LOC2(devSim.KLMN,2,JJJ-1,3,devSim.nbasis), \
                          J + 1, coefAngularL, angularL);
    for (int i = 0; i<numAngularL; i++) {
        for (int j = 0; j<numAngularR; j++) {
            if (angularL[i] <= STOREDIM && angularR[j] <= STOREDIM) {
                *Ybbx = *Ybbx + coefAngularL[i] * coefAngularR[j] * LOCSTORE(storeBB, angularL[i]-1, angularR[j]-1, STOREDIM, STOREDIM);
            }
        }
    }
    
    numAngularL = lefthrr(RAx, RAy, RAz, RBx, RBy, RBz, \
                          LOC2(devSim.KLMN,0,III-1,3,devSim.nbasis), LOC2(devSim.KLMN,1,III-1,3,devSim.nbasis), LOC2(devSim.KLMN,2,III-1,3,devSim.nbasis), \
                          LOC2(devSim.KLMN,0,JJJ-1,3,devSim.nbasis), LOC2(devSim.KLMN,1,JJJ-1,3,devSim.nbasis) + 1, LOC2(devSim.KLMN,2,JJJ-1,3,devSim.nbasis), \
                          J + 1, coefAngularL, angularL);
    for (int i = 0; i<numAngularL; i++) {
        for (int j = 0; j<numAngularR; j++) {
            if (angularL[i] <= STOREDIM && angularR[j] <= STOREDIM) {
                *Ybby = *Ybby + coefAngularL[i] * coefAngularR[j] * LOCSTORE(storeBB, angularL[i]-1, angularR[j]-1, STOREDIM, STOREDIM);
            }
        }
    }
    
    numAngularL = lefthrr(RAx, RAy, RAz, RBx, RBy, RBz, \
                          LOC2(devSim.KLMN,0,III-1,3,devSim.nbasis), LOC2(devSim.KLMN,1,III-1,3,devSim.nbasis), LOC2(devSim.KLMN,2,III-1,3,devSim.nbasis), \
                          LOC2(devSim.KLMN,0,JJJ-1,3,devSim.nbasis), LOC2(devSim.KLMN,1,JJJ-1,3,devSim.nbasis), LOC2(devSim.KLMN,2,JJJ-1,3,devSim.nbasis) + 1, \
                          J + 1, coefAngularL, angularL);
    for (int i = 0; i<numAngularL; i++) {
        for (int j = 0; j<numAngularR; j++) {
            if (angularL[i] <= STOREDIM && angularR[j] <= STOREDIM) {
                *Ybbz = *Ybbz + coefAngularL[i] * coefAngularR[j] * LOCSTORE(storeBB, angularL[i]-1, angularR[j]-1, STOREDIM, STOREDIM);
            }
        }
    }
    
    
    
    
    if (LOC2(devSim.KLMN,0,III-1,3,devSim.nbasis) >= 1) {
        
        numAngularL = lefthrr(RAx, RAy, RAz, RBx, RBy, RBz,
                              LOC2(devSim.KLMN,0,III-1,3,devSim.nbasis) - 1, LOC2(devSim.KLMN,1,III-1,3,devSim.nbasis), LOC2(devSim.KLMN,2,III-1,3,devSim.nbasis),
                              LOC2(devSim.KLMN,0,JJJ-1,3,devSim.nbasis), LOC2(devSim.KLMN,1,JJJ-1,3,devSim.nbasis), LOC2(devSim.KLMN,2,JJJ-1,3,devSim.nbasis),
                              J, coefAngularL, angularL);
        for (int i = 0; i<numAngularL; i++) {
            for (int j = 0; j<numAngularR; j++) {
                
                if (angularL[i] <= STOREDIM && angularR[j] <= STOREDIM) {
                    *Yaax = *Yaax - LOC2(devSim.KLMN,0,III-1,3,devSim.nbasis) * coefAngularL[i] * coefAngularR[j] * LOCSTORE(store, angularL[i]-1, angularR[j]-1, STOREDIM, STOREDIM);
                }
            }
        }
        
    }
    
    if (LOC2(devSim.KLMN,1,III-1,3,devSim.nbasis) >= 1) {
        
        numAngularL = lefthrr(RAx, RAy, RAz, RBx, RBy, RBz,
                              LOC2(devSim.KLMN,0,III-1,3,devSim.nbasis), LOC2(devSim.KLMN,1,III-1,3,devSim.nbasis) - 1, LOC2(devSim.KLMN,2,III-1,3,devSim.nbasis),
                              LOC2(devSim.KLMN,0,JJJ-1,3,devSim.nbasis), LOC2(devSim.KLMN,1,JJJ-1,3,devSim.nbasis), LOC2(devSim.KLMN,2,JJJ-1,3,devSim.nbasis),
                              J, coefAngularL, angularL);
        for (int i = 0; i<numAngularL; i++) {
            for (int j = 0; j<numAngularR; j++) {
                
                if (angularL[i] <= STOREDIM && angularR[j] <= STOREDIM) {
                    *Yaay = *Yaay - LOC2(devSim.KLMN,1,III-1,3,devSim.nbasis) * coefAngularL[i] * coefAngularR[j] * LOCSTORE(store, angularL[i]-1, angularR[j]-1, STOREDIM, STOREDIM);
                }
            }
        }
    }
    
    if (LOC2(devSim.KLMN,2,III-1,3,devSim.nbasis) >= 1) {
        
        numAngularL = lefthrr(RAx, RAy, RAz, RBx, RBy, RBz,
                              LOC2(devSim.KLMN,0,III-1,3,devSim.nbasis), LOC2(devSim.KLMN,1,III-1,3,devSim.nbasis), LOC2(devSim.KLMN,2,III-1,3,devSim.nbasis) - 1,
                              LOC2(devSim.KLMN,0,JJJ-1,3,devSim.nbasis), LOC2(devSim.KLMN,1,JJJ-1,3,devSim.nbasis), LOC2(devSim.KLMN,2,JJJ-1,3,devSim.nbasis),
                              J, coefAngularL, angularL);
        for (int i = 0; i<numAngularL; i++) {
            for (int j = 0; j<numAngularR; j++) {
                
                if (angularL[i] <= STOREDIM && angularR[j] <= STOREDIM) {
                    *Yaaz = *Yaaz - LOC2(devSim.KLMN,2,III-1,3,devSim.nbasis) * coefAngularL[i] * coefAngularR[j] * LOCSTORE(store, angularL[i]-1, angularR[j]-1, STOREDIM, STOREDIM);
                }
            }
        }
    }
    
    
    
    
    if (LOC2(devSim.KLMN,0,JJJ-1,3,devSim.nbasis) >= 1) {
        
        numAngularL = lefthrr(RAx, RAy, RAz, RBx, RBy, RBz,
                              LOC2(devSim.KLMN,0,III-1,3,devSim.nbasis), LOC2(devSim.KLMN,1,III-1,3,devSim.nbasis), LOC2(devSim.KLMN,2,III-1,3,devSim.nbasis),
                              LOC2(devSim.KLMN,0,JJJ-1,3,devSim.nbasis) - 1, LOC2(devSim.KLMN,1,JJJ-1,3,devSim.nbasis), LOC2(devSim.KLMN,2,JJJ-1,3,devSim.nbasis),
                              J - 1, coefAngularL, angularL);
        for (int i = 0; i<numAngularL; i++) {
            for (int j = 0; j<numAngularR; j++) {
                
                if (angularL[i] <= STOREDIM && angularR[j] <= STOREDIM) {
                    *Ybbx = *Ybbx - LOC2(devSim.KLMN,0,JJJ-1,3,devSim.nbasis) * coefAngularL[i] * coefAngularR[j] * LOCSTORE(store, angularL[i]-1, angularR[j]-1, STOREDIM, STOREDIM);
                }
            }
        }
    }
    
    
    
    if (LOC2(devSim.KLMN,1,JJJ-1,3,devSim.nbasis) >= 1) {
        
        numAngularL = lefthrr(RAx, RAy, RAz, RBx, RBy, RBz,
                              LOC2(devSim.KLMN,0,III-1,3,devSim.nbasis), LOC2(devSim.KLMN,1,III-1,3,devSim.nbasis), LOC2(devSim.KLMN,2,III-1,3,devSim.nbasis),
                              LOC2(devSim.KLMN,0,JJJ-1,3,devSim.nbasis), LOC2(devSim.KLMN,1,JJJ-1,3,devSim.nbasis) - 1, LOC2(devSim.KLMN,2,JJJ-1,3,devSim.nbasis),
                              J - 1, coefAngularL, angularL);
        for (int i = 0; i<numAngularL; i++) {
            for (int j = 0; j<numAngularR; j++) {
                
                if (angularL[i] <= STOREDIM && angularR[j] <= STOREDIM) {
                    *Ybby = *Ybby - LOC2(devSim.KLMN,1,JJJ-1,3,devSim.nbasis) * coefAngularL[i] * coefAngularR[j] * LOCSTORE(store, angularL[i]-1, angularR[j]-1, STOREDIM, STOREDIM);
                }
            }
        }
    }
    
    
    if (LOC2(devSim.KLMN,2,JJJ-1,3,devSim.nbasis) >= 1) {
        
        numAngularL = lefthrr(RAx, RAy, RAz, RBx, RBy, RBz,
                              LOC2(devSim.KLMN,0,III-1,3,devSim.nbasis), LOC2(devSim.KLMN,1,III-1,3,devSim.nbasis), LOC2(devSim.KLMN,2,III-1,3,devSim.nbasis),
                              LOC2(devSim.KLMN,0,JJJ-1,3,devSim.nbasis), LOC2(devSim.KLMN,1,JJJ-1,3,devSim.nbasis), LOC2(devSim.KLMN,2,JJJ-1,3,devSim.nbasis) - 1,
                              J - 1, coefAngularL, angularL);
        for (int i = 0; i<numAngularL; i++) {
            for (int j = 0; j<numAngularR; j++) {
                
                if (angularL[i] <= STOREDIM && angularR[j] <= STOREDIM) {
                    *Ybbz = *Ybbz - LOC2(devSim.KLMN,2,JJJ-1,3,devSim.nbasis) * coefAngularL[i] * coefAngularR[j] * LOCSTORE(store, angularL[i]-1, angularR[j]-1, STOREDIM, STOREDIM);
                    
                }
            }
        }
    }
    
    
    
    // KET PART =====================================
    
    // Part C - x
    
    
    numAngularL = lefthrr(RAx, RAy, RAz, RBx, RBy, RBz,
                          LOC2(devSim.KLMN,0,III-1,3,devSim.nbasis), LOC2(devSim.KLMN,1,III-1,3,devSim.nbasis), LOC2(devSim.KLMN,2,III-1,3,devSim.nbasis),
                          LOC2(devSim.KLMN,0,JJJ-1,3,devSim.nbasis), LOC2(devSim.KLMN,1,JJJ-1,3,devSim.nbasis), LOC2(devSim.KLMN,2,JJJ-1,3,devSim.nbasis),
                          J, coefAngularL, angularL);
    
    numAngularR = lefthrr(RCx, RCy, RCz, RDx, RDy, RDz,
                          LOC2(devSim.KLMN,0,KKK-1,3,devSim.nbasis) + 1, LOC2(devSim.KLMN,1,KKK-1,3,devSim.nbasis), LOC2(devSim.KLMN,2,KKK-1,3,devSim.nbasis),
                          LOC2(devSim.KLMN,0,LLL-1,3,devSim.nbasis), LOC2(devSim.KLMN,1,LLL-1,3,devSim.nbasis), LOC2(devSim.KLMN,2,LLL-1,3,devSim.nbasis),
                          L, coefAngularR, angularR);
    
    
    for (int i = 0; i<numAngularL; i++) {
        for (int j = 0; j<numAngularR; j++) {
            
            if (angularL[i] <= STOREDIM && angularR[j] <= STOREDIM) {
                *Yccx = *Yccx + coefAngularL[i] * coefAngularR[j] * LOCSTORE(storeCC, angularL[i]-1, angularR[j]-1, STOREDIM, STOREDIM);
            }
        }
    }
    
    if (LOC2(devSim.KLMN,0,KKK-1,3,devSim.nbasis) >= 1) {
        
        numAngularR = lefthrr(RCx, RCy, RCz, RDx, RDy, RDz,
                              LOC2(devSim.KLMN,0,KKK-1,3,devSim.nbasis) - 1, LOC2(devSim.KLMN,1,KKK-1,3,devSim.nbasis), LOC2(devSim.KLMN,2,KKK-1,3,devSim.nbasis),
                              LOC2(devSim.KLMN,0,LLL-1,3,devSim.nbasis), LOC2(devSim.KLMN,1,LLL-1,3,devSim.nbasis), LOC2(devSim.KLMN,2,LLL-1,3,devSim.nbasis),
                              L, coefAngularR, angularR);
        
        for (int i = 0; i<numAngularL; i++) {
            for (int j = 0; j<numAngularR; j++) {
                if (angularL[i] <= STOREDIM && angularR[j] <= STOREDIM) {
                    *Yccx = *Yccx - LOC2(devSim.KLMN,0,KKK-1,3,devSim.nbasis) * coefAngularL[i] * coefAngularR[j] * LOCSTORE(store, angularL[i]-1, angularR[j]-1, STOREDIM, STOREDIM);
                }
            }
        }
    }
    
    
    // Part C - y
    
    numAngularR = lefthrr(RCx, RCy, RCz, RDx, RDy, RDz,
                          LOC2(devSim.KLMN,0,KKK-1,3,devSim.nbasis), LOC2(devSim.KLMN,1,KKK-1,3,devSim.nbasis) + 1, LOC2(devSim.KLMN,2,KKK-1,3,devSim.nbasis),
                          LOC2(devSim.KLMN,0,LLL-1,3,devSim.nbasis), LOC2(devSim.KLMN,1,LLL-1,3,devSim.nbasis), LOC2(devSim.KLMN,2,LLL-1,3,devSim.nbasis),
                          L, coefAngularR, angularR);
    
    
    for (int i = 0; i<numAngularL; i++) {
        for (int j = 0; j<numAngularR; j++) {
            
            if (angularL[i] <= STOREDIM && angularR[j] <= STOREDIM) {
                *Yccy = *Yccy + coefAngularL[i] * coefAngularR[j] * LOCSTORE(storeCC, angularL[i]-1, angularR[j]-1, STOREDIM, STOREDIM);
            }
        }
    }
    
    if (LOC2(devSim.KLMN,1,KKK-1,3,devSim.nbasis) >= 1) {
        
        numAngularR = lefthrr(RCx, RCy, RCz, RDx, RDy, RDz,
                              LOC2(devSim.KLMN,0,KKK-1,3,devSim.nbasis), LOC2(devSim.KLMN,1,KKK-1,3,devSim.nbasis) - 1, LOC2(devSim.KLMN,2,KKK-1,3,devSim.nbasis),
                              LOC2(devSim.KLMN,0,LLL-1,3,devSim.nbasis), LOC2(devSim.KLMN,1,LLL-1,3,devSim.nbasis), LOC2(devSim.KLMN,2,LLL-1,3,devSim.nbasis),
                              L, coefAngularR, angularR);
        
        for (int i = 0; i<numAngularL; i++) {
            for (int j = 0; j<numAngularR; j++) {
                
                if (angularL[i] <= STOREDIM && angularR[j] <= STOREDIM) {
                    *Yccy = *Yccy - LOC2(devSim.KLMN,1,KKK-1,3,devSim.nbasis) * coefAngularL[i] * coefAngularR[j] * LOCSTORE(store, angularL[i]-1, angularR[j]-1, STOREDIM, STOREDIM);
                }
            }
        }
    }
    
    // Part C - z
    
    numAngularR = lefthrr(RCx, RCy, RCz, RDx, RDy, RDz,
                          LOC2(devSim.KLMN,0,KKK-1,3,devSim.nbasis), LOC2(devSim.KLMN,1,KKK-1,3,devSim.nbasis), LOC2(devSim.KLMN,2,KKK-1,3,devSim.nbasis) + 1,
                          LOC2(devSim.KLMN,0,LLL-1,3,devSim.nbasis), LOC2(devSim.KLMN,1,LLL-1,3,devSim.nbasis), LOC2(devSim.KLMN,2,LLL-1,3,devSim.nbasis),
                          L, coefAngularR, angularR);
    
    
    for (int i = 0; i<numAngularL; i++) {
        for (int j = 0; j<numAngularR; j++) {
            
            if (angularL[i] <= STOREDIM && angularR[j] <= STOREDIM) {
                *Yccz = *Yccz + coefAngularL[i] * coefAngularR[j] * LOCSTORE(storeCC, angularL[i]-1, angularR[j]-1, STOREDIM, STOREDIM);
            }
        }
    }
    
    if (LOC2(devSim.KLMN,2,KKK-1,3,devSim.nbasis) >= 1) {
        
        numAngularR = lefthrr(RCx, RCy, RCz, RDx, RDy, RDz,
                              LOC2(devSim.KLMN,0,KKK-1,3,devSim.nbasis), LOC2(devSim.KLMN,1,KKK-1,3,devSim.nbasis), LOC2(devSim.KLMN,2,KKK-1,3,devSim.nbasis) - 1,
                              LOC2(devSim.KLMN,0,LLL-1,3,devSim.nbasis), LOC2(devSim.KLMN,1,LLL-1,3,devSim.nbasis), LOC2(devSim.KLMN,2,LLL-1,3,devSim.nbasis),
                              L, coefAngularR, angularR);
        
        for (int i = 0; i<numAngularL; i++) {
            for (int j = 0; j<numAngularR; j++) {
                
                if (angularL[i] <= STOREDIM && angularR[j] <= STOREDIM) {
                    *Yccz = *Yccz - LOC2(devSim.KLMN,2,KKK-1,3,devSim.nbasis) * coefAngularL[i] * coefAngularR[j] * LOCSTORE(store, angularL[i]-1, angularR[j]-1, STOREDIM, STOREDIM);
                }
            }
        }
    }
    
    
    *Yaax = *Yaax * constant;
    *Yaay = *Yaay * constant;
    *Yaaz = *Yaaz * constant;
    
    
    *Ybbx = *Ybbx * constant;
    *Ybby = *Ybby * constant;
    *Ybbz = *Ybbz * constant;
    
    
    *Yccx = *Yccx * constant;
    *Yccy = *Yccy * constant;
    *Yccz = *Yccz * constant;
    
    
    
    return;
    
}

#undef STOREDIM
#define STOREDIM STOREDIM_GRAD_S

__device__ __forceinline__ void hrrwholegrad2_1(QUICKDouble* Yaax, QUICKDouble* Yaay, QUICKDouble* Yaaz, \
                                              QUICKDouble* Ybbx, QUICKDouble* Ybby, QUICKDouble* Ybbz, \
                                              QUICKDouble* Yccx, QUICKDouble* Yccy, QUICKDouble* Yccz, \
                                              int I, int J, int K, int L, \
                                              int III, int JJJ, int KKK, int LLL, int IJKLTYPE,
                                              QUICKDouble* store, QUICKDouble* storeAA, QUICKDouble* storeBB, QUICKDouble* storeCC, \
                                              QUICKDouble RAx,QUICKDouble RAy,QUICKDouble RAz, \
                                              QUICKDouble RBx,QUICKDouble RBy,QUICKDouble RBz, \
                                              QUICKDouble RCx,QUICKDouble RCy,QUICKDouble RCz, \
                                              QUICKDouble RDx,QUICKDouble RDy,QUICKDouble RDz)
{
    unsigned char angularL[12], angularR[12];
    QUICKDouble coefAngularL[12], coefAngularR[12];
    
    *Yaax = 0.0;
    *Yaay = 0.0;
    *Yaaz = 0.0;
    *Ybbx = 0.0;
    *Ybby = 0.0;
    *Ybbz = 0.0;
    *Yccx = 0.0;
    *Yccy = 0.0;
    *Yccz = 0.0;
    
    QUICKDouble constant = devSim.cons[III-1] * devSim.cons[JJJ-1] * devSim.cons[KKK-1] * devSim.cons[LLL-1];
    int numAngularL, numAngularR;
    
    
    // KET PART =====================================
    
    // Part C - x
    
    
    numAngularL = lefthrr(RAx, RAy, RAz, RBx, RBy, RBz,
                          LOC2(devSim.KLMN,0,III-1,3,devSim.nbasis), LOC2(devSim.KLMN,1,III-1,3,devSim.nbasis), LOC2(devSim.KLMN,2,III-1,3,devSim.nbasis),
                          LOC2(devSim.KLMN,0,JJJ-1,3,devSim.nbasis), LOC2(devSim.KLMN,1,JJJ-1,3,devSim.nbasis), LOC2(devSim.KLMN,2,JJJ-1,3,devSim.nbasis),
                          J, coefAngularL, angularL);
    
    numAngularR = lefthrr(RCx, RCy, RCz, RDx, RDy, RDz,
                          LOC2(devSim.KLMN,0,KKK-1,3,devSim.nbasis) + 1, LOC2(devSim.KLMN,1,KKK-1,3,devSim.nbasis), LOC2(devSim.KLMN,2,KKK-1,3,devSim.nbasis),
                          LOC2(devSim.KLMN,0,LLL-1,3,devSim.nbasis), LOC2(devSim.KLMN,1,LLL-1,3,devSim.nbasis), LOC2(devSim.KLMN,2,LLL-1,3,devSim.nbasis),
                          L, coefAngularR, angularR);
    
    
    for (int i = 0; i<numAngularL; i++) {
        for (int j = 0; j<numAngularR; j++) {
            
            if (angularL[i] <= STOREDIM && angularR[j] <= STOREDIM) {
                *Yccx = *Yccx + coefAngularL[i] * coefAngularR[j] * LOCSTORE(storeCC, angularL[i]-1, angularR[j]-1, STOREDIM, STOREDIM);
            }
        }
    }
    
    // Part C - y
    
    numAngularR = lefthrr(RCx, RCy, RCz, RDx, RDy, RDz,
                          LOC2(devSim.KLMN,0,KKK-1,3,devSim.nbasis), LOC2(devSim.KLMN,1,KKK-1,3,devSim.nbasis) + 1, LOC2(devSim.KLMN,2,KKK-1,3,devSim.nbasis),
                          LOC2(devSim.KLMN,0,LLL-1,3,devSim.nbasis), LOC2(devSim.KLMN,1,LLL-1,3,devSim.nbasis), LOC2(devSim.KLMN,2,LLL-1,3,devSim.nbasis),
                          L, coefAngularR, angularR);
    
    
    for (int i = 0; i<numAngularL; i++) {
        for (int j = 0; j<numAngularR; j++) {
            
            if (angularL[i] <= STOREDIM && angularR[j] <= STOREDIM) {
                *Yccy = *Yccy + coefAngularL[i] * coefAngularR[j] * LOCSTORE(storeCC, angularL[i]-1, angularR[j]-1, STOREDIM, STOREDIM);
            }
        }
    }
    
    // Part C - z
    
    numAngularR = lefthrr(RCx, RCy, RCz, RDx, RDy, RDz,
                          LOC2(devSim.KLMN,0,KKK-1,3,devSim.nbasis), LOC2(devSim.KLMN,1,KKK-1,3,devSim.nbasis), LOC2(devSim.KLMN,2,KKK-1,3,devSim.nbasis) + 1,
                          LOC2(devSim.KLMN,0,LLL-1,3,devSim.nbasis), LOC2(devSim.KLMN,1,LLL-1,3,devSim.nbasis), LOC2(devSim.KLMN,2,LLL-1,3,devSim.nbasis),
                          L, coefAngularR, angularR);
    
    
    for (int i = 0; i<numAngularL; i++) {
        for (int j = 0; j<numAngularR; j++) {
            
            if (angularL[i] <= STOREDIM && angularR[j] <= STOREDIM) {
                *Yccz = *Yccz + coefAngularL[i] * coefAngularR[j] * LOCSTORE(storeCC, angularL[i]-1, angularR[j]-1, STOREDIM, STOREDIM);
            }
        }
    }
    
    *Yaax = *Yaax * constant;
    *Yaay = *Yaay * constant;
    *Yaaz = *Yaaz * constant;
    
    
    *Ybbx = *Ybbx * constant;
    *Ybby = *Ybby * constant;
    *Ybbz = *Ybbz * constant;
    
    
    *Yccx = *Yccx * constant;
    *Yccy = *Yccy * constant;
    *Yccz = *Yccz * constant;
    
    
    
    return;
    
}


__device__ __forceinline__ void hrrwholegrad2_2(QUICKDouble* Yaax, QUICKDouble* Yaay, QUICKDouble* Yaaz, \
                                              QUICKDouble* Ybbx, QUICKDouble* Ybby, QUICKDouble* Ybbz, \
                                              QUICKDouble* Yccx, QUICKDouble* Yccy, QUICKDouble* Yccz, \
                                              int I, int J, int K, int L, \
                                              int III, int JJJ, int KKK, int LLL, int IJKLTYPE,
                                              QUICKDouble* store, QUICKDouble* storeAA, QUICKDouble* storeBB, QUICKDouble* storeCC, \
                                              QUICKDouble RAx,QUICKDouble RAy,QUICKDouble RAz, \
                                              QUICKDouble RBx,QUICKDouble RBy,QUICKDouble RBz, \
                                              QUICKDouble RCx,QUICKDouble RCy,QUICKDouble RCz, \
                                              QUICKDouble RDx,QUICKDouble RDy,QUICKDouble RDz)
{
    unsigned char angularL[12], angularR[12];
    QUICKDouble coefAngularL[12], coefAngularR[12];
    
    *Yaax = 0.0;
    *Yaay = 0.0;
    *Yaaz = 0.0;
    *Ybbx = 0.0;
    *Ybby = 0.0;
    *Ybbz = 0.0;
    *Yccx = 0.0;
    *Yccy = 0.0;
    *Yccz = 0.0;
    
    QUICKDouble constant = devSim.cons[III-1] * devSim.cons[JJJ-1] * devSim.cons[KKK-1] * devSim.cons[LLL-1];
    int numAngularL, numAngularR;
    
    numAngularR = lefthrr(RCx, RCy, RCz, RDx, RDy, RDz, \
                          LOC2(devSim.KLMN,0,KKK-1,3,devSim.nbasis), LOC2(devSim.KLMN,1,KKK-1,3,devSim.nbasis), LOC2(devSim.KLMN,2,KKK-1,3,devSim.nbasis), \
                          LOC2(devSim.KLMN,0,LLL-1,3,devSim.nbasis), LOC2(devSim.KLMN,1,LLL-1,3,devSim.nbasis), LOC2(devSim.KLMN,2,LLL-1,3,devSim.nbasis), \
                          L, coefAngularR, angularR);
    
    
    //  Part A - x
    
    numAngularL = lefthrr(RAx, RAy, RAz, RBx, RBy, RBz, \
                          LOC2(devSim.KLMN,0,III-1,3,devSim.nbasis) + 1, LOC2(devSim.KLMN,1,III-1,3,devSim.nbasis), LOC2(devSim.KLMN,2,III-1,3,devSim.nbasis), \
                          LOC2(devSim.KLMN,0,JJJ-1,3,devSim.nbasis), LOC2(devSim.KLMN,1,JJJ-1,3,devSim.nbasis), LOC2(devSim.KLMN,2,JJJ-1,3,devSim.nbasis), \
                          J, coefAngularL, angularL);
    for (int i = 0; i<numAngularL; i++) {
        for (int j = 0; j<numAngularR; j++) {
            if (angularL[i] <= STOREDIM && angularR[j] <= STOREDIM) {
                *Yaax = *Yaax + coefAngularL[i] * coefAngularR[j] * LOCSTORE(storeAA, angularL[i]-1, angularR[j]-1, STOREDIM, STOREDIM);
            }
        }
    }
    
    numAngularL = lefthrr(RAx, RAy, RAz, RBx, RBy, RBz, \
                          LOC2(devSim.KLMN,0,III-1,3,devSim.nbasis), LOC2(devSim.KLMN,1,III-1,3,devSim.nbasis) + 1, LOC2(devSim.KLMN,2,III-1,3,devSim.nbasis), \
                          LOC2(devSim.KLMN,0,JJJ-1,3,devSim.nbasis), LOC2(devSim.KLMN,1,JJJ-1,3,devSim.nbasis), LOC2(devSim.KLMN,2,JJJ-1,3,devSim.nbasis), \
                          J, coefAngularL, angularL);
    for (int i = 0; i<numAngularL; i++) {
        for (int j = 0; j<numAngularR; j++) {
            if (angularL[i] <= STOREDIM && angularR[j] <= STOREDIM) {
                *Yaay = *Yaay + coefAngularL[i] * coefAngularR[j] * LOCSTORE(storeAA, angularL[i]-1, angularR[j]-1, STOREDIM, STOREDIM);
            }
        }
    }
    
    numAngularL = lefthrr(RAx, RAy, RAz, RBx, RBy, RBz, \
                          LOC2(devSim.KLMN,0,III-1,3,devSim.nbasis), LOC2(devSim.KLMN,1,III-1,3,devSim.nbasis), LOC2(devSim.KLMN,2,III-1,3,devSim.nbasis) + 1, \
                          LOC2(devSim.KLMN,0,JJJ-1,3,devSim.nbasis), LOC2(devSim.KLMN,1,JJJ-1,3,devSim.nbasis), LOC2(devSim.KLMN,2,JJJ-1,3,devSim.nbasis), \
                          J, coefAngularL, angularL);
    for (int i = 0; i<numAngularL; i++) {
        for (int j = 0; j<numAngularR; j++) {
            if (angularL[i] <= STOREDIM && angularR[j] <= STOREDIM) {
                *Yaaz = *Yaaz + coefAngularL[i] * coefAngularR[j] * LOCSTORE(storeAA, angularL[i]-1, angularR[j]-1, STOREDIM, STOREDIM);
            }
        }
    }
    
    numAngularL = lefthrr(RAx, RAy, RAz, RBx, RBy, RBz, \
                          LOC2(devSim.KLMN,0,III-1,3,devSim.nbasis), LOC2(devSim.KLMN,1,III-1,3,devSim.nbasis), LOC2(devSim.KLMN,2,III-1,3,devSim.nbasis), \
                          LOC2(devSim.KLMN,0,JJJ-1,3,devSim.nbasis) + 1, LOC2(devSim.KLMN,1,JJJ-1,3,devSim.nbasis), LOC2(devSim.KLMN,2,JJJ-1,3,devSim.nbasis), \
                          J + 1, coefAngularL, angularL);
    for (int i = 0; i<numAngularL; i++) {
        for (int j = 0; j<numAngularR; j++) {
            if (angularL[i] <= STOREDIM && angularR[j] <= STOREDIM) {
                *Ybbx = *Ybbx + coefAngularL[i] * coefAngularR[j] * LOCSTORE(storeBB, angularL[i]-1, angularR[j]-1, STOREDIM, STOREDIM);
            }
        }
    }
    
    numAngularL = lefthrr(RAx, RAy, RAz, RBx, RBy, RBz, \
                          LOC2(devSim.KLMN,0,III-1,3,devSim.nbasis), LOC2(devSim.KLMN,1,III-1,3,devSim.nbasis), LOC2(devSim.KLMN,2,III-1,3,devSim.nbasis), \
                          LOC2(devSim.KLMN,0,JJJ-1,3,devSim.nbasis), LOC2(devSim.KLMN,1,JJJ-1,3,devSim.nbasis) + 1, LOC2(devSim.KLMN,2,JJJ-1,3,devSim.nbasis), \
                          J + 1, coefAngularL, angularL);
    for (int i = 0; i<numAngularL; i++) {
        for (int j = 0; j<numAngularR; j++) {
            if (angularL[i] <= STOREDIM && angularR[j] <= STOREDIM) {
                *Ybby = *Ybby + coefAngularL[i] * coefAngularR[j] * LOCSTORE(storeBB, angularL[i]-1, angularR[j]-1, STOREDIM, STOREDIM);
            }
        }
    }
    
    numAngularL = lefthrr(RAx, RAy, RAz, RBx, RBy, RBz, \
                          LOC2(devSim.KLMN,0,III-1,3,devSim.nbasis), LOC2(devSim.KLMN,1,III-1,3,devSim.nbasis), LOC2(devSim.KLMN,2,III-1,3,devSim.nbasis), \
                          LOC2(devSim.KLMN,0,JJJ-1,3,devSim.nbasis), LOC2(devSim.KLMN,1,JJJ-1,3,devSim.nbasis), LOC2(devSim.KLMN,2,JJJ-1,3,devSim.nbasis) + 1, \
                          J + 1, coefAngularL, angularL);
    for (int i = 0; i<numAngularL; i++) {
        for (int j = 0; j<numAngularR; j++) {
            if (angularL[i] <= STOREDIM && angularR[j] <= STOREDIM) {
                *Ybbz = *Ybbz + coefAngularL[i] * coefAngularR[j] * LOCSTORE(storeBB, angularL[i]-1, angularR[j]-1, STOREDIM, STOREDIM);
            }
        }
    }
    
    
    *Yaax = *Yaax * constant;
    *Yaay = *Yaay * constant;
    *Yaaz = *Yaaz * constant;
    
    
    *Ybbx = *Ybbx * constant;
    *Ybby = *Ybby * constant;
    *Ybbz = *Ybbz * constant;
    
    
    *Yccx = *Yccx * constant;
    *Yccy = *Yccy * constant;
    *Yccz = *Yccz * constant;
    
    
    
    return;
    
}

#endif

#ifdef int_sp
#ifndef sp_grad_fmt
#define sp_grad_fmt
#undef FMT_NAME
#define FMT_NAME FmT_grad_sp
#include "gpu_fmt.h"
#endif
#endif



