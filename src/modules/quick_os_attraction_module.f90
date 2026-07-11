#include "util.fh"
!---------------------------------------------------------------------!
! Obara--Saika attraction integrals and derivatives with respect to   !
! the external center C.                                              !
!                                                                     !
! Copyright (C) 2026 QUICK contributors                               !
!                                                                     !
! This Source Code Form is subject to the terms of the Mozilla Public !
! License, v. 2.0. If a copy of the MPL was not distributed with this !
! file, You can obtain one at http://mozilla.org/MPL/2.0/.             !
!_____________________________________________________________________!

module quick_os_attraction_module
   implicit none
   private

   public :: os_attraction_derivatives

contains

   ! Return the scalar attraction integral, its three C derivatives,
   ! and the six independent second C derivatives in one traversal.
   recursive function os_attraction_derivatives(i,j,k,ii,jj,kk,m,aux, &
         Ax,Ay,Az,Bx,By,Bz,Cx,Cy,Cz,Px,Py,Pz,g) result(deriv)
      implicit none

      integer, intent(in) :: i,j,k,ii,jj,kk,m
      double precision, intent(in) :: aux(0:20)
      double precision, intent(in) :: Ax,Ay,Az,Bx,By,Bz,Cx,Cy,Cz,Px,Py,Pz,g
      double precision :: deriv(10)

      integer, parameter :: pair_i(6) = (/1,1,1,2,2,3/)
      integer, parameter :: pair_j(6) = (/1,2,3,2,3,3/)
      integer :: exponents(6), ilowex, ilownum, l, mu, pair
      double precision :: center(12), pc(3), coeff, pa
      double precision :: lower(10), higher(10), same_m(10), same_m1(10)

      deriv(:) = 0.0d0
      if (i+j+k+ii+jj+kk == 0) then
         pc(:) = (/Px-Cx,Py-Cy,Pz-Cz/)
         deriv(1) = aux(m)
         do mu=1,3
            deriv(1+mu) = 2.0d0*g*pc(mu)*aux(m+1)
         enddo
         do pair=1,6
            deriv(4+pair) = 4.0d0*g*g*pc(pair_i(pair))*pc(pair_j(pair))*aux(m+2)
            if (pair_i(pair) == pair_j(pair)) &
               deriv(4+pair) = deriv(4+pair) - 2.0d0*g*aux(m+1)
         enddo
         return
      endif

      exponents(:) = (/i,j,k,ii,jj,kk/)
      center(7:12) = (/Cx,Cy,Cz,Px,Py,Pz/)
      ilownum = 0
      ilowex = huge(ilowex)
      do l=1,6
         if (exponents(l) /= 0 .and. exponents(l) < ilowex) then
            ilowex = exponents(l)
            ilownum = l
         endif
      enddo

      if (ilownum <= 3) then
         center(1:6) = (/Ax,Ay,Az,Bx,By,Bz/)
      else
         center(1:6) = (/Bx,By,Bz,Ax,Ay,Az/)
         exponents(:) = (/ii,jj,kk,i,j,k/)
         ilownum = ilownum-3
      endif

      exponents(ilownum) = exponents(ilownum)-1
      lower = os_attraction_derivatives(exponents(1),exponents(2),exponents(3), &
         exponents(4),exponents(5),exponents(6),m,aux,center(1),center(2), &
         center(3),center(4),center(5),center(6),center(7),center(8),center(9), &
         center(10),center(11),center(12),g)
      higher = os_attraction_derivatives(exponents(1),exponents(2),exponents(3), &
         exponents(4),exponents(5),exponents(6),m+1,aux,center(1),center(2), &
         center(3),center(4),center(5),center(6),center(7),center(8),center(9), &
         center(10),center(11),center(12),g)

      pa = center(9+ilownum)-center(ilownum)
      deriv(:) = pa*lower(:) - (center(9+ilownum)-center(6+ilownum))*higher(:)

      if (exponents(ilownum) /= 0) then
         coeff = dble(exponents(ilownum))/(2.0d0*g)
         exponents(ilownum) = exponents(ilownum)-1
         same_m = os_attraction_derivatives(exponents(1),exponents(2),exponents(3), &
            exponents(4),exponents(5),exponents(6),m,aux,center(1),center(2), &
            center(3),center(4),center(5),center(6),center(7),center(8),center(9), &
            center(10),center(11),center(12),g)
         same_m1 = os_attraction_derivatives(exponents(1),exponents(2),exponents(3), &
            exponents(4),exponents(5),exponents(6),m+1,aux,center(1),center(2), &
            center(3),center(4),center(5),center(6),center(7),center(8),center(9), &
            center(10),center(11),center(12),g)
         deriv(:) = deriv(:) + coeff*(same_m(:)-same_m1(:))
         exponents(ilownum) = exponents(ilownum)+1
      endif

      if (exponents(ilownum+3) /= 0) then
         coeff = dble(exponents(ilownum+3))/(2.0d0*g)
         exponents(ilownum+3) = exponents(ilownum+3)-1
         same_m = os_attraction_derivatives(exponents(1),exponents(2),exponents(3), &
            exponents(4),exponents(5),exponents(6),m,aux,center(1),center(2), &
            center(3),center(4),center(5),center(6),center(7),center(8),center(9), &
            center(10),center(11),center(12),g)
         same_m1 = os_attraction_derivatives(exponents(1),exponents(2),exponents(3), &
            exponents(4),exponents(5),exponents(6),m+1,aux,center(1),center(2), &
            center(3),center(4),center(5),center(6),center(7),center(8),center(9), &
            center(10),center(11),center(12),g)
         deriv(:) = deriv(:) + coeff*(same_m(:)-same_m1(:))
      endif

      ! Differentiating -(P_q-C_q)R^(m+1) generates lower-rank terms.
      deriv(1+ilownum) = deriv(1+ilownum) + higher(1)
      do pair=1,6
         if (pair_i(pair) == ilownum) &
            deriv(4+pair) = deriv(4+pair) + higher(1+pair_j(pair))
         if (pair_j(pair) == ilownum) &
            deriv(4+pair) = deriv(4+pair) + higher(1+pair_i(pair))
      enddo
   end function os_attraction_derivatives

end module quick_os_attraction_module
