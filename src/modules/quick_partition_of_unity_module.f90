#include "util.fh"
! (C) Copyright 2026 QUICK contributors. All rights reserved.
! This Source Code Form is subject to the terms of the Mozilla Public
! License, v. 2.0. If a copy of the MPL was not distributed with this
! file, You can obtain one at http://mozilla.org/MPL/2.0/.
module quick_partition_of_unity_module
   use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
   use, intrinsic :: iso_fortran_env, only: int64
   implicit none
   private
   public :: evaluate_electronic_cover,evaluate_electronic_cover_gpu,get_adaptive_gpu_info
#if defined(CUDA) || defined(CUDA_MPIV)
   interface
      subroutine gpu_adaptive_electrostatics(nside,degree,lower,upper,va,ea,ga,ha,majorants, &
            has_anchor_errors,anchor_errors,npoints,points,v,e,g,budgets,max_overlap,ierr)
         implicit none
         integer, intent(in) :: nside,degree,has_anchor_errors,npoints
         double precision, intent(in) :: lower(3),upper(3),va(*),ea(3,*),ga(3,3,*),ha(3,3,3,*)
         double precision, intent(in) :: majorants(2,*),anchor_errors(4,*),points(3,npoints)
         double precision, intent(inout) :: v(npoints),e(3,npoints),g(3,3,npoints),budgets(3,npoints)
         integer, intent(out) :: max_overlap,ierr
      end subroutine gpu_adaptive_electrostatics
      subroutine gpu_adaptive_electrostatics_launch_count(count)
         implicit none
         integer, intent(out) :: count
      end subroutine gpu_adaptive_electrostatics_launch_count
      subroutine gpu_adaptive_electrostatics_allocation_bytes(live,last)
         import int64
         implicit none
         integer(int64), intent(out) :: live,last
      end subroutine gpu_adaptive_electrostatics_allocation_bytes
   end interface
#endif
contains
   subroutine evaluate_electronic_cover_gpu(lower,upper,nside,degree,va,ea,ga,ha,majorants, &
         points,v,e,g,budgets,max_overlap,ierr,anchor_errors)
      implicit none
      double precision, intent(in) :: lower(3),upper(3),va(:),ea(:,:),ga(:,:,:),ha(:,:,:,:)
      double precision, intent(in) :: majorants(:,:),points(:,:)
      integer, intent(in) :: nside,degree
      double precision, intent(inout) :: v(:),e(:,:),g(:,:,:),budgets(:,:)
      integer, intent(out) :: max_overlap,ierr
      double precision, intent(in), optional :: anchor_errors(:,:)
      integer :: nanchors,npoints
#if defined(CUDA) || defined(CUDA_MPIV)
      double precision :: unused(4,1)
#endif
      ierr=46;max_overlap=0
      if (nside<1.or.nside>1024.or.(degree/=2.and.degree/=3)) return
      nanchors=nside**3;npoints=size(points,2)
      if (npoints<1.or.size(points,1)/=3) return
      if (size(va)/=nanchors.or.any(shape(ea)/=[3,nanchors])) return
      if (any(shape(ga)/=[3,3,nanchors]).or.any(shape(ha)/=[3,3,3,nanchors])) return
      if (any(shape(majorants)/=[2,nanchors])) return
      if (size(v)/=npoints.or.any(shape(e)/=[3,npoints]).or.any(shape(g)/=[3,3,npoints])) return
      if (any(shape(budgets)/=[3,npoints])) return
      if (present(anchor_errors)) then
         if (any(shape(anchor_errors)/=[4,nanchors])) return
      endif
      ierr=56
#if defined(CUDA) || defined(CUDA_MPIV)
      if (present(anchor_errors)) then
         call gpu_adaptive_electrostatics(nside,degree,lower,upper,va,ea,ga,ha,majorants, &
            1,anchor_errors,npoints,points,v,e,g,budgets,max_overlap,ierr)
      else
         unused=0.0d0
         call gpu_adaptive_electrostatics(nside,degree,lower,upper,va,ea,ga,ha,majorants, &
            0,unused,npoints,points,v,e,g,budgets,max_overlap,ierr)
      endif
#endif
   end subroutine evaluate_electronic_cover_gpu

   subroutine get_adaptive_gpu_info(launches,live_bytes,last_bytes)
      implicit none
      integer, intent(out) :: launches
      integer(int64), intent(out) :: live_bytes,last_bytes
      launches=0;live_bytes=0_int64;last_bytes=0_int64
#if defined(CUDA) || defined(CUDA_MPIV)
      call gpu_adaptive_electrostatics_launch_count(launches)
      call gpu_adaptive_electrostatics_allocation_bytes(live_bytes,last_bytes)
#endif
   end subroutine get_adaptive_gpu_info

   subroutine evaluate_electronic_cover(lower,upper,nside,degree,va,ea,ga,ha,majorants, &
         points,v,e,g,budgets,max_overlap,ierr,anchor_errors)
      !---------------------------------------------------------------------!
      ! One fixed, uniform cover of an electronic scalar potential. The   !
      ! compact weights psi(t)=(1-t^2)^4 (|t|<1), zero otherwise, are C3.   !
      ! Supports have halfwidth equal to the leaf width, hence at most     !
      ! eight active patches. Centers are indexed x, then y, then z, with  !
      ! z varying fastest. ALL first/second normalized-weight derivatives !
      ! enter E=-grad(V) and G=-Hess(V). Classical sources are added later. !
      ! M3/M4 are whole SUPPORT Frobenius majorants, never center norms.   !
      ! Budgets are real-arithmetic Taylor/POU inequalities evaluated FP64;!
      ! no total floating-point error certificate is implied. Optional    !
      ! anchor errors are [deltaV,deltaE,deltaG,deltaH] norm budgets; absent!
      ! values mean UNAVAILABLE, not zero numerical error. The caller must!
      ! retain this distinction. On failure caller outputs are unchanged. !
      !---------------------------------------------------------------------!
      implicit none
      double precision, intent(in) :: lower(3),upper(3),va(:),ea(:,:),ga(:,:,:),ha(:,:,:,:)
      double precision, intent(in) :: majorants(:,:),points(:,:)
      integer, intent(in) :: nside,degree
      double precision, intent(inout) :: v(:),e(:,:),g(:,:,:),budgets(:,:)
      integer, intent(out) :: max_overlap,ierr
      double precision, intent(in), optional :: anchor_errors(:,:)
      double precision, allocatable :: wv(:),we(:,:),wg(:,:,:),wb(:,:)
      double precision :: width(3),center(3),u(3,8),q(8),dq(3,8),ddq(3,3,8)
      double precision :: weight(8),dw(3,8),ddw(3,3,8),s,ds(3),dds(3,3)
      double precision :: t(3),psi(3),first(3),second(3),p(8),ep(3,8),gp(3,3,8)
      double precision :: remainder(3,8),r,m,huu(3),hu(3,3),delta(4),dn,ddn,p_reference,symmetry_tolerance
      integer :: count,nanchors,point,patch,ix,iy,iz,i,j,k,axis,other,base(3),ijk(3),ids(8),status

      ierr=46
      max_overlap=0
      if (nside<1.or.nside>1024.or.(degree/=2.and.degree/=3)) return
      if (nside>huge(nanchors)/nside/nside) return
      nanchors=nside**3
      count=size(points,2)
      if (count<1.or.size(points,1)/=3) return
      if (size(va)/=nanchors.or.any(shape(ea)/=[3,nanchors])) return
      if (any(shape(ga)/=[3,3,nanchors]).or.any(shape(ha)/=[3,3,3,nanchors])) return
      if (any(shape(majorants)/=[2,nanchors])) return
      if (size(v)/=count.or.any(shape(e)/=[3,count]).or.any(shape(g)/=[3,3,count])) return
      if (any(shape(budgets)/=[3,count])) return
      if (present(anchor_errors)) then
         if (any(shape(anchor_errors)/=[4,nanchors])) return
         if (.not.all(ieee_is_finite(anchor_errors)).or.any(anchor_errors<0)) return
      endif
      ierr=55
      if (.not.all(ieee_is_finite(lower)).or..not.all(ieee_is_finite(upper))) return
      if (any(upper<=lower).or..not.all(ieee_is_finite(points))) return
      if (any(minval(points,dim=2)<lower).or.any(maxval(points,dim=2)>upper)) return
      if (.not.all(ieee_is_finite(va)).or..not.all(ieee_is_finite(ea))) return
      if (.not.all(ieee_is_finite(ga)).or..not.all(ieee_is_finite(ha))) return
      if (.not.all(ieee_is_finite(majorants)).or.any(majorants<0)) return
      ! A nonsymmetric anchor tensor would make the supplied vector/tensor
      ! polynomials disagree with derivatives of the scalar polynomial.
      ! Measure and reject it; never conceal it by a symmetry projection.
      symmetry_tolerance=64.0d0*epsilon(1.0d0)*max(1.0d0,maxval(abs(ga)))
      do i=1,3
         do j=i+1,3
            if (maxval(abs(ga(i,j,:)-ga(j,i,:)))>symmetry_tolerance) return
         enddo
      enddo
      symmetry_tolerance=64.0d0*epsilon(1.0d0)*max(1.0d0,maxval(abs(ha)))
      do i=1,3
         do j=1,3
            do k=1,3
               if (maxval(abs(ha(i,j,k,:)-ha(j,i,k,:)))>symmetry_tolerance) return
               if (maxval(abs(ha(i,j,k,:)-ha(i,k,j,:)))>symmetry_tolerance) return
            enddo
         enddo
      enddo
      width=(upper-lower)/nside
      if (any(width<=0).or..not.all(ieee_is_finite(1.0d0/width**2))) return
      allocate(wv(count),we(3,count),wg(3,3,count),wb(3,count),stat=status)
      ierr=56
      if (status/=0) return
      ierr=55
      do point=1,count
         base=floor((points(:,point)-lower)/width-0.5d0)
         q=0;dq=0;ddq=0;p=0;ep=0;gp=0;remainder=0
         patch=0
         do ix=0,1
            do iy=0,1
               do iz=0,1
                  patch=patch+1
                  ijk=base+[ix,iy,iz]
                  if (any(ijk<0).or.any(ijk>=nside)) cycle
                  ids(patch)=(ijk(1)*nside+ijk(2))*nside+ijk(3)+1
                  center=lower+(real(ijk,8)+0.5d0)*width
                  u(:,patch)=points(:,point)-center
                  t=u(:,patch)/width
                  if (any(abs(t)>=1.0d0)) cycle
                  psi=(1.0d0-t*t)**4
                  first=-8.0d0*t*(1.0d0-t*t)**3/width
                  second=8.0d0*(1.0d0-t*t)**2*(7.0d0*t*t-1.0d0)/width**2
                  q(patch)=product(psi)
                  do i=1,3
                     dq(i,patch)=first(i)
                     ddq(i,i,patch)=second(i)
                     do axis=1,3
                        if (axis==i) cycle
                        dq(i,patch)=dq(i,patch)*psi(axis)
                        ddq(i,i,patch)=ddq(i,i,patch)*psi(axis)
                     enddo
                     do j=i+1,3
                        other=6-i-j
                        ddq(i,j,patch)=first(i)*first(j)*psi(other)
                        ddq(j,i,patch)=ddq(i,j,patch)
                     enddo
                  enddo
                  k=ids(patch)
                  p(patch)=va(k)-dot_product(ea(:,k),u(:,patch)) &
                     -0.5d0*dot_product(u(:,patch),matmul(ga(:,:,k),u(:,patch)))
                  ep(:,patch)=ea(:,k)+matmul(ga(:,:,k),u(:,patch))
                  gp(:,:,patch)=ga(:,:,k)
                  if (degree==3) then
                     hu=0
                     do i=1,3
                        do j=1,3
                           hu(i,j)=dot_product(ha(i,j,:,k),u(:,patch))
                        enddo
                     enddo
                     huu=matmul(hu,u(:,patch))
                     p(patch)=p(patch)-dot_product(huu,u(:,patch))/6.0d0
                     ep(:,patch)=ep(:,patch)+0.5d0*huu
                     gp(:,:,patch)=gp(:,:,patch)+hu
                  endif
                  r=sqrt(sum(u(:,patch)**2))
                  m=majorants(degree-1,k)
                  if (degree==2) then
                     remainder(:,patch)=m*[r**3/6.0d0,r*r/2.0d0,r]
                  else
                     remainder(:,patch)=m*[r**4/24.0d0,r**3/6.0d0,r*r/2.0d0]
                  endif
                  if (present(anchor_errors)) then
                     delta=anchor_errors(:,k)
                     remainder(:,patch)=remainder(:,patch)+ &
                        [delta(1)+delta(2)*r+delta(3)*r*r/2.0d0,delta(2)+delta(3)*r,delta(3)]
                     if (degree==3) remainder(:,patch)=remainder(:,patch)+ &
                        delta(4)*[r**3/6.0d0,r*r/2.0d0,r]
                  endif
               enddo
            enddo
         enddo
         s=sum(q);ds=sum(dq,dim=2);dds=sum(ddq,dim=3)
         ! Every covered query is within half a support in each dimension.
         ! A small relative slack permits the boundary arithmetic in FP64.
         if (.not.ieee_is_finite(s).or.s<(0.75d0**12)*(1.0d0-1.0d-12)) return
         max_overlap=max(max_overlap,count_active(q))
         weight=q/s
         do patch=1,8
            dw(:,patch)=(dq(:,patch)-weight(patch)*ds)/s
            do i=1,3
               do j=1,3
                  ddw(i,j,patch)=(ddq(i,j,patch)-weight(patch)*dds(i,j) &
                     -dw(i,patch)*ds(j)-ds(i)*dw(j,patch))/s
               enddo
            enddo
         enddo
         ! A common potential offset cancels from derivatives analytically.
         ! Subtract an active patch value before accumulation to avoid large
         ! gauge constants multiplying nearly cancelling weight derivatives.
         k=maxloc(q,dim=1)
         p_reference=p(k)
         p=p-p_reference
         wv(point)=p_reference+sum(weight*p)
         we(:,point)=0;wg(:,:,point)=0;wb(:,point)=0
         do patch=1,8
            we(:,point)=we(:,point)+weight(patch)*ep(:,patch)-p(patch)*dw(:,patch)
            do i=1,3
               do j=1,3
                  wg(i,j,point)=wg(i,j,point)+weight(patch)*gp(i,j,patch) &
                     +dw(i,patch)*ep(j,patch)+ep(i,patch)*dw(j,patch)-p(patch)*ddw(i,j,patch)
               enddo
            enddo
            dn=sqrt(sum(dw(:,patch)**2));ddn=sqrt(sum(ddw(:,:,patch)**2))
            wb(:,point)=wb(:,point)+weight(patch)*remainder(:,patch) &
               +[0.0d0,dn*remainder(1,patch), &
                  2.0d0*dn*remainder(2,patch)+ddn*remainder(1,patch)]
         enddo
      enddo
      if (.not.all(ieee_is_finite(wv)).or..not.all(ieee_is_finite(we))) return
      if (.not.all(ieee_is_finite(wg)).or..not.all(ieee_is_finite(wb))) return
      v=wv;e=we;g=wg;budgets=wb
      ierr=0
   end subroutine evaluate_electronic_cover

   integer function count_active(q) result(active)
      implicit none
      double precision, intent(in) :: q(8)
      integer :: i
      active=0
      do i=1,8
         if (q(i)>0) active=active+1
      enddo
   end function count_active
end module quick_partition_of_unity_module
