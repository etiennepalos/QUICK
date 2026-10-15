#include "util.fh"
!
! (C) Copyright 2026 QUICK contributors. Mozilla Public License, v. 2.0.
! Test-only sampling of the regular and explicit exterior-continuation bases.
!
program quick_ddx_source_convergence_driver
   use quick_ddx_backend_module, only: quick_ddx_backend_type, ddx_backend_initialize, &
      ddx_backend_finalize, ddx_backend_scaled_ylm
   use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
   implicit none
   integer, parameter :: lmax=18, nlm=(lmax+1)**2, np=10
   type(quick_ddx_backend_type) :: ctx
   double precision :: xyz(3,1), radii(1), charges(1), points(3,np), dir(3)
   double precision :: y(nlm), dy(3,nlm), plus(nlm), minus(nlm), numeric(3,nlm,2), point(3)
   double precision, parameter :: h(2)=[1.d-5,3.d-6]
   integer :: ierr, unit, p, mode, l, m, ind, axis, step, do_fd
   character(len=512) :: message
   xyz=0.d0
   radii=2.5d0
   charges=1.d0
   ierr=0
   call ddx_backend_initialize(ctx,2,1,xyz,radii,charges,78.3553d0,lmax,974,1.d-10,ierr,message)
   if(ierr/=0) stop 1
   if(ctx%source_mode/=1) stop 2
   dir=[2.d0,-3.d0,4.d0]/sqrt(29.d0)
   points(:,1)=0.d0
   points(:,2)=0.2d0*radii(1)*dir
   points(:,3)=0.999d0*radii(1)*dir
   points(:,4)=[1.5d0,2.d0,0.d0]
   points(:,5)=1.001d0*radii(1)*dir
   points(:,6)=1.5d0*radii(1)*dir
   points(:,7)=3.d0*radii(1)*dir
   points(:,8)=(1.d0-1.d-8)*radii(1)*dir
   points(:,9)=(1.d0+1.d-8)*radii(1)*dir
   points(:,10)=1.d40*radii(1)*dir
   open(newunit=unit,file='source_basis.csv',status='new')
   write(unit,'(A)') 'mode,point,l,m,fd_enabled,x,y,z,value,gx,gy,gz,fd1x,fd1y,fd1z,fd2x,fd2y,fd2z'
   do mode=1,2
      ctx%source_mode=mode
      do p=1,np
         if(mode==1.and.p==10) cycle
         do_fd=1
         if(p==4.or.p==8.or.p==9.or.p==10) do_fd=0
         call ddx_backend_scaled_ylm(ctx,points(:,p),1,y,ierr,message,dy)
         if(ierr/=0) stop 3
         if(.not.all(ieee_is_finite(y)).or..not.all(ieee_is_finite(dy))) stop 4
         numeric=0.d0
         if(do_fd==1) then
            do step=1,2
               do axis=1,3
                  point=points(:,p)
                  point(axis)=point(axis)+h(step)
                  call ddx_backend_scaled_ylm(ctx,point,1,plus,ierr,message)
                  if(ierr/=0) stop 5
                  point=points(:,p)
                  point(axis)=point(axis)-h(step)
                  call ddx_backend_scaled_ylm(ctx,point,1,minus,ierr,message)
                  if(ierr/=0) stop 6
                  numeric(axis,:,step)=(plus-minus)/(2.d0*h(step))
               enddo
            enddo
         endif
         do l=0,lmax
            do m=-l,l
               ind=l*(l+1)+m+1
               write(unit,'(5(I0,","),12(ES26.17E3,","),ES26.17E3)') &
                  mode,p,l,m,do_fd,points(:,p),y(ind),dy(:,ind),numeric(:,ind,1),numeric(:,ind,2)
            enddo
         enddo
      enddo
   enddo
   close(unit)
   ctx%source_mode=3
   call ddx_backend_scaled_ylm(ctx,points(:,2),1,y,ierr,message,dy)
   if(ierr==0.or.any(y/=0.d0).or.any(dy/=0.d0)) stop 7
   call ddx_backend_finalize(ctx)
   if(ctx%source_mode/=1) stop 8
   write(*,'(A)') 'source_basis_complete default_reset_and_invalid_mode_pass'
end program quick_ddx_source_convergence_driver
