!
! (C) Copyright 2026 QUICK contributors. All rights reserved.
! This Source Code Form is subject to the terms of the Mozilla Public
! License, v. 2.0. If a copy of the MPL was not distributed with this
! file, You can obtain one at https://mozilla.org/MPL/2.0/.
!
#include "util.fh"
program quick_sg1_export_driver
   use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
   use quick_api_module, only: setQuickJob,getQuickEnergyGradients,deleteQuickJob
   use quick_gridpoints_module, only: quick_dft_grid
   use quick_method_module, only: quick_method
   use quick_basis_module, only: nbasis,quick_basis,itype
   use quick_calculated_module, only: quick_qm_struct
   implicit none
   character(len=256) :: keyword,filename
   character(len=80) :: jobname
   character(len=300) :: raw
   integer :: ierr,unit,natom,i,j,cleanup
   integer,allocatable :: numbers(:)
   double precision,allocatable :: coordinates(:,:),gradient(:,:)
   double precision :: energy,charges(4,0),charge_gradient(3,0)
   if(command_argument_count()/=1) stop 2
   call get_command_argument(1,filename)
   open(newunit=unit,file=trim(filename),status='old',action='read',iostat=ierr)
   if(ierr/=0) stop 2
   read(unit,'(A)') raw
   if(len_trim(raw)>len(keyword)) stop 2
   keyword=raw
   read(unit,*) natom
   allocate(numbers(natom),coordinates(3,natom),gradient(3,natom))
   do i=1,natom
      read(unit,*) numbers(i),coordinates(:,i)
   enddo
   close(unit)
   jobname='matched_sg1'
   call setQuickJob(jobname,keyword,natom,numbers,.false.,ierr)
   if(ierr/=0) stop 3
   call getQuickEnergyGradients(coordinates,0,charges,energy,gradient,charge_gradient,ierr)
   if(ierr/=0.or..not.quick_method%scf_conv.or..not.ieee_is_finite(energy)) stop 3
   if(.not.all(ieee_is_finite(gradient))) stop 3
   if(.not.allocated(quick_dft_grid%gridxb).or.quick_dft_grid%gridb_count<1) stop 4
   open(newunit=unit,file='grid.dat',status='new',action='write')
   write(unit,'(I0)') quick_dft_grid%gridb_count
   do i=1,quick_dft_grid%gridb_count
      write(unit,'(4(ES26.17E3,1X),I0,1X,ES26.17E3)') quick_dft_grid%gridxb(i), &
         quick_dft_grid%gridyb(i),quick_dft_grid%gridzb(i),quick_dft_grid%gridb_weight(i), &
         quick_dft_grid%gridb_atm(i),quick_dft_grid%gridb_sswt(i)
   enddo
   close(unit)
   open(newunit=unit,file='result.dat',status='new',action='write')
   write(unit,'(A,1X,ES26.17E3)') 'energy',energy
   write(unit,'(A,1X,ES26.17E3)') 'matrix_cutoff',quick_method%DMCutoff
   write(unit,'(A,1X,ES26.17E3)') 'basis_cutoff',quick_method%basisCutoff
   write(unit,'(A,1X,ES26.17E3)') 'xc_cutoff',quick_method%XCCutoff
   do i=1,natom
      write(unit,'(A,1X,I0,3(1X,ES26.17E3))') 'gradient',i,gradient(:,i)
   enddo
   close(unit)
   open(newunit=unit,file='density.dat',status='new',action='write')
   write(unit,'(I0)') nbasis
   do i=1,nbasis
      write(unit,'(4(I0,1X))') quick_basis%ncenter(i),itype(:,i)
   enddo
   do j=1,nbasis
      write(unit,'(*(ES26.17E3,1X))') quick_qm_struct%dense(:,j)
   enddo
   close(unit)
   call deleteQuickJob(cleanup)
   if(cleanup/=0) stop 5
end program quick_sg1_export_driver
