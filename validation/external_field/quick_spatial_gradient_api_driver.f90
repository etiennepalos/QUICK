#include "util.fh"
! (C) Copyright 2026 QUICK contributors. Mozilla Public License, v. 2.0.
! Test-only client. All operator inspections save and restore the live Fock
! matrix and method state, and happen after the requested energy/force call.
program quick_spatial_gradient_api_driver
   use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
   use quick_api_module, only: setQuickJob, getQuickEnergy, getQuickEnergyGradients, deleteQuickJob, quick_api
   use quick_method_module, only: quick_method, quick_method_type, init, read, check
   use quick_basis_module, only: nbasis, quick_basis, itype, ncontract, aexp, dcoeff
   use quick_calculated_module, only: quick_qm_struct
   use quick_oei_module, only: add_external_field_operator, get_external_field_gpu_counts
   implicit none
   character(len=300) :: filename, raw_keyword
   character(len=256) :: keyword
   character(len=80) :: jobname
   character(len=32) :: mode
   type(quick_method_type) :: method
   integer :: ierr, cleanup_error, unit, natoms, ncharges, atom, charge, i, j, p
   integer :: counts_before(2), counts_after(2)
   integer, allocatable :: numbers(:)
   double precision, allocatable :: coords(:,:), charges(:,:), gradients(:,:), charge_gradients(:,:)
   double precision :: energy
   logical :: want_gradient, converged

   ierr=0
   if (command_argument_count() /= 2) stop 1
   call get_command_argument(1,mode)
   call get_command_argument(2,filename)
   if (trim(mode)/='energy' .and. trim(mode)/='gradient' .and. trim(mode)/='options') stop 1
   open(newunit=unit,file=trim(filename),status='old',action='read',iostat=ierr)
   if (ierr/=0) stop 1
   read(unit,'(A)',iostat=ierr) raw_keyword
   if (ierr/=0 .or. len_trim(raw_keyword)>len(keyword)) stop 1
   keyword=raw_keyword
   if (trim(mode)=='options') then
      close(unit)
      call init(method,ierr)
      call read(method,raw_keyword,ierr)
      open(newunit=unit,status='scratch',action='readwrite')
      if (ierr==0) call check(method,unit,ierr)
      close(unit)
      if (ierr/=0) then
         write(*,'(A,I0)') 'options_ierr ',ierr
         stop 1
      endif
      open(newunit=unit,file='result.dat',status='new',action='write',iostat=ierr)
      if (ierr/=0) stop 1
      call write_options(unit,method)
      close(unit)
      stop
   endif
   read(unit,*,iostat=ierr) natoms,ncharges
   if (ierr/=0 .or. natoms<1 .or. ncharges<0) stop 1
   allocate(numbers(natoms),coords(3,natoms),charges(4,ncharges), &
      gradients(3,natoms),charge_gradients(3,ncharges),stat=ierr)
   if (ierr/=0) stop 1
   do atom=1,natoms
      read(unit,*,iostat=ierr) numbers(atom),coords(:,atom)
      if (ierr/=0) stop 1
   enddo
   do charge=1,ncharges
      read(unit,*,iostat=ierr) charges(:,charge)
      if (ierr/=0) stop 1
   enddo
   close(unit)
   want_gradient=trim(mode)=='gradient'
   jobname='spatial_gradient_api'
   call setQuickJob(jobname,keyword,natoms,numbers,.false.,ierr)
   ! The public API honors the job card: calling the gradient entry point
   ! alone does not enable the method's gradient calculation.
   if (ierr==0 .and. want_gradient) then
      if (.not.quick_method%grad .or. .not.quick_method%analGrad) ierr=53
   endif
   if (ierr==0) then
      call get_external_field_gpu_counts(counts_before(1),counts_before(2))
      if (want_gradient) then
         call getQuickEnergyGradients(coords,ncharges,charges,energy,gradients,charge_gradients,ierr)
      else
         call getQuickEnergy(coords,ncharges,charges,energy,ierr)
      endif
      call get_external_field_gpu_counts(counts_after(1),counts_after(2))
      converged=merge(quick_method%uscf_conv,quick_method%scf_conv,quick_method%unrst)
      if (.not.converged .or. .not.ieee_is_finite(energy)) ierr=53
      if (want_gradient) then
         if (.not.all(ieee_is_finite(gradients)) .or. .not.all(ieee_is_finite(charge_gradients))) ierr=53
      endif
      if (ierr==0) then
         open(newunit=unit,file='result.dat',status='new',action='write',iostat=ierr)
         if (ierr==0) then
            write(unit,'(A,1X,ES26.17E3)') 'energy',energy
            write(unit,'(A,1X,L1)') 'scf_converged',converged
            write(unit,'(A,1X,I0)') 'basis_functions',nbasis
            write(unit,'(A,1X,I0)') 'api_steps',quick_api%step
            write(unit,'(A,1X,L1)') 'analytic_gradient_active',quick_method%grad.and.quick_method%analGrad
            write(unit,'(A,2(1X,I0))') 'native_gpu_field_counts',counts_after-counts_before
            call write_options(unit,quick_method)
            if (want_gradient) then
               do atom=1,natoms
                  write(unit,'(A,1X,I0,3(1X,ES26.17E3))') 'qm_gradient',atom,gradients(:,atom)
               enddo
               do charge=1,ncharges
                  write(unit,'(A,1X,I0,3(1X,ES26.17E3))') 'charge_gradient',charge,charge_gradients(:,charge)
               enddo
            endif
            close(unit)
            call export_operators(ierr)
         endif
      endif
   endif
   cleanup_error=0
   call deleteQuickJob(cleanup_error)
   if (ierr==0) ierr=cleanup_error
   if (ierr/=0) then
      write(*,'(A,I0)') 'Spatial gradient API driver failed, ierr=',ierr
      stop 1
   endif
contains
   subroutine write_options(out,options)
      implicit none
      integer,intent(in) :: out
      type(quick_method_type),intent(in) :: options
      integer :: axis
      write(out,'(A,1X,L1)') 'external_field_active',options%external_efield
      write(out,'(A,3(1X,ES26.17E3))') 'external_field',options%external_efield_vector
      write(out,'(A,3(1X,ES26.17E3))') 'external_origin',options%external_efield_origin
      do axis=1,3
         write(out,'(A,1X,I0,3(1X,ES26.17E3))') 'external_tensor',axis,options%external_efield_tensor(axis,:)
      enddo
   end subroutine write_options

   subroutine export_operators(status)
      implicit none
      integer,intent(out) :: status
      integer :: out,mu,nu,primitive,component,left(6),right(6)
      double precision,allocatable :: saved_o(:,:), matrix(:,:),density(:,:)
      double precision :: saved_field(3),saved_tensor(3,3)
      logical :: saved_active
      status=0
      allocate(saved_o(nbasis,nbasis),matrix(nbasis,nbasis),density(nbasis,nbasis),stat=status)
      if (status/=0) return
      saved_o=quick_qm_struct%o
      saved_field=quick_method%external_efield_vector
      saved_tensor=quick_method%external_efield_tensor
      saved_active=quick_method%external_efield
      density=quick_qm_struct%dense
      if (quick_method%unrst) density=density+quick_qm_struct%denseb
      quick_qm_struct%o=0.d0
      call add_external_field_operator(.false.)
      matrix=quick_qm_struct%o
      do mu=1,nbasis
         do nu=1,mu-1
            matrix(nu,mu)=matrix(mu,nu)
         enddo
      enddo
      open(newunit=out,file='operators.dat',status='new',action='write',iostat=status)
      if (status==0) then
         write(out,*) nbasis
         do mu=1,nbasis
            do nu=1,nbasis
               write(out,'(2(I0,1X),3(ES26.17E3,1X))') mu,nu,quick_qm_struct%s(mu,nu),density(mu,nu),matrix(mu,nu)
            enddo
         enddo
         close(out)
      endif
      ! Six unit-tensor operators directly test diagonal 1/2 and off-diagonal
      ! factors against independent quadrupole integrals, without extra SCFs.
      left=(/1,2,3,1,1,2/)
      right=(/1,2,3,2,3,3/)
      if (status==0) open(newunit=out,file='tensor_operators.dat',status='new',action='write',iostat=status)
      if (status==0) then
         write(out,*) nbasis
         quick_method%external_efield=.true.
         quick_method%external_efield_vector=0.d0
         do component=1,6
            quick_method%external_efield_tensor=0.d0
            quick_method%external_efield_tensor(left(component),right(component))=1.d0
            quick_method%external_efield_tensor(right(component),left(component))=1.d0
            quick_qm_struct%o=0.d0
            call add_external_field_operator(.false.)
            do mu=1,nbasis
               do nu=1,mu
                  write(out,'(3(I0,1X),ES26.17E3)') component,mu,nu,quick_qm_struct%o(mu,nu)
               enddo
            enddo
         enddo
         close(out)
      endif
      quick_qm_struct%o=saved_o
      quick_method%external_efield_vector=saved_field
      quick_method%external_efield_tensor=saved_tensor
      quick_method%external_efield=saved_active
      if (status/=0) return
      open(newunit=out,file='ao_basis.dat',status='new',action='write',iostat=status)
      if (status/=0) return
      write(out,*) nbasis
      do mu=1,nbasis
         write(out,'(6(I0,1X))') mu,quick_basis%ncenter(mu),itype(:,mu),ncontract(mu)
         do primitive=1,ncontract(mu)
            write(out,'(2(ES26.17E3,1X))') aexp(primitive,mu),dcoeff(primitive,mu)
         enddo
      enddo
      close(out)
   end subroutine export_operators
end program quick_spatial_gradient_api_driver
