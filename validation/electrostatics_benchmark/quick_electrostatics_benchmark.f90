#include "util.fh"
! (C) Copyright 2026 QUICK contributors. All rights reserved.
! This Source Code Form is subject to the terms of the Mozilla Public
! License, v. 2.0. If a copy of the MPL was not distributed with this
! file, You can obtain one at http://mozilla.org/MPL/2.0/.
!
! Serial, fully linked fixed-density API benchmark. Each timed call includes
! the native API's allocation, upload, kernels, download and classical work.
! The V/E/G API dispatches separate routes: it is not a fused GPU kernel.
program quick_electrostatics_benchmark
   use, intrinsic :: iso_fortran_env, only: int64
   use, intrinsic :: ieee_arithmetic, only: ieee_is_finite, ieee_value, ieee_quiet_nan
   use quick_api_module, only: setQuickJob, getQuickEnergy, getQuickOEPROP, &
      getQuickFieldHessian, deleteQuickJob, quick_api
   use quick_method_module, only: quick_method
   use quick_calculated_module, only: quick_qm_struct
   use quick_basis_module, only: nbasis, quick_basis, itype, ncontract, aexp, dcoeff
   use quick_field_hessian_module, only: get_field_hessian_gpu_count
   implicit none
   character(len=512) :: input_path, density_path, argument
   character(len=1024) :: raw_keyword
   character(len=256) :: keyword
   character(len=80) :: job_name
   character(len=5), parameter :: routes(6) = [character(len=5) :: 'V', 'E', 'G', 'H', 'VEG', 'VEGH']
   integer :: ierr, cleanup, input, output, timings, natoms, ncharges, npoints, repeats
   integer :: atom, point, mu, primitive, phase, iteration, api_before, launches_before(4), launches_after(4)
   integer, allocatable :: numbers(:)
   integer(int64) :: start_tick, stop_tick, tick_rate, job_start, setup_stop, job_stop
   double precision :: energy, elapsed, density_error, fock_error
   double precision, allocatable :: coords(:,:), charges(:,:), points(:,:), v(:), e(:,:), g(:,:,:), h(:,:,:,:)
   double precision, allocatable :: density(:,:), beta(:,:), fock(:,:)
   logical :: converged, hessian_requested

   ierr=0
   if (command_argument_count()<1.or.command_argument_count()>4) call fail(49,'input/density/repeat/output arguments')
   call get_command_argument(1,input_path)
   density_path='-'
   if (command_argument_count()>=2) call get_command_argument(2,density_path)
   repeats=3
   if (command_argument_count()>=3) then
      call get_command_argument(3,argument)
      read(argument,*,iostat=ierr) repeats
      call fail(ierr,'repeat count')
   endif
   if (repeats<1.or.repeats>100) call fail(49,'repeat range 1:100')
   hessian_requested=.true.
   if (command_argument_count()>=4) then
      call get_command_argument(4,argument)
      select case(trim(argument))
      case('VEG')
         hessian_requested=.false.
      case('VEGH')
         hessian_requested=.true.
      case default
         call fail(49,'output must be VEG or VEGH')
      end select
   endif
   open(newunit=input,file=trim(input_path),status='old',action='read',iostat=ierr)
   call fail(ierr,'open input')
   read(input,'(A)',iostat=ierr) raw_keyword
   call fail(ierr,'keyword')
   if (len_trim(raw_keyword)>len(keyword)) call fail(49,'keyword exceeds API capacity')
   keyword=raw_keyword
   read(input,*,iostat=ierr) natoms,ncharges,npoints
   call fail(ierr,'input counts')
   if (natoms<1.or.ncharges<0.or.npoints<1) call fail(49,'positive counts')
   allocate(numbers(natoms),coords(3,natoms),charges(4,ncharges),points(3,npoints), &
      v(npoints),e(3,npoints),g(3,3,npoints),h(3,3,3,npoints),stat=ierr)
   call fail(ierr,'allocate input/output')
   h=ieee_value(0.0d0,ieee_quiet_nan)
   do atom=1,natoms
      read(input,*,iostat=ierr) numbers(atom),coords(:,atom)
      call fail(ierr,'atom')
   enddo
   do point=1,ncharges
      read(input,*,iostat=ierr) charges(:,point)
      call fail(ierr,'charge')
   enddo
   do point=1,npoints
      read(input,*,iostat=ierr) points(:,point)
      call fail(ierr,'probe')
   enddo
   close(input)
   call system_clock(count_rate=tick_rate)
   if (tick_rate<=0) call fail(49,'wall clock resolution')
   call system_clock(job_start)
   job_name='electrostatics_benchmark'
   call setQuickJob(job_name,keyword,natoms,numbers,.false.,ierr)
   call fail(ierr,'setQuickJob')
   call getQuickEnergy(coords,ncharges,charges,energy,ierr)
   call fail(ierr,'initial SCF')
   converged=merge(quick_method%uscf_conv,quick_method%scf_conv,quick_method%unrst)
   if (.not.converged.or..not.ieee_is_finite(energy)) call fail(53,'converged finite SCF')
   if (trim(density_path)/='-') call replace_density(trim(density_path))
   allocate(density(nbasis,nbasis),fock(nbasis,nbasis),stat=ierr)
   call fail(ierr,'state snapshot')
   density=quick_qm_struct%dense
   fock=quick_qm_struct%o
   if (quick_method%unrst) then
      allocate(beta(nbasis,nbasis),stat=ierr)
      call fail(ierr,'beta snapshot')
      beta=quick_qm_struct%denseb
   endif
   api_before=quick_api%step
   call system_clock(setup_stop)
   open(newunit=timings,file='timings.csv',status='new',action='write',iostat=ierr)
   call fail(ierr,'timing output')
   write(timings,'(A)') 'route,iteration,seconds,esp_launches,efield_launches,efg_launches,hessian_launches'
   do phase=1,size(routes)
      if (.not.hessian_requested.and.(phase==4.or.phase==6)) cycle
      do iteration=0,repeats
         call launch_counts(launches_before)
         call system_clock(start_tick)
         select case(phase)
         case(1)
            call getQuickOEPROP(npoints,points,ierr,esp=v)
         case(2)
            call getQuickOEPROP(npoints,points,ierr,efield=e)
         case(3)
            call getQuickOEPROP(npoints,points,ierr,efg=g)
         case(4)
            call getQuickFieldHessian(npoints,points,h,ierr)
         case(5)
            call getQuickOEPROP(npoints,points,ierr,esp=v,efield=e,efg=g)
         case(6)
            call getQuickOEPROP(npoints,points,ierr,esp=v,efield=e,efg=g)
            call fail(ierr,'V/E/G within VEGH')
            call getQuickFieldHessian(npoints,points,h,ierr)
         end select
         call system_clock(stop_tick)
         call fail(ierr,'timed property API')
         call launch_counts(launches_after)
         elapsed=real(stop_tick-start_tick,8)/real(tick_rate,8)
         write(timings,'(A,",",I0,",",ES26.17E3,4(",",I0))') trim(routes(phase)), &
            iteration,elapsed,launches_after-launches_before
      enddo
      flush(timings)
   enddo
   close(timings)
   if (.not.all(ieee_is_finite(v)).or..not.all(ieee_is_finite(e)).or. &
      .not.all(ieee_is_finite(g))) call fail(55,'finite V/E/G output')
   if (hessian_requested.and..not.all(ieee_is_finite(h))) call fail(55,'finite H output')
   density_error=maxval(abs(quick_qm_struct%dense-density))
   if (quick_method%unrst) density_error=max(density_error,maxval(abs(quick_qm_struct%denseb-beta)))
   fock_error=maxval(abs(quick_qm_struct%o-fock))
   open(newunit=output,file='properties.dat',status='new',action='write',iostat=ierr)
   call fail(ierr,'property output')
   write(output,*) npoints
   do point=1,npoints
      write(output,'(I0,43(1X,ES26.17E3))') point,points(:,point),v(point),e(:,point),g(:,:,point),h(:,:,:,point)
   enddo
   close(output)
   call export_density_basis()
   open(newunit=output,file='result.dat',status='new',action='write',iostat=ierr)
   call fail(ierr,'result output')
   write(output,'(A,1X,I0)') 'basis_functions',nbasis
   write(output,'(A,1X,I0)') 'points',npoints
   write(output,'(A,1X,I0)') 'repeats',repeats
   write(output,'(A,1X,I0)') 'extra_API_calls',quick_api%step-api_before
   write(output,'(A,1X,L1)') 'scf_converged',converged
   write(output,'(A,1X,L1)') 'unrestricted',quick_method%unrst
   write(output,'(A,1X,L1)') 'hessian_requested',hessian_requested
   write(output,'(A,1X,ES26.17E3)') 'initial_scf_energy',energy
   write(output,'(A,1X,ES26.17E3)') 'density_error',density_error
   write(output,'(A,1X,ES26.17E3)') 'fock_error',fock_error
   write(output,'(A,1X,ES26.17E3)') 'setup_scf_seconds',real(setup_stop-job_start,8)/real(tick_rate,8)
   cleanup=0
   call deleteQuickJob(cleanup)
   call fail(cleanup,'deleteQuickJob')
   call system_clock(job_stop)
   write(output,'(A,1X,ES26.17E3)') 'setup_to_teardown_seconds',real(job_stop-job_start,8)/real(tick_rate,8)
   close(output)
contains
   subroutine fail(status,label)
      implicit none
      integer,intent(in) :: status
      character(len=*),intent(in) :: label
      if (status==0) return
      write(*,'(A,1X,A,1X,I0)') 'ELECTROSTATICS_BENCHMARK_ERROR',label,status
      stop 1
   end subroutine fail

   subroutine launch_counts(counts)
      implicit none
      integer,intent(out) :: counts(4)
      counts=0
#if defined(CUDA) || defined(CUDA_MPIV)
      call gpu_oeprop_launch_counts(counts(1),counts(2),counts(3))
#endif
      call get_field_hessian_gpu_count(counts(4))
   end subroutine launch_counts

   subroutine replace_density(filename)
      implicit none
      character(len=*),intent(in) :: filename
      integer :: source,dimension
      logical :: unrestricted
      open(newunit=source,file=filename,status='old',action='read',iostat=ierr)
      call fail(ierr,'open density')
      read(source,*,iostat=ierr) dimension,unrestricted
      call fail(ierr,'density header')
      if (dimension/=nbasis.or.(unrestricted.neqv.quick_method%unrst)) call fail(49,'density shape/spin')
      read(source,*,iostat=ierr) quick_qm_struct%dense
      call fail(ierr,'alpha density')
      if (unrestricted) then
         read(source,*,iostat=ierr) quick_qm_struct%denseb
         call fail(ierr,'beta density')
         quick_qm_struct%denseab=quick_qm_struct%dense+quick_qm_struct%denseb
      endif
      close(source)
   end subroutine replace_density

   subroutine export_density_basis()
      implicit none
      integer :: out
      open(newunit=out,file='ao_basis.dat',status='new',action='write',iostat=ierr)
      call fail(ierr,'basis export')
      write(out,*) nbasis
      do mu=1,nbasis
         write(out,'(6(I0,1X))') mu,quick_basis%ncenter(mu),itype(:,mu),ncontract(mu)
         do primitive=1,ncontract(mu)
            write(out,'(2(ES26.17E3,1X))') aexp(primitive,mu),dcoeff(primitive,mu)
         enddo
      enddo
      close(out)
      open(newunit=out,file='density.dat',status='new',action='write',iostat=ierr)
      call fail(ierr,'density export')
      write(out,*) nbasis,quick_method%unrst
      write(out,'(ES26.17E3)') density
      if (quick_method%unrst) write(out,'(ES26.17E3)') beta
      close(out)
      open(newunit=out,file='overlap.dat',status='new',action='write',iostat=ierr)
      call fail(ierr,'overlap export')
      write(out,*) nbasis
      write(out,'(ES26.17E3)') quick_qm_struct%s
      close(out)
   end subroutine export_density_basis
end program quick_electrostatics_benchmark
