#include "util.fh"
! (C) Copyright 2026 QUICK contributors. Mozilla Public License, v. 2.0.
! Test-only public API/raw-kernel client. Observation coordinates are bohr;
! QM/MM input coordinates are Angstrom. No production fixture is modified.
program quick_field_hessian_api_driver
   use, intrinsic :: ieee_arithmetic, only: ieee_is_finite, ieee_value, ieee_quiet_nan
   use quick_api_module, only: setQuickJob, getQuickEnergy, getQuickOEPROP, getQuickFieldHessian, &
      deleteQuickJob, quick_api
   use quick_method_module, only: quick_method
   use quick_basis_module, only: nbasis, quick_basis, itype, ncontract, aexp, dcoeff
   use quick_calculated_module, only: quick_qm_struct
   use quick_molspec_module, only: xyz
   use quick_field_hessian_module, only: field_hessian_cpu, get_field_hessian_gpu_count
#ifdef MPIV
   use mpi
   use quick_api_module, only: setQuickMPI
   use quick_mpi_module, only: quick_comm, quick_comm_rank, quick_mpi_error
#endif
   implicit none
   character(len=300) :: input_file, density_file
   character(len=256) :: keyword
   character(len=80) :: jobname
   character(len=32) :: mode
   integer :: ierr, cleanup, unit, natoms, ncharges, npoints, atom, charge, point, first, last
   integer :: count_before, count_after, before_step, status_before, status_zero, status_shape, status_nan, status_singular
   integer :: status_after, direction, trial, saved_n
   integer, allocatable :: numbers(:)
   double precision, allocatable :: coords(:,:), charges(:,:), points(:,:), displaced(:,:)
   double precision, allocatable :: v(:),e(:,:),g(:,:,:),v_after(:),e_after(:,:),g_after(:,:,:)
   double precision, allocatable :: h(:,:,:,:),h_repeat(:,:,:,:),h_chunk(:,:,:,:),scratch(:,:,:,:)
   double precision, allocatable :: g_plus(:,:,:),g_minus(:,:,:),saved_density(:,:),saved_fock(:,:),saved_beta(:,:)
   double precision :: energy, steps(2), step, repeat_error, chunk_error, density_error, fock_error, property_error(3)
   logical :: is_master, converged

   ierr=0
   is_master=.true.
#ifdef MPIV
   call MPI_INIT(quick_mpi_error)
   call setQuickMPI(MPI_COMM_WORLD,ierr)
   is_master=quick_comm_rank==0
#endif
   if (command_argument_count()<2) call fail(49,'arguments')
   call get_command_argument(1,mode)
   call get_command_argument(2,input_file)
   if (trim(mode)=='raw') then
      call raw_case()
#ifdef MPIV
      call MPI_FINALIZE(quick_mpi_error)
#endif
      stop
   endif
   if (trim(mode)/='api') call fail(49,'mode')
   density_file='-'
   if (command_argument_count()>=3) call get_command_argument(3,density_file)
   open(newunit=unit,file=trim(input_file),status='old',action='read',iostat=ierr)
   call fail(ierr,'open input')
   read(unit,'(A)',iostat=ierr) keyword
   call fail(ierr,'keyword')
   read(unit,*,iostat=ierr) natoms,ncharges,npoints
   call fail(ierr,'counts')
   if (natoms<1.or.ncharges<0.or.npoints<1) call fail(49,'positive counts')
   allocate(numbers(natoms),coords(3,natoms),charges(4,ncharges),points(3,npoints),displaced(3,npoints), &
      v(npoints),e(3,npoints),g(3,3,npoints),v_after(npoints),e_after(3,npoints),g_after(3,3,npoints), &
      h(3,3,3,npoints),h_repeat(3,3,3,npoints),h_chunk(3,3,3,npoints),scratch(3,3,3,npoints), &
      g_plus(3,3,npoints),g_minus(3,3,npoints),stat=ierr)
   call fail(ierr,'allocate')
   do atom=1,natoms
      read(unit,*,iostat=ierr) numbers(atom),coords(:,atom)
      call fail(ierr,'atom')
   enddo
   do charge=1,ncharges
      read(unit,*,iostat=ierr) charges(:,charge)
      call fail(ierr,'charge')
   enddo
   do point=1,npoints
      read(unit,*,iostat=ierr) points(:,point)
      call fail(ierr,'probe')
   enddo
   close(unit)
   jobname='field_hessian_api'
   call setQuickJob(jobname,keyword,natoms,numbers,.false.,ierr)
   call fail(ierr,'setQuickJob')
   scratch=12.5d0
   call getQuickFieldHessian(npoints,points,scratch,status_before)
   call getQuickEnergy(coords,ncharges,charges,energy,ierr)
   call fail(ierr,'SCF')
   converged=merge(quick_method%uscf_conv,quick_method%scf_conv,quick_method%unrst)
   if (.not.converged.or..not.ieee_is_finite(energy)) call fail(53,'converged finite SCF')
   if (trim(density_file)/='-') call replace_density(trim(density_file))
   allocate(saved_density(nbasis,nbasis),saved_fock(nbasis,nbasis),stat=ierr)
   call fail(ierr,'state snapshots')
   saved_density=quick_qm_struct%dense
   saved_fock=quick_qm_struct%o
   if (quick_method%unrst) then
      allocate(saved_beta(nbasis,nbasis),stat=ierr)
      call fail(ierr,'beta snapshot')
      saved_beta=quick_qm_struct%denseb
   endif
   before_step=quick_api%step
   call getQuickOEPROP(npoints,points,ierr,esp=v,efield=e,efg=g)
   call fail(ierr,'old properties before H')
   call get_field_hessian_gpu_count(count_before)
   call getQuickFieldHessian(npoints,points,h,ierr)
   call fail(ierr,'field Hessian')
   call get_field_hessian_gpu_count(count_after)
   call getQuickFieldHessian(npoints,points,h_repeat,ierr)
   call fail(ierr,'repeat field Hessian')
   repeat_error=maxval(abs(h_repeat-h))
   do first=1,npoints,17
      last=min(npoints,first+16)
      call getQuickFieldHessian(last-first+1,points(:,first:last),h_chunk(:,:,:,first:last),ierr)
      call fail(ierr,'chunk field Hessian')
   enddo
   chunk_error=maxval(abs(h_chunk-h))
   call getQuickOEPROP(npoints,points,ierr,esp=v_after,efield=e_after,efg=g_after)
   call fail(ierr,'old properties after H')
   property_error=(/maxval(abs(v_after-v)),maxval(abs(e_after-e)),maxval(abs(g_after-g))/)
   density_error=maxval(abs(quick_qm_struct%dense-saved_density))
   if (quick_method%unrst) density_error=max(density_error,maxval(abs(quick_qm_struct%denseb-saved_beta)))
   fock_error=maxval(abs(quick_qm_struct%o-saved_fock))
   if (.not.all(ieee_is_finite(h))) call fail(53,'finite Hessian')
   if (is_master) then
      open(newunit=unit,file='properties.dat',status='new',action='write',iostat=ierr)
      call fail(ierr,'open properties')
      write(unit,*) npoints
      do point=1,npoints
         write(unit,'(I0,43(1X,ES26.17E3))') point,points(:,point),v(point),e(:,point),g(:,:,point),h(:,:,:,point)
      enddo
      close(unit)
      open(newunit=unit,file='finite_difference.dat',status='new',action='write',iostat=ierr)
      call fail(ierr,'open finite differences')
   endif
   steps=(/1.d-3,5.d-4/)
   do trial=1,2
      step=steps(trial)
      do direction=1,3
         displaced=points
         displaced(direction,:)=points(direction,:)+step
         call getQuickOEPROP(npoints,displaced,ierr,efg=g_plus)
         call fail(ierr,'positive displaced EFG')
         displaced(direction,:)=points(direction,:)-step
         call getQuickOEPROP(npoints,displaced,ierr,efg=g_minus)
         call fail(ierr,'negative displaced EFG')
         if (is_master) then
            do point=1,npoints
               write(unit,'(ES26.17E3,2(1X,I0),9(1X,ES26.17E3))') step,direction,point, &
                  (g_plus(:,:,point)-g_minus(:,:,point))/(2.d0*step)
            enddo
         endif
      enddo
   enddo
   if (is_master) close(unit)
   scratch=12.5d0
   call getQuickFieldHessian(0,points(:,:0),scratch(:,:,:,:0),status_zero)
   call getQuickFieldHessian(npoints,points,scratch(:2,:,:,:),status_shape)
   displaced=points
   displaced(1,1)=ieee_value(0.d0,ieee_quiet_nan)
   call getQuickFieldHessian(npoints,displaced,scratch,status_nan)
   displaced=points
   displaced(:,1)=xyz(:,1)
   call getQuickFieldHessian(npoints,displaced,scratch,status_singular)
   saved_n=nbasis
   if (is_master) then
      call export_basis_density()
      open(newunit=unit,file='result.dat',status='new',action='write',iostat=ierr)
      call fail(ierr,'open result')
      write(unit,'(A,1X,ES26.17E3)') 'energy',energy
      write(unit,'(A,1X,I0)') 'basis_functions',nbasis
      write(unit,'(A,1X,L1)') 'unrestricted',quick_method%unrst
      write(unit,'(A,1X,L1)') 'scf_converged',converged
      write(unit,'(A,1X,I0)') 'main_native_gpu_launches',count_after-count_before
      write(unit,'(A,1X,I0)') 'extra_API_calls',quick_api%step-before_step
      write(unit,'(A,1X,ES26.17E3)') 'density_error',density_error
      write(unit,'(A,1X,ES26.17E3)') 'fock_error',fock_error
      write(unit,'(A,3(1X,ES26.17E3))') 'old_property_errors',property_error
      write(unit,'(A,1X,ES26.17E3)') 'repeat_error',repeat_error
      write(unit,'(A,1X,ES26.17E3)') 'chunk_error',chunk_error
      write(unit,'(A,5(1X,I0))') 'error_statuses',status_before,status_zero,status_shape,status_nan,status_singular
      write(unit,'(A,1X,ES26.17E3)') 'error_output_error',maxval(abs(scratch-12.5d0))
      close(unit)
   endif
   cleanup=0
   call deleteQuickJob(cleanup)
   call fail(cleanup,'delete job')
   call getQuickFieldHessian(npoints,points,scratch,status_after)
   if (is_master) then
      open(newunit=unit,file='result.dat',status='old',position='append',action='write',iostat=ierr)
      call fail(ierr,'append lifecycle')
      write(unit,'(A,1X,I0)') 'post_delete_status',status_after
      write(unit,'(A,1X,ES26.17E3)') 'post_delete_output_error',maxval(abs(scratch-12.5d0))
      close(unit)
   endif
#ifdef MPIV
   call MPI_FINALIZE(quick_mpi_error)
#endif
contains
   subroutine fail(status,label)
      implicit none
      integer,intent(in) :: status
      character(len=*),intent(in) :: label
      if (status==0) return
      if (is_master) write(*,'(A,1X,A,1X,I0)') 'FIELD_HESSIAN_VALIDATION_ERROR',label,status
#ifdef MPIV
      call MPI_ABORT(quick_comm,max(1,abs(status)),quick_mpi_error)
#endif
      stop 1
   end subroutine fail

   subroutine replace_density(filename)
      implicit none
      character(len=*),intent(in) :: filename
      integer :: input,dimension
      logical :: unrestricted
      if (filename=='ZERO') then
         quick_qm_struct%dense=0.d0
         if (quick_method%unrst) then
            quick_qm_struct%denseb=0.d0
            quick_qm_struct%denseab=0.d0
         endif
         return
      endif
      open(newunit=input,file=filename,status='old',action='read',iostat=ierr)
      call fail(ierr,'open replacement density')
      read(input,*,iostat=ierr) dimension,unrestricted
      call fail(ierr,'replacement density header')
      if (dimension/=nbasis.or.(unrestricted.neqv.quick_method%unrst)) call fail(49,'replacement density dimension')
      read(input,*,iostat=ierr) quick_qm_struct%dense
      call fail(ierr,'replacement alpha density')
      if (unrestricted) then
         read(input,*,iostat=ierr) quick_qm_struct%denseb
         call fail(ierr,'replacement beta density')
         quick_qm_struct%denseab=quick_qm_struct%dense+quick_qm_struct%denseb
      endif
      close(input)
   end subroutine replace_density

   subroutine export_basis_density()
      implicit none
      integer :: out,mu,primitive
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
      write(out,'(ES26.17E3)') quick_qm_struct%dense
      if (quick_method%unrst) write(out,'(ES26.17E3)') quick_qm_struct%denseb
      close(out)
      open(newunit=out,file='overlap.dat',status='new',action='write',iostat=ierr)
      call fail(ierr,'overlap export')
      write(out,*) nbasis
      write(out,'(ES26.17E3)') quick_qm_struct%s
      close(out)
   end subroutine export_basis_density

   subroutine raw_case()
      implicit none
      integer :: nb,na,mp,np,out,mu,primitive,status,count_a,count_b
      integer,allocatable :: nc(:),angular(:,:),center(:),owned(:)
      double precision,allocatable :: exponent(:,:),coefficient(:,:),atoms(:,:),density(:,:),probe(:,:),result(:,:)
      double precision :: cutoff,seed
      open(newunit=out,file=trim(input_file),status='old',action='read',iostat=ierr)
      call fail(ierr,'raw input')
      read(out,*,iostat=ierr) nb,na,mp,np
      call fail(ierr,'raw sizes')
      read(out,*,iostat=ierr) cutoff,seed
      call fail(ierr,'raw controls')
      if (nb<1.or.na<1.or.mp<1.or.np<0) call fail(49,'raw allocation sizes')
      allocate(nc(nb),angular(3,nb),center(nb),owned(nb),exponent(mp,nb),coefficient(mp,nb), &
         atoms(3,na),density(nb,nb),probe(3,np),result(27,np),stat=ierr)
      call fail(ierr,'raw allocate')
      do mu=1,na
         read(out,*,iostat=ierr) atoms(:,mu)
         call fail(ierr,'raw atom')
      enddo
      do mu=1,nb
         read(out,*,iostat=ierr) center(mu),angular(:,mu),nc(mu),owned(mu)
         call fail(ierr,'raw AO')
         do primitive=1,mp
            read(out,*,iostat=ierr) exponent(primitive,mu),coefficient(primitive,mu)
            call fail(ierr,'raw primitive')
         enddo
      enddo
      read(out,*,iostat=ierr) density
      call fail(ierr,'raw density')
      do mu=1,np
         read(out,*,iostat=ierr) probe(:,mu)
         call fail(ierr,'raw probe')
      enddo
      close(out)
      result=seed
      call get_field_hessian_gpu_count(count_a)
#if defined(CUDA) || defined(CUDA_MPIV)
      call gpu_field_hessian(nb,na,mp,nc,angular,center,exponent,coefficient,atoms,density,np,probe,owned,cutoff,result,status)
#else
      call field_hessian_cpu(nb,na,mp,nc,angular,center,exponent,coefficient,atoms,density,np,probe,owned,cutoff,result,status)
#endif
      call get_field_hessian_gpu_count(count_b)
      if (is_master) then
         open(newunit=out,file='raw_result.dat',status='new',action='write',iostat=ierr)
         call fail(ierr,'raw output')
         write(out,*) status,count_b-count_a,np
         write(out,'(ES26.17E3)') result
         close(out)
      endif
   end subroutine raw_case
end program quick_field_hessian_api_driver
