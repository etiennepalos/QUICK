#include "util.fh"
! (C) Copyright 2026 QUICK contributors. Mozilla Public License, v. 2.0.
! Test-only serial/MPI client: successive jobs must reset optional XC grids.
program quick_xc_grid_lifecycle_driver
   use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
   use quick_api_module, only: setQuickJob, getQuickEnergyGradients, getQuickOEPROP, deleteQuickJob, quick_api
   use quick_method_module, only: quick_method, quick_method_type, init, read, check
   use quick_gridpoints_module, only: quick_dft_grid, quick_xcg_tmp
   use quick_calculated_module, only: quick_qm_struct
   use quick_basis_module, only: nbasis
#ifdef MPIV
   use mpi
   use quick_api_module, only: setQuickMPI
   use quick_mpi_module, only: quick_comm, quick_comm_rank, quick_comm_size
#endif
   implicit none
   integer, parameter :: njobs=8
   character(len=20), parameter :: labels(njobs) = [character(len=20) :: &
      'default-1','refined-75-302','default-2','refined-100-434','refined-50-194','default-3','sg0-1','sg0-2']
   integer, parameter :: radial(njobs)=[0,75,0,100,50,0,0,0], angular(njobs)=[0,302,0,434,194,0,0,0]
   character(len=32) :: mode
   character(len=80) :: jobname
   character(len=256) :: keyword,filename
   character(len=300) :: raw
   type(quick_method_type) :: options
   integer :: ierr,cleanup,job,out,localout,atom,rank,nranks,nmin,nmax,api_before,grid_count,packed_count
   integer :: numbers(3)=[8,1,1]
   double precision :: coordinates(3,3),gradient(3,3),energy,charges(4,0),charge_gradient(3,0)
   double precision :: points(3,1),efield(3,1),efg(3,3,1),density_error,fock_error
   double precision,allocatable :: density(:,:),fock(:,:)
   logical :: custom,is_master
   rank=0;nranks=1;ierr=0
#ifdef MPIV
   call MPI_INIT(ierr)
   call setQuickMPI(MPI_COMM_WORLD,ierr)
   if(ierr/=0) call abort_test('setQuickMPI')
   rank=quick_comm_rank;nranks=quick_comm_size
#endif
   is_master=rank==0
   mode='lifecycle'
   if(command_argument_count()>=1) call get_command_argument(1,mode)
   if(trim(mode)=='options') then
      if(command_argument_count()/=2) call abort_test('options filename argument')
      call get_command_argument(2,filename)
      open(newunit=out,file=trim(filename),status='old',action='read',iostat=ierr)
      if(ierr/=0) call abort_test('open options')
      read(out,'(A)',iostat=ierr) raw
      close(out)
      if(ierr/=0.or.len_trim(raw)>256) call abort_test('read options')
      call init(options,ierr)
      call read(options,raw,ierr)
      open(newunit=out,status='scratch',action='readwrite')
      if(ierr==0) call check(options,out,ierr)
      close(out)
      if(is_master) then
         write(*,'(A,I0)') 'options_ierr ',ierr
         write(*,'(A,L1)') 'options_custom ',options%xc_custom_grid
         write(*,'(A,2(I0,1X))') 'options_sizes ',options%xc_radial,options%xc_angular
      endif
   else if(trim(mode)=='lifecycle') then
      coordinates(:,1)=[-.33840d0,.00380d0,.23923d0]
      coordinates(:,2)=[-.33510d0,-.00190d0,-.83277d0]
      coordinates(:,3)=[.67350d0,-.00190d0,.59353d0]
      points(:,1)=[.4d0,-.3d0,3.8d0]
      if(is_master) then
         open(newunit=out,file='lifecycle.csv',status='new',action='write',iostat=ierr)
         if(ierr/=0) call abort_test('open lifecycle')
         write(out,'(A)') 'job,label,custom,nrad,nang,init_ngpts,packed_ngpts,api_steps,extra_API_calls,energy,'// &
            'g1x,g1y,g1z,g2x,g2y,g2z,g3x,g3y,g3z,ex,ey,ez,'// &
            'gxx,gyx,gzx,gxy,gyy,gzy,gxz,gyz,gzz,density_error,fock_error'
      endif
      write(filename,'(A,I0,A)') 'rank-',rank,'.csv'
      open(newunit=localout,file=trim(filename),status='new',action='write',iostat=ierr)
      if(ierr/=0) call abort_test('open rank lifecycle')
      write(localout,'(A)') 'job,rank,init_ngpts,packed_ngpts,rad_gps,ang_gps,tmp_arrays_released'
      do job=1,njobs
         keyword='PBE0 BASIS=STO-3G GRADIENT SCF=400 DENSERMS=1e-10 CUTOFF=1e-12 MATRIXZERO=1e-12 XCCUTOFF=1e-12'
         custom=radial(job)>0
         if(custom) then
            write(raw,'(A,I0,A,I0)') ' XC_RADIAL=',radial(job),' XC_ANGULAR=',angular(job)
            keyword=trim(keyword)//trim(raw)
         endif
         if(job>=7) keyword=trim(keyword)//' SG0'
         write(jobname,'(A,I2.2)') 'xc_lifecycle_',job
         call setQuickJob(jobname,keyword,3,numbers,.false.,ierr)
         if(ierr/=0) call abort_test('setQuickJob')
         call validate_method()
         call getQuickEnergyGradients(coordinates,0,charges,energy,gradient,charge_gradient,ierr)
         if(ierr/=0) call abort_test('energy/gradient')
         if(.not.quick_method%scf_conv.or..not.ieee_is_finite(energy).or..not.all(ieee_is_finite(gradient))) &
            call abort_test('converged finite result')
         call validate_method() ! SAD must restore the complete DFT request.
         grid_count=quick_dft_grid%init_ngpts;packed_count=quick_dft_grid%gridb_count
         nmin=grid_count;nmax=grid_count
#ifdef MPIV
         call MPI_ALLREDUCE(grid_count,nmin,1,MPI_INTEGER,MPI_MIN,quick_comm,ierr)
         if(ierr/=0) call abort_test('minimum count reduction')
         call MPI_ALLREDUCE(grid_count,nmax,1,MPI_INTEGER,MPI_MAX,quick_comm,ierr)
         if(ierr/=0) call abort_test('maximum count reduction')
#endif
         if(nmin/=nmax.or.nmin<1.or.packed_count<1.or.packed_count>grid_count) call abort_test('per-rank grid counts')
         if(grid_count>3*quick_xcg_tmp%rad_gps*quick_xcg_tmp%ang_gps) call abort_test('grid bounds')
         if(custom.and.grid_count/=3*radial(job)*angular(job)) call abort_test('unpruned initial count')
         if(allocated(quick_xcg_tmp%sswt).or.allocated(quick_xcg_tmp%init_grid_ptx)) call abort_test('temporary grid release')
         write(localout,'(*(g0,:,","))') job,rank,grid_count,packed_count,quick_xcg_tmp%rad_gps,quick_xcg_tmp%ang_gps,.true.
         allocate(density(nbasis,nbasis),fock(nbasis,nbasis))
         density=quick_qm_struct%dense;fock=quick_qm_struct%o;api_before=quick_api%step
         call getQuickOEPROP(1,points,ierr,efield=efield,efg=efg)
         if(ierr/=0) call abort_test('retained-density properties')
         density_error=maxval(abs(quick_qm_struct%dense-density));fock_error=maxval(abs(quick_qm_struct%o-fock))
         if(density_error/=0.d0.or.fock_error/=0.d0.or.quick_api%step/=api_before) call abort_test('property state mutation')
         if(is_master) then
            if(.not.all(ieee_is_finite(efield)).or..not.all(ieee_is_finite(efg))) call abort_test('finite E/G')
            write(out,'(*(g0,:,","))') job,trim(labels(job)),custom,quick_method%xc_radial,quick_method%xc_angular, &
               grid_count,packed_count,quick_api%step,quick_api%step-api_before,energy,gradient,efield,efg,density_error,fock_error
         endif
         deallocate(density,fock)
         cleanup=0;call deleteQuickJob(cleanup)
         if(cleanup/=0) call abort_test('deleteQuickJob')
      enddo
      if(is_master) close(out)
      close(localout)
      if(is_master) write(*,'(A,I0,A,I0)') 'XC_LIFECYCLE_PASS jobs=',njobs,' ranks=',nranks
   else
      call abort_test('unknown mode')
   endif
#ifdef MPIV
   call MPI_FINALIZE(cleanup)
#endif
contains
   subroutine validate_method()
      if(.not.quick_method%DFT.or..not.quick_method%grad.or..not.quick_method%analGrad) call abort_test('active DFT analytic gradient')
      if(quick_method%xc_custom_grid.neqv.custom) call abort_test('custom grid reset')
      if(quick_method%xc_radial/=radial(job).or.quick_method%xc_angular/=angular(job)) call abort_test('method sizes reset')
      if(job>=7) then
         if(quick_method%iSG/=0) call abort_test('SG0 request')
      else
         if(quick_method%iSG/=1) call abort_test('SG1 request')
      endif
   end subroutine validate_method
   subroutine abort_test(label)
      character(len=*),intent(in) :: label
      write(*,'(A,I0,2A)') 'XC_LIFECYCLE_ERROR rank=',rank,' ',label
#ifdef MPIV
      call MPI_ABORT(quick_comm,57,cleanup)
#endif
      stop 2
   end subroutine abort_test
end program quick_xc_grid_lifecycle_driver
