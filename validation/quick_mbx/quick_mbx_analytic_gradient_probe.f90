#include "util.fh"
!---------------------------------------------------------------------!
! Analytic versus finite-difference QUICK-MBX gradient validation.    !
!_____________________________________________________________________!

program quick_mbx_analytic_gradient_probe
   use quick_api_module, only: setQuickJob, setQuickMBXWaterSystem, setQuickMBXWaterCrossResidual, &
      getQuickMBXEnergyGradients, getQuickMBXEnergyGradientsFD, &
      clearQuickMBXSystem, deleteQuickJob
#ifdef MPIV
   use mpi
   use quick_api_module, only: setQuickMPI
#endif
   use quick_constants_module, only: A_TO_BOHRS
   use quick_mbx_module, only: quick_mbx_get_energy_terms
   implicit none

   integer, parameter :: natom_qm=3, nsteps=3
   double precision, parameter :: fd_steps(nsteps)=(/1.0d-3,5.0d-4,1.0d-4/)
   double precision, parameter :: gradient_tolerance=1.0d-5
   double precision, parameter :: invariance_tolerance=1.0d-4

   integer :: ierr, argc, unit
   double precision :: mon_a(3,3), mon_b(3,3), mon_c(3,3), mbx_pair(3,6)
   character(len=256) :: csv_file, json_mbx
   character(len=32) :: case_filter, qm_method, residual_mode
#ifdef MPIV
   integer :: mpierror, mpirank, mpisize
#endif
   logical :: master

   ierr=0
   master=.true.
#ifdef MPIV
   call MPI_INIT(mpierror)
   call MPI_COMM_RANK(MPI_COMM_WORLD,mpirank,mpierror)
   call MPI_COMM_SIZE(MPI_COMM_WORLD,mpisize,mpierror)
   master=mpirank == 0
   call setQuickMPI(mpirank,mpisize,ierr)
   call stop_on_error(ierr,'setQuickMPI')
#endif
   csv_file='quick_mbx_analytic_gradient_probe.csv'
   json_mbx='mbx.json'
   case_filter='all'
   qm_method='hf'
   residual_mode='cross'
   argc=command_argument_count()
   if (argc >= 1) call get_command_argument(1,csv_file)
   if (argc >= 2) call get_command_argument(2,json_mbx)
   if (argc >= 3) call get_command_argument(3,case_filter)
   if (argc >= 4) call get_command_argument(4,qm_method)
   if (argc >= 5) call get_command_argument(5,residual_mode)
   if (trim(residual_mode) /= 'cross' .and. trim(residual_mode) /= 'raw') then
      if (master) write(*,'(A)') 'Residual mode must be cross or raw.'
      stop 1
   endif
   call setup_water_trimer(mon_a,mon_b,mon_c)

   unit=-1
   if (master) then
      open(newunit=unit,file=csv_file,status='replace',action='write',iostat=ierr)
      call stop_on_error(ierr,'open csv')
      write(unit,'(A)') 'case,qm_method,model,residual_mode,group,index,axis,analytic,mbx_base,induced_efg,'// &
         'cross_residual,'// &
         'fd_1e3,fd_5e4,fd_1e4,err_1e3,err_5e4,err_1e4'
   endif

   if (selected('A_QM_B_MBX')) then
      call run_case('A_QM_B_MBX','h2o',qm_method,mon_a,mon_b,unit,json_mbx,ierr)
      call stop_on_error(ierr,'A_QM_B_MBX')
   endif
   if (selected('B_QM_A_MBX')) then
      call run_case('B_QM_A_MBX','h2o',qm_method,mon_b,mon_a,unit,json_mbx,ierr)
      call stop_on_error(ierr,'B_QM_A_MBX')
   endif
   mbx_pair(:,1:3)=mon_b
   mbx_pair(:,4:6)=mon_c
   if (selected('A_QM_BC_MBX')) then
      call run_case('A_QM_BC_MBX','h2o',qm_method,mon_a,mbx_pair,unit,json_mbx,ierr)
      call stop_on_error(ierr,'A_QM_BC_MBX')
   endif
   if (selected('A_QM_B_MBPBE')) then
      call run_case('A_QM_B_MBPBE','mbpbe',qm_method,mon_a,mon_b,unit,json_mbx,ierr)
      call stop_on_error(ierr,'A_QM_B_MBPBE')
   endif
   if (master) close(unit)
#ifdef MPIV
   call MPI_FINALIZE(mpierror)
#endif

contains

   logical function selected(label)
      implicit none
      character(len=*), intent(in) :: label
      selected=trim(case_filter) == 'all' .or. trim(case_filter) == trim(label)
   end function selected

   subroutine setup_water_trimer(mon_a,mon_b,mon_c)
      implicit none
      double precision, intent(out) :: mon_a(3,3), mon_b(3,3), mon_c(3,3)

      mon_a(:,1)=(/-1.58972425d0, 1.04337922d0,-0.08780840d0/)
      mon_a(:,2)=(/-0.63591971d0, 0.97898520d0, 0.00000000d0/)
      mon_a(:,3)=(/-1.90066280d0, 1.74501050d0,-0.66454990d0/)
      mon_b(:,1)=(/ 1.64924507d0, 1.08594656d0, 0.00000000d0/)
      mon_b(:,2)=(/ 2.60878026d0, 1.09587704d0,-0.02817115d0/)
      mon_b(:,3)=(/ 1.33830653d0, 1.78757784d0, 0.57674150d0/)
      mon_c(:,1)=(/-0.61315209d0, 2.46976336d0, 2.07005086d0/)
      mon_c(:,2)=(/ 0.34684791d0, 2.46976336d0, 2.07005086d0/)
      mon_c(:,3)=(/-0.93360667d0, 3.37469919d0, 2.07005086d0/)
   end subroutine setup_water_trimer

   subroutine run_case(label,model,qm_method,qm_xyz,mbx_xyz,unit,json_mbx,ierr)
      implicit none
      character(len=*), intent(in) :: label, model, qm_method, json_mbx
      double precision, intent(in) :: qm_xyz(3,natom_qm), mbx_xyz(:,:)
      integer, intent(in) :: unit
      integer, intent(out) :: ierr

      integer :: atomic_numbers(natom_qm), istep, nwaters, nmbx
      logical :: validation_failed
      double precision :: energy_analytic, energy_fd(nsteps)
      double precision :: mbx_raw, mbx_external, cross_2b, cross_3b, cross_disp
      double precision :: qm_analytic(3,natom_qm), qm_cross(3,natom_qm), qm_fd(3,natom_qm,nsteps)
      double precision :: maxerr(nsteps), rmserr(nsteps), net_norm, torque_norm
      double precision, allocatable :: mbx_analytic(:,:), mbx_base(:,:), induced_efg(:,:), cross_residual(:,:)
      double precision, allocatable :: mbx_fd(:,:,:)
      character(len=80) :: job_name
      character(len=256) :: keywd

      ierr=0
      atomic_numbers=(/8,1,1/)
      nmbx=size(mbx_xyz,2)
      if (size(mbx_xyz,1) < 3 .or. mod(nmbx,3) /= 0) then
         ierr=45
         return
      endif
      nwaters=nmbx/3
      allocate(mbx_analytic(3,nmbx),mbx_base(3,nmbx),induced_efg(3,nmbx),cross_residual(3,nmbx), &
         mbx_fd(3,nmbx,nsteps))

      job_name='quick_mbx_analytic_'//trim(label)//'_'//trim(qm_method)
      call make_keywords(qm_method,.true.,keywd,ierr)
      if (ierr /= 0) goto 900
      call setQuickJob(job_name,keywd,natom_qm,atomic_numbers,.true.,ierr)
      if (ierr /= 0) goto 900
      call setQuickMBXWaterSystem(nwaters,mbx_xyz,json_mbx,ierr,model)
      if (ierr /= 0) goto 900
      if (trim(residual_mode) == 'cross') then
         call setQuickMBXWaterCrossResidual(1,qm_xyz,nwaters,mbx_xyz,json_mbx,ierr,model)
         if (ierr /= 0) goto 900
      endif
      call getQuickMBXEnergyGradients(qm_xyz,mbx_xyz,energy_analytic,qm_analytic,mbx_analytic,ierr, &
         mbx_base,induced_efg,cross_residual,qm_cross)
      if (ierr /= 0) goto 900
      call quick_mbx_get_energy_terms(mbx_raw,mbx_external,cross_2b_au=cross_2b, &
         cross_3b_au=cross_3b,cross_disp_au=cross_disp)
      call clearQuickMBXSystem(ierr)
      if (ierr /= 0) goto 900
      call deleteQuickJob(ierr)
      if (ierr /= 0) goto 900

      job_name='quick_mbx_fd_'//trim(label)//'_'//trim(qm_method)
      call make_keywords(qm_method,.false.,keywd,ierr)
      if (ierr /= 0) goto 900
      call setQuickJob(job_name,keywd,natom_qm,atomic_numbers,.true.,ierr)
      if (ierr /= 0) goto 900
      do istep=1,nsteps
         call getQuickMBXEnergyGradientsFD(qm_xyz,nwaters,mbx_xyz,json_mbx,fd_steps(istep), &
            energy_fd(istep),qm_fd(:,:,istep),mbx_fd(:,:,istep),ierr,model, &
            trim(residual_mode) == 'cross')
         if (ierr /= 0) goto 900
      enddo

      call compare_all(qm_xyz,mbx_xyz,qm_analytic,mbx_analytic,qm_fd,mbx_fd, &
         maxerr,rmserr,net_norm,torque_norm)
      if (master) then
         call write_case(unit,label,qm_method,model,qm_analytic,qm_cross,mbx_analytic,mbx_base,induced_efg, &
            cross_residual,qm_fd,mbx_fd)
         write(*,'(A,1X,A,1X,A,1X,A,1X,ES24.16)') 'QUICK_MBX_ANALYTIC_ENERGY',trim(label), &
            trim(qm_method),trim(model),energy_analytic
         write(*,'(A,1X,A,3(1X,ES24.16))') 'QUICK_MBX_CROSS_TERMS',trim(label), &
            cross_2b,cross_3b,cross_disp
         do istep=1,nsteps
            write(*,'(A,1X,A,1X,ES10.3,1X,ES12.4,1X,ES12.4)') &
               'QUICK_MBX_ANALYTIC_ERROR',trim(label),fd_steps(istep),maxerr(istep),rmserr(istep)
         enddo
         write(*,'(A,1X,A,1X,ES12.4)') 'QUICK_MBX_ANALYTIC_NET_GRAD',trim(label),net_norm
         write(*,'(A,1X,A,1X,ES12.4)') 'QUICK_MBX_ANALYTIC_NET_TORQUE',trim(label),torque_norm
         write(*,'(A,1X,A,1X,ES12.4)') 'QUICK_MBX_ANALYTIC_ENERGY_FD_DIFF',trim(label), &
            maxval(abs(energy_fd-energy_analytic))
      endif

      validation_failed=maxerr(nsteps) > gradient_tolerance .or. net_norm > invariance_tolerance .or. &
         torque_norm > invariance_tolerance

      call clearQuickMBXSystem(ierr)
      if (ierr == 0) call deleteQuickJob(ierr)
      if (validation_failed .and. ierr == 0) ierr=45
900   continue
      deallocate(mbx_analytic,mbx_base,induced_efg,cross_residual,mbx_fd)
   end subroutine run_case

   subroutine make_keywords(qm_method,gradient,keywd,ierr)
      implicit none
      character(len=*), intent(in) :: qm_method
      logical, intent(in) :: gradient
      character(len=*), intent(out) :: keywd
      integer, intent(out) :: ierr

      ierr=0
      select case(trim(qm_method))
      case('hf')
         keywd='HF BASIS=STO-3G CUTOFF=1.0D-12 DENSERMS=1.0D-10 MBX_QMMM'
      case('pbe0-d3')
         keywd='DFT PBE0 D3 BASIS=DEF2-SVP CUTOFF=1.0D-12 GRADCUTOFF=1.0D-12 '// &
            'XCCUTOFF=1.0D-10 BASISCUTOFF=1.0D-10 DENSERMS=1.0D-9 MBX_QMMM'
      case('pbe-d3')
         keywd='DFT PBE D3 BASIS=DEF2-SVP CUTOFF=1.0D-12 GRADCUTOFF=1.0D-12 '// &
            'XCCUTOFF=1.0D-10 BASISCUTOFF=1.0D-10 DENSERMS=1.0D-9 MBX_QMMM'
      case default
         write(*,'(A,1X,A)') 'Unknown QM method:',trim(qm_method)
         ierr=45
         return
      end select
      if (gradient) keywd=trim(keywd)//' GRADIENT'
   end subroutine make_keywords

   subroutine compare_all(qm_xyz,mbx_xyz,qm_a,mbx_a,qm_fd,mbx_fd,maxerr,rmserr,net_norm,torque_norm)
      implicit none
      double precision, intent(in) :: qm_xyz(3,natom_qm), mbx_xyz(:,:)
      double precision, intent(in) :: qm_a(3,natom_qm), mbx_a(:,:)
      double precision, intent(in) :: qm_fd(3,natom_qm,nsteps), mbx_fd(:,:,:)
      double precision, intent(out) :: maxerr(nsteps), rmserr(nsteps), net_norm, torque_norm
      integer :: istep, i, j, ncomp
      double precision :: diff, accum, net(3), torque(3)

      net=sum(qm_a,dim=2)+sum(mbx_a,dim=2)
      torque(:)=0.0d0
      do j=1,natom_qm
         torque=torque+cross_product(A_TO_BOHRS*qm_xyz(:,j),qm_a(:,j))
      enddo
      do j=1,size(mbx_a,2)
         torque=torque+cross_product(A_TO_BOHRS*mbx_xyz(:,j),mbx_a(:,j))
      enddo
      net_norm=dsqrt(dot_product(net,net))
      torque_norm=dsqrt(dot_product(torque,torque))

      do istep=1,nsteps
         maxerr(istep)=0.0d0
         accum=0.0d0
         ncomp=0
         do j=1,natom_qm
            do i=1,3
               diff=qm_a(i,j)-qm_fd(i,j,istep)
               maxerr(istep)=max(maxerr(istep),abs(diff))
               accum=accum+diff*diff
               ncomp=ncomp+1
            enddo
         enddo
         do j=1,size(mbx_a,2)
            do i=1,3
               diff=mbx_a(i,j)-mbx_fd(i,j,istep)
               maxerr(istep)=max(maxerr(istep),abs(diff))
               accum=accum+diff*diff
               ncomp=ncomp+1
            enddo
         enddo
         rmserr(istep)=dsqrt(accum/dble(ncomp))
      enddo
   end subroutine compare_all

   subroutine write_case(unit,label,qm_method,model,qm_a,qm_cross,mbx_a,mbx_base,induced_efg, &
         cross_residual,qm_fd,mbx_fd)
      implicit none
      integer, intent(in) :: unit
      character(len=*), intent(in) :: label, qm_method, model
      double precision, intent(in) :: qm_a(3,natom_qm), qm_cross(3,natom_qm)
      double precision, intent(in) :: mbx_a(:,:), mbx_base(:,:), induced_efg(:,:)
      double precision, intent(in) :: cross_residual(:,:)
      double precision, intent(in) :: qm_fd(3,natom_qm,nsteps), mbx_fd(:,:,:)
      integer :: i, j

      do j=1,natom_qm
         do i=1,3
            call write_row(unit,label,qm_method,model,'qm',j,i,qm_a(i,j),0.0d0,0.0d0, &
               qm_cross(i,j),qm_fd(i,j,:))
         enddo
      enddo
      do j=1,size(mbx_a,2)
         do i=1,3
            call write_row(unit,label,qm_method,model,'mbx',j,i,mbx_a(i,j),mbx_base(i,j), &
               induced_efg(i,j),cross_residual(i,j),mbx_fd(i,j,:))
         enddo
      enddo
   end subroutine write_case

   subroutine write_row(unit,label,qm_method,model,group,index,idir,analytic,mbx_base,induced_efg, &
         cross_residual,fd)
      implicit none
      integer, intent(in) :: unit, index, idir
      character(len=*), intent(in) :: label, qm_method, model, group
      double precision, intent(in) :: analytic, mbx_base, induced_efg, cross_residual, fd(nsteps)

      write(unit,'(A,",",A,",",A,",",A,",",A,",",I0,",",A,10(",",ES24.16))') trim(label),trim(qm_method), &
         trim(model),trim(residual_mode),trim(group),index,axis_name(idir),analytic,mbx_base,induced_efg, &
         cross_residual,fd(1),fd(2),fd(3), &
         abs(analytic-fd(1)),abs(analytic-fd(2)),abs(analytic-fd(3))
   end subroutine write_row

   pure function cross_product(a,b) result(c)
      implicit none
      double precision, intent(in) :: a(3), b(3)
      double precision :: c(3)

      c=(/a(2)*b(3)-a(3)*b(2),a(3)*b(1)-a(1)*b(3),a(1)*b(2)-a(2)*b(1)/)
   end function cross_product

   character(len=1) function axis_name(idir)
      implicit none
      integer, intent(in) :: idir
      character(len=3), parameter :: axes='xyz'
      axis_name=axes(idir:idir)
   end function axis_name

   subroutine stop_on_error(ierr_in,label)
      implicit none
      integer, intent(in) :: ierr_in
      character(len=*), intent(in) :: label
      if (ierr_in /= 0) then
         if (master) write(*,'(A,1X,A,1X,I0)') 'QUICK_MBX_ANALYTIC_ERROR',trim(label),ierr_in
#ifdef MPIV
         call MPI_ABORT(MPI_COMM_WORLD,ierr_in,mpierror)
#endif
         stop 1
      endif
   end subroutine stop_on_error
end program quick_mbx_analytic_gradient_probe
