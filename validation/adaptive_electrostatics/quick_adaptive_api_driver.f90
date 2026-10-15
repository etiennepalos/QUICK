#include "util.fh"
! Test-only linked adaptive API client. QM/MM Angstrom; probes bohr.
program quick_adaptive_api_driver
   use quick_api_module, only: setQuickJob,getQuickEnergy,getQuickOEPROP,getQuickFieldHessian, &
      getQuickAdaptiveElectrostatics,deleteQuickJob,quick_api,adaptive_electrostatics_options, &
      adaptive_electrostatics_report
   use quick_method_module, only: quick_method
   use quick_basis_module, only: nbasis,quick_basis,ncontract,itype,aexp,dcoeff
   use quick_calculated_module, only: quick_qm_struct
   use quick_molspec_module, only: xyz,quick_molspec
   use, intrinsic :: ieee_arithmetic, only: ieee_is_finite,ieee_value,ieee_quiet_nan
#ifdef MPIV
   use mpi
   use quick_api_module, only: setQuickMPI
   use quick_mpi_module, only: quick_comm,quick_comm_rank,quick_mpi_error
#endif
   implicit none
   character(len=2048) :: input_file,outdir,path
   character(len=256) :: keyword
   character(len=80) :: jobname
   character(len=32) :: request_name
   type(adaptive_electrostatics_options) :: options
   type(adaptive_electrostatics_report) :: report
   integer :: ierr,unit,atom,charge,point,na,nq,np,request,before_step,cleanup,status_before,status_after
   integer :: bad_status,shape_np,mu,primitive,color=0
#ifdef ADAPTIVE_GPU_QUERIES
   integer,parameter :: NREQUESTS=16
#else
   integer,parameter :: NREQUESTS=15
#endif
   integer, allocatable :: numbers(:)
   double precision, allocatable :: coords(:,:),charges(:,:),points(:,:),probe(:,:)
   double precision, allocatable :: v(:),e(:,:),g(:,:,:),vd(:),ed(:,:),gd(:,:,:),ha(:,:,:,:),hb(:,:,:,:)
   double precision, allocatable :: saved_d(:,:),saved_b(:,:),saved_o(:,:),state_d(:,:),state_b(:,:)
   double precision :: energy,density_error,fock_error,output_error,step_error,density_refresh_change
   logical :: is_master=.true.,converged
#ifdef MPIV
   integer :: world_rank,world_size,job_comm,comm_error
   call MPI_INIT(comm_error)
   call MPI_COMM_RANK(MPI_COMM_WORLD,world_rank,comm_error)
   call MPI_COMM_SIZE(MPI_COMM_WORLD,world_size,comm_error)
   job_comm=MPI_COMM_WORLD
   if (command_argument_count()==4) then
      if (world_size/=4) call MPI_ABORT(MPI_COMM_WORLD,1,comm_error)
      color=world_rank/2
      call MPI_COMM_SPLIT(MPI_COMM_WORLD,color,world_rank,job_comm,comm_error)
   endif
   call setQuickMPI(job_comm,ierr)
   if (ierr/=0) call MPI_ABORT(MPI_COMM_WORLD,ierr,comm_error)
   is_master=quick_comm_rank==0
#endif
   call get_command_argument(1+2*color,input_file)
   call get_command_argument(2+2*color,outdir)
   open(newunit=unit,file=trim(input_file),status='old',action='read',iostat=ierr)
   call fail(ierr,'open input')
   read(unit,'(a)') keyword
   read(unit,*) na,nq,np
   allocate(numbers(na),coords(3,na),charges(4,nq),points(3,np),probe(3,np), &
      v(np),e(3,np),g(3,3,np),vd(np),ed(3,np),gd(3,3,np),ha(3,3,3,np),hb(3,3,3,np))
   do atom=1,na
      read(unit,*) numbers(atom),coords(:,atom)
   enddo
   do charge=1,nq
      read(unit,*) charges(:,charge)
   enddo
   do point=1,np
      read(unit,*) points(:,point)
   enddo
   close(unit)
   write(jobname,'(a,i0)') 'adaptive_api_job',color
   call setQuickJob(jobname,keyword,na,numbers,.false.,ierr)
   call fail(ierr,'set job')
   v=0.125d0;e=-0.25d0;g=0.5d0
   call getQuickAdaptiveElectrostatics(np,points,options,v,e,g,report,status_before)
   call getQuickEnergy(coords,nq,charges,energy,ierr)
   call fail(ierr,'SCF')
   converged=merge(quick_method%uscf_conv,quick_method%scf_conv,quick_method%unrst)
   if (.not.converged.or..not.ieee_is_finite(energy)) call fail(55,'SCF convergence')
   allocate(saved_d(nbasis,nbasis),saved_b(nbasis,nbasis),saved_o(nbasis,nbasis), &
      state_d(nbasis,nbasis),state_b(nbasis,nbasis))
   saved_d=quick_qm_struct%dense;saved_b=0
   if (quick_method%unrst) saved_b=quick_qm_struct%denseb
   saved_o=quick_qm_struct%o
   call getQuickFieldHessian(np,points,ha,ierr)
   call fail(ierr,'initial H')
   before_step=quick_api%step
   density_error=0;fock_error=0;step_error=0;density_refresh_change=0
   do request=1,NREQUESTS
      options=adaptive_electrostatics_options()
      options%fixed_domain=.true.;options%lower=[4.75d0,-0.25d0,-0.25d0];options%upper=[5.25d0,0.25d0,0.25d0]
      options%tolerances=1.0d3
      options%max_prepare_seconds=10.0d0
      select case(request)
      case(1);request_name='degree2-majorant';options%degree=2
      case(2);request_name='degree3-majorant'
      case(3);request_name='degree2-indicator';options%degree=2;options%error_mode=1
      case(4);request_name='degree3-indicator';options%error_mode=1
      case(5);request_name='requested-tolerances';options%tolerances=[1d-6,1d-5,1d-4]
      case(6);request_name='anchor-fallback';options%max_anchors=1;options%tolerances=1d-20
      case(7);request_name='memory-fallback';options%max_memory_mb=1d-4
      case(8);request_name='time-fallback';options%max_prepare_seconds=1d-12
      case(9);request_name='overlap-fallback';options%max_overlap=1;options%max_depth=1;options%tolerances=1d-20
      case(10);request_name='certified-unavailable';options%error_mode=3
      case(11);request_name='bad-shape'
      case(12);request_name='nonfinite-one-rank'
      case(13);request_name='singular-probe'
      case(14);request_name='density-refresh-scaled'
         quick_qm_struct%dense=1.013d0*saved_d
         if (quick_method%unrst) quick_qm_struct%denseb=1.013d0*saved_b
      case(15);request_name='density-refresh-restored'
         quick_qm_struct%dense=saved_d
         if (quick_method%unrst) quick_qm_struct%denseb=saved_b
#ifdef ADAPTIVE_GPU_QUERIES
      case(16);request_name='GPU-query-unsupported-build';options%gpu_queries=.true.
#endif
      end select
      if (quick_method%unrst) quick_qm_struct%denseab=quick_qm_struct%dense+quick_qm_struct%denseb
      state_d=quick_qm_struct%dense;state_b=0
      if (quick_method%unrst) state_b=quick_qm_struct%denseb
      call getQuickOEPROP(np,points,ierr,esp=vd,efield=ed,efg=gd)
      call fail(ierr,'direct reference')
      probe=points;v=0.125d0;e=-0.25d0;g=0.5d0
      shape_np=np
      if (request==11.and.is_master) shape_np=np-1
      if (request==12.and.is_master) probe(1,1)=ieee_value(0.0d0,ieee_quiet_nan)
      if (request==13) probe(:,1)=xyz(:,1)
      call getQuickAdaptiveElectrostatics(np,probe,options,v(1:shape_np),e,g,report,bad_status)
      output_error=0
      if (bad_status/=0) output_error=max(maxval(abs(v-0.125d0)),maxval(abs(e+0.25d0)),maxval(abs(g-0.5d0)))
      density_error=max(density_error,maxval(abs(quick_qm_struct%dense-state_d)))
      if (quick_method%unrst) density_error=max(density_error,maxval(abs(quick_qm_struct%denseb-state_b)))
      fock_error=max(fock_error,maxval(abs(quick_qm_struct%o-saved_o)))
      step_error=max(step_error,real(abs(quick_api%step-before_step),8))
      if (request==14) density_refresh_change=maxval(abs(v-vd))
      if (is_master) then
         write(path,'(a,a,i2.2,a)') trim(outdir),'/request-',request,'.meta'
         open(newunit=unit,file=trim(path),status='new',action='write')
         write(unit,'(a,1x,a)') 'name',trim(request_name)
         write(unit,*) 'ierr',bad_status
         write(unit,*) 'flags',report%fallback,report%approximation_accepted,report%certified, &
            report%numerical_budget_known,report%continuous_truncation_formula,report%density_refreshed
         write(unit,*) 'counts',report%depth,report%anchors,report%bound_count,report%max_overlap
         write(unit,*) 'underflow',report%underflow_detected,report%prefactor_underflow_count,report%boys_underflow_count
         write(unit,'(a,3(1x,es26.17e3))') 'budget',report%truncation_budget
         write(unit,'(a,3(1x,es26.17e3))') 'total_budget',report%total_error_budget
         write(unit,'(a,3(1x,es26.17e3))') 'times_bytes',report%preparation_seconds,report%query_seconds,report%estimated_work_bytes
         write(unit,'(a,1x,es26.17e3)') 'output_error',output_error
         write(unit,'(a,1x,a)') 'reason',trim(report%reason)
         close(unit)
         write(path,'(a,a,i2.2,a)') trim(outdir),'/request-',request,'.dat'
         open(newunit=unit,file=trim(path),status='new',action='write')
         do point=1,np
            write(unit,'(29(es26.17e3,1x))') points(:,point),v(point),e(:,point),g(:,:,point), &
               vd(point),ed(:,point),gd(:,:,point)
         enddo
         close(unit)
      endif
   enddo
   call getQuickFieldHessian(np,points,hb,ierr)
   call fail(ierr,'unchanged direct H')
   if (is_master) then
      call export_state()
      path=trim(outdir)//'/result.dat'
      open(newunit=unit,file=trim(path),status='new',action='write')
      write(unit,'(a,1x,es26.17e3)') 'energy',energy
      write(unit,*) 'basis',nbasis
      write(unit,*) 'unrestricted',quick_method%unrst
      write(unit,*) 'converged',converged
      write(unit,*) 'pre_ready_status',status_before
      write(unit,'(a,4(1x,es26.17e3))') 'state_errors',density_error,fock_error,step_error,maxval(abs(hb-ha))
      close(unit)
   endif
   call deleteQuickJob(cleanup)
   call fail(cleanup,'delete')
   call getQuickAdaptiveElectrostatics(np,points,options,v,e,g,report,status_after)
   if (is_master) then
      open(newunit=unit,file=trim(outdir)//'/result.dat',status='old',position='append',action='write')
      write(unit,*) 'post_delete_status',status_after
      close(unit)
   endif
#ifdef MPIV
   if (command_argument_count()==4) call MPI_COMM_FREE(job_comm,comm_error)
   call MPI_FINALIZE(comm_error)
#endif
contains
   subroutine fail(status,label)
      implicit none
      integer,intent(in) :: status
      character(len=*),intent(in) :: label
      if (status==0) return
      write(*,*) 'ADAPTIVE_API_VALIDATION_ERROR',label,status
#ifdef MPIV
      call MPI_ABORT(quick_comm,max(1,abs(status)),quick_mpi_error)
#endif
      stop 1
   end subroutine fail

   subroutine export_state()
      implicit none
      integer :: out,mu,primitive
      open(newunit=out,file=trim(outdir)//'/ao_basis.dat',status='new',action='write')
      write(out,*) nbasis
      do mu=1,nbasis
         write(out,*) mu,quick_basis%ncenter(mu),itype(:,mu),ncontract(mu)
         do primitive=1,ncontract(mu)
            write(out,'(2(es26.17e3,1x))') aexp(primitive,mu),dcoeff(primitive,mu)
         enddo
      enddo
      close(out)
      open(newunit=out,file=trim(outdir)//'/density.dat',status='new',action='write')
      write(out,*) nbasis,quick_method%unrst
      write(out,'(es26.17e3)') quick_qm_struct%dense
      if (quick_method%unrst) write(out,'(es26.17e3)') quick_qm_struct%denseb
      close(out)
      open(newunit=out,file=trim(outdir)//'/geometry_bohr.dat',status='new',action='write')
      do mu=1,na
         write(out,'(3(es26.17e3,1x))') xyz(:,mu)
      enddo
      close(out)
      open(newunit=out,file=trim(outdir)//'/classical_sources.dat',status='new',action='write')
      do mu=1,na
         write(out,'(4(es26.17e3,1x))') xyz(:,mu),quick_molspec%chg(mu)
      enddo
      do mu=1,quick_molspec%nextatom
         write(out,'(4(es26.17e3,1x))') quick_molspec%extxyz(:,mu),quick_molspec%extchg(mu)
      enddo
      close(out)
   end subroutine export_state
end program quick_adaptive_api_driver
