#include "util.fh"
! (C) Copyright 2026 QUICK contributors. Mozilla Public License, v. 2.0.
! Serial fully linked matched V/E/G benchmark. Setup SCF is excluded from
! API timings; adaptive timings include preparation, bounds and fallback.
program quick_adaptive_benchmark
   use, intrinsic :: iso_fortran_env, only: int64
   use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
   use quick_api_module, only: setQuickJob,getQuickEnergy,getQuickOEPROP, &
      getQuickAdaptiveElectrostatics,deleteQuickJob,quick_api, &
      adaptive_electrostatics_options,adaptive_electrostatics_report
   use quick_method_module, only: quick_method
   use quick_calculated_module, only: quick_qm_struct
   use quick_basis_module, only: nbasis,quick_basis,itype,ncontract,aexp,dcoeff
#ifdef ADAPTIVE_GPU_QUERIES
   use quick_partition_of_unity_module, only: get_adaptive_gpu_info
#endif
   implicit none
   character(len=2048) :: input_path,density_path,option_path,argument
   character(len=256) :: keyword
   character(len=80) :: job_name
   character(len=32) :: phase
   type(adaptive_electrostatics_options) :: options
   type(adaptive_electrostatics_report) :: report
   integer :: ierr,cleanup,input,output,timings,natoms,ncharges,npoints,repeats
   integer :: atom,point,mu,primitive,iteration,route,api_before
   integer :: gpu_launches_before=0,gpu_launches_after=0
   integer,allocatable :: numbers(:)
   integer(int64) :: tick_rate,start_tick,stop_tick,job_start,setup_stop,job_stop
   integer(int64) :: device_live_bytes=0_int64,device_last_bytes=0_int64
   double precision :: energy,elapsed,density_error,fock_error,errors(3)
   double precision,allocatable :: coords(:,:),charges(:,:),points(:,:),v(:),e(:,:),g(:,:,:)
   double precision,allocatable :: vd(:),ed(:,:),gd(:,:,:),density(:,:),beta(:,:),fock(:,:)
   logical :: converged

   if (command_argument_count()/=4) call fail(46,'input,density,options,repeats')
   call get_command_argument(1,input_path)
   call get_command_argument(2,density_path)
   call get_command_argument(3,option_path)
   call get_command_argument(4,argument)
   read(argument,*,iostat=ierr) repeats
   call fail(ierr,'repeat')
   if (repeats<1.or.repeats>100) call fail(46,'repeat range')
   open(newunit=input,file=trim(option_path),status='old',action='read')
#ifdef ADAPTIVE_GPU_QUERIES
   read(input,*) options%degree,options%error_mode,options%max_depth,options%max_anchors,options%max_overlap,options%gpu_queries
#else
   read(input,*) options%degree,options%error_mode,options%max_depth,options%max_anchors,options%max_overlap
#endif
   read(input,*) options%tolerances,options%max_prepare_seconds,options%max_memory_mb,options%indicator_factor
   read(input,*) options%lower,options%upper
   close(input)
   options%fixed_domain=.true.
   open(newunit=input,file=trim(input_path),status='old',action='read')
   read(input,'(a)') keyword
   read(input,*) natoms,ncharges,npoints
   allocate(numbers(natoms),coords(3,natoms),charges(4,ncharges),points(3,npoints), &
      v(npoints),e(3,npoints),g(3,3,npoints),vd(npoints),ed(3,npoints),gd(3,3,npoints))
   do atom=1,natoms
      read(input,*) numbers(atom),coords(:,atom)
   enddo
   do point=1,ncharges
      read(input,*) charges(:,point)
   enddo
   do point=1,npoints
      read(input,*) points(:,point)
   enddo
   close(input)
   call system_clock(job_start,tick_rate)
   job_name='adaptive_benchmark'
   call setQuickJob(job_name,keyword,natoms,numbers,.false.,ierr)
   call fail(ierr,'set job')
   call getQuickEnergy(coords,ncharges,charges,energy,ierr)
   call fail(ierr,'SCF')
   converged=merge(quick_method%uscf_conv,quick_method%scf_conv,quick_method%unrst)
   if (.not.converged.or..not.ieee_is_finite(energy)) call fail(55,'SCF convergence')
   if (trim(density_path)/='-') call replace_density(trim(density_path))
   allocate(density(nbasis,nbasis),fock(nbasis,nbasis))
   density=quick_qm_struct%dense;fock=quick_qm_struct%o
   if (quick_method%unrst) then
      allocate(beta(nbasis,nbasis));beta=quick_qm_struct%denseb
   endif
   api_before=quick_api%step
   call system_clock(setup_stop)
   open(newunit=timings,file='timings.csv',status='new',action='write')
   write(timings,'(a)') 'route,iteration,wall_seconds,preparation_seconds,query_seconds,fallback,accepted,certified,numerical_budget_known,depth,anchors,bound_count,budget_V,budget_E,budget_G,total_budget_V,total_budget_E,total_budget_G,error_V,error_E,error_G,gpu_query_launches,device_live_bytes,device_last_bytes'
   do route=1,2
      if (route==1) then
         phase='direct'
      else
         phase='adaptive'
      endif
      do iteration=0,repeats
#ifdef ADAPTIVE_GPU_QUERIES
         call get_adaptive_gpu_info(gpu_launches_before,device_live_bytes,device_last_bytes)
#endif
         call system_clock(start_tick)
         if (route==1) then
            call getQuickOEPROP(npoints,points,ierr,esp=v,efield=e,efg=g)
         else
            call getQuickAdaptiveElectrostatics(npoints,points,options,v,e,g,report,ierr)
         endif
         call system_clock(stop_tick)
         call fail(ierr,'timed property API')
#ifdef ADAPTIVE_GPU_QUERIES
         call get_adaptive_gpu_info(gpu_launches_after,device_live_bytes,device_last_bytes)
#endif
         elapsed=real(stop_tick-start_tick,8)/real(tick_rate,8)
         errors=0
         if (route==1) then
            vd=v;ed=e;gd=g
            report=adaptive_electrostatics_report()
            report%query_seconds=elapsed
         else
            errors(1)=maxval(abs(v-vd))
            do point=1,npoints
               errors(2)=max(errors(2),sqrt(sum((e(:,point)-ed(:,point))**2)))
               errors(3)=max(errors(3),sqrt(sum((g(:,:,point)-gd(:,:,point))**2)))
            enddo
         endif
         write(timings,'(a,",",i0,3(",",es26.17e3),4(",",l1),3(",",i0),9(",",es26.17e3),3(",",i0))') &
            trim(phase),iteration,elapsed,report%preparation_seconds,report%query_seconds,report%fallback, &
            report%approximation_accepted,report%certified,report%numerical_budget_known,report%depth, &
            report%anchors,report%bound_count,report%truncation_budget,report%total_error_budget,errors, &
            gpu_launches_after-gpu_launches_before,device_live_bytes,device_last_bytes
      enddo
      flush(timings)
   enddo
   close(timings)
   if (.not.all(ieee_is_finite(v)).or..not.all(ieee_is_finite(e)).or..not.all(ieee_is_finite(g))) call fail(55,'finite output')
   open(newunit=output,file='properties.dat',status='new',action='write')
   do point=1,npoints
      write(output,'(29(es26.17e3,1x))') points(:,point),v(point),e(:,point),g(:,:,point),vd(point),ed(:,point),gd(:,:,point)
   enddo
   close(output)
   density_error=maxval(abs(quick_qm_struct%dense-density))
   if (quick_method%unrst) density_error=max(density_error,maxval(abs(quick_qm_struct%denseb-beta)))
   fock_error=maxval(abs(quick_qm_struct%o-fock))
   call export_density_basis()
   open(newunit=output,file='result.dat',status='new',action='write')
   write(output,*) 'basis',nbasis
   write(output,*) 'points',npoints
   write(output,*) 'converged',converged
   write(output,*) 'extra_API_calls',quick_api%step-api_before
   write(output,'(a,3(1x,es26.17e3))') 'state_errors',density_error,fock_error,real(quick_api%step-api_before,8)
   write(output,'(a,1x,es26.17e3)') 'setup_scf_seconds',real(setup_stop-job_start,8)/real(tick_rate,8)
   write(output,'(a,1x,a)') 'adaptive_reason',trim(report%reason)
   write(output,'(a,2(1x,a))') 'adaptive_backends',trim(report%anchor_backend),trim(report%query_backend)
   write(output,*) 'underflow',report%underflow_detected,report%prefactor_underflow_count,report%boys_underflow_count
   cleanup=0
   call deleteQuickJob(cleanup)
   call fail(cleanup,'delete job')
   call system_clock(job_stop)
   write(output,'(a,1x,es26.17e3)') 'setup_to_teardown_seconds',real(job_stop-job_start,8)/real(tick_rate,8)
   close(output)
contains
   subroutine fail(status,label)
      implicit none
      integer,intent(in) :: status
      character(len=*),intent(in) :: label
      if (status==0) return
      write(*,*) 'ADAPTIVE_BENCHMARK_ERROR',label,status
      stop 1
   end subroutine fail

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
end program quick_adaptive_benchmark
