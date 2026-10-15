! Standalone controller/POU client with a normalized fixed s-Gaussian density.
program native_controller_driver
   use quick_electrostatic_jet_module
   use quick_adaptive_electrostatics_module
   use quick_partition_of_unity_module
   use, intrinsic :: ieee_arithmetic, only: ieee_value,ieee_quiet_nan
   implicit none
   type(adaptive_electrostatics_options) :: options
   type(adaptive_electrostatics_report) :: report
   type(electrostatic_jet_status) :: status
   integer :: np,mode,method,uin,uout,ierr,nside,nanchors,index,ix,iy,iz,i,overlap
   integer :: nc(1)=[1],ang(3,1)=0,centers(1)=[1],owned(1)=[1]
   double precision :: expo(1,1)=0.7d0,coeff(1,1),atoms(3,1)=0.0d0,density(1,1)
   double precision :: width(3),amplitude,pi
   double precision, allocatable :: points(:,:),v(:),e(:,:),g(:,:,:),budgets(:,:)
   double precision, allocatable :: anchors(:,:),va(:),ea(:,:),ga(:,:,:),ha(:,:,:,:),lower(:,:),upper(:,:)
   double precision, allocatable :: bounds(:,:),majorants(:,:)
   character(len=2048) :: input_path,output_path

   call get_command_argument(1,input_path)
   call get_command_argument(2,output_path)
   open(newunit=uin,file=trim(input_path),status='old',action='read')
   read(uin,*) np,mode,method,amplitude
   read(uin,*) options%degree,options%error_mode,options%max_depth,options%max_anchors,options%max_overlap
   read(uin,*) options%max_prepare_seconds,options%max_memory_mb,options%tolerances,options%indicator_factor
   read(uin,*) options%lower,options%upper
   options%fixed_domain=.true.
   allocate(points(3,np),v(np),e(3,np),g(3,3,np),budgets(3,np))
   read(uin,*) points
   close(uin)
   pi=acos(-1.0d0)
   coeff=(1.4d0/pi)**0.75d0
   density=amplitude
   v=0.125d0;e=-0.25d0;g=0.5d0;budgets=0
   if (mode==2) points(1,1)=ieee_value(0.0d0,ieee_quiet_nan)
   if (method==0) then
      if (mode==1) then
         call evaluate_adaptive_electronic(1,1,1,nc,ang,centers,expo,coeff,atoms,density,np,points, &
            options,v(1:np-1),e,g,report,ierr)
      else
         call evaluate_adaptive_electronic(1,1,1,nc,ang,centers,expo,coeff,atoms,density,np,points, &
            options,v,e,g,report,ierr)
      endif
   else
      nside=2**options%max_depth;nanchors=nside**3
      allocate(anchors(3,nanchors),va(nanchors),ea(3,nanchors),ga(3,3,nanchors),ha(3,3,3,nanchors), &
         lower(3,nanchors),upper(3,nanchors),bounds(35,nanchors),majorants(2,nanchors))
      width=(options%upper-options%lower)/nside
      index=0
      do ix=0,nside-1
         do iy=0,nside-1
            do iz=0,nside-1
               index=index+1
               anchors(:,index)=options%lower+([real(ix,8),real(iy,8),real(iz,8)]+0.5d0)*width
               lower(:,index)=anchors(:,index)-width;upper(:,index)=anchors(:,index)+width
            enddo
         enddo
      enddo
      va=0;ea=0;ga=0;ha=0;bounds=0
      call electronic_jet_cpu(1,1,1,nc,ang,centers,expo,coeff,atoms,density,nanchors,anchors, &
         owned,va,ea,ga,ha,status,ierr)
      if (ierr/=0) stop 2
      call electronic_cell_majorants_cpu(1,1,1,nc,ang,centers,expo,coeff,atoms,density,nanchors, &
         lower,upper,owned,bounds,status,ierr)
      if (ierr/=0) stop 3
      do i=1,nanchors
         majorants(:,i)=[jet_rank_norm(bounds(:,i),3),jet_rank_norm(bounds(:,i),4)]
      enddo
      if (mode==3) va=va+2.3d0
      if (mode==4) ga(1,2,1)=ga(1,2,1)+0.1d0
      if (mode==5) ha(1,2,3,1)=ha(1,2,3,1)+0.1d0
      call evaluate_electronic_cover(options%lower,options%upper,nside,options%degree,va,ea,ga,ha, &
         majorants,points,v,e,g,budgets,overlap,ierr)
      report=adaptive_electrostatics_report()
      report%total_error_budget=ieee_value(0.0d0,ieee_quiet_nan)
      report%depth=options%max_depth;report%anchors=nanchors;report%max_overlap=overlap
      report%fallback=ierr/=0;report%approximation_accepted=ierr==0
      report%continuous_truncation_formula=.true.
      report%truncation_budget=maxval(budgets,dim=2)
      report%reason='explicit frozen cover test'
   endif
   open(newunit=uout,file=trim(output_path),status='replace',action='write')
   write(uout,*) ierr,report%fallback,report%approximation_accepted,report%certified,report%numerical_budget_known, &
      report%continuous_truncation_formula,report%underflow_detected
   write(uout,*) report%depth,report%anchors,report%bound_count,report%max_overlap
   write(uout,'(9(es26.17,1x))') report%requested_tolerances,report%truncation_budget,report%total_error_budget
   write(uout,'(3(es26.17,1x))') report%preparation_seconds,report%query_seconds,report%estimated_work_bytes
   write(uout,'(a)') trim(report%reason)
   do i=1,np
      write(uout,'(13(es26.17,1x))') v(i),e(:,i),g(:,:,i)
   enddo
   close(uout)
end program native_controller_driver
