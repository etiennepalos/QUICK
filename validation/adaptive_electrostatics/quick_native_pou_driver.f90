! (C) Copyright 2026 QUICK contributors. Mozilla Public License, v. 2.0.
! Standalone acceptance client for the production partition-of-unity module.
! Input tensors and output records use Fortran component order.
program quick_native_pou_driver
   use quick_partition_of_unity_module, only: evaluate_electronic_cover
   implicit none
   character(len=512) :: input_path,output_path
   integer :: nside,degree,na,np,has_errors,mode,unit,ierr,overlap,point
   double precision :: lower(3),upper(3)
   double precision,allocatable :: va(:),ea(:,:),ga(:,:,:),ha(:,:,:,:),majorants(:,:),errors(:,:),points(:,:)
   double precision,allocatable :: v(:),e(:,:),g(:,:,:),budgets(:,:)
   logical :: preserved
   if(command_argument_count()/=2) stop 2
   call get_command_argument(1,input_path);call get_command_argument(2,output_path)
   open(newunit=unit,file=trim(input_path),status='old',action='read',iostat=ierr)
   if(ierr/=0) stop 2
   read(unit,*,iostat=ierr) nside,degree,na,np,has_errors,mode
   if(ierr/=0.or.na<0.or.np<0) stop 2
   allocate(va(na),ea(3,na),ga(3,3,na),ha(3,3,3,na),majorants(2,na),errors(4,na),points(3,np), &
      v(np),e(3,np),g(3,3,np),budgets(3,np))
   read(unit,*) lower;read(unit,*) upper
   if(na>0) then
      read(unit,*) va;read(unit,*) ea;read(unit,*) ga;read(unit,*) ha;read(unit,*) majorants
      if(has_errors/=0) read(unit,*) errors
   endif
   if(np>0) read(unit,*) points
   close(unit)
   v=13.25d0;e=-24.5d0;g=37.75d0;budgets=-48.125d0
   overlap=-1;ierr=-1
   if(has_errors/=0) then
      if(mode==3) then
         call evaluate_electronic_cover(lower,upper,nside,degree,va,ea,ga,ha,majorants,points, &
            v,e,g,budgets,overlap,ierr,errors(1:3,:))
      else
         call evaluate_electronic_cover(lower,upper,nside,degree,va,ea,ga,ha,majorants,points, &
            v,e,g,budgets,overlap,ierr,errors)
      endif
   else
      select case(mode)
      case(1)
         call evaluate_electronic_cover(lower,upper,nside,degree,va,ea,ga,ha,majorants,points(1:2,:), &
            v,e,g,budgets,overlap,ierr)
      case(2)
         call evaluate_electronic_cover(lower,upper,nside,degree,va,ea,ga(:,1:2,:),ha,majorants,points, &
            v,e,g,budgets,overlap,ierr)
      case(4)
         call evaluate_electronic_cover(lower,upper,nside,degree,va,ea(1:2,:),ga,ha,majorants,points, &
            v,e,g,budgets,overlap,ierr)
      case(5)
         call evaluate_electronic_cover(lower,upper,nside,degree,va,ea,ga,ha,majorants,points, &
            v,e,g(1:2,:,:),budgets,overlap,ierr)
      case(6)
         call evaluate_electronic_cover(lower,upper,nside,degree,va,ea,ga,ha,majorants,points, &
            v,e,g,budgets(1:2,:),overlap,ierr)
      case default
         call evaluate_electronic_cover(lower,upper,nside,degree,va,ea,ga,ha,majorants,points, &
            v,e,g,budgets,overlap,ierr)
      end select
   endif
   preserved=all(v==13.25d0).and.all(e==-24.5d0).and.all(g==37.75d0).and.all(budgets==-48.125d0)
   open(newunit=unit,file=trim(output_path),status='new',action='write',iostat=point)
   if(point/=0) stop 2
   write(unit,'(2(I0,1X))') ierr,overlap
   write(unit,'(I0,1X,L1)') np,preserved
   do point=1,np
      write(unit,'(I0,16(1X,ES26.17E3))') point,v(point),e(:,point),g(:,:,point),budgets(:,point)
   enddo
   close(unit)
end program quick_native_pou_driver
