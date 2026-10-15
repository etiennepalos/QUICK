#include "util.fh"
! (C) Copyright 2026 QUICK contributors. All rights reserved.
! This Source Code Form is subject to the terms of the Mozilla Public
! License, v. 2.0. If a copy of the MPL was not distributed with this
! file, You can obtain one at http://mozilla.org/MPL/2.0/.
module quick_adaptive_electrostatics_module
   use, intrinsic :: iso_fortran_env, only: int64
   use, intrinsic :: ieee_arithmetic, only: ieee_is_finite,ieee_value,ieee_quiet_nan
   use quick_electrostatic_jet_module, only: electronic_jet_cpu,electronic_cell_majorants_cpu, &
      electrostatic_jet_status,jet_rank_norm
   use quick_partition_of_unity_module, only: evaluate_electronic_cover,evaluate_electronic_cover_gpu
   implicit none
   private
   public :: adaptive_electrostatics_options,adaptive_electrostatics_report,evaluate_adaptive_electronic
   public :: ADAPTIVE_INDICATOR,ADAPTIVE_ANALYTIC_MAJORANT,ADAPTIVE_CERTIFIED
   integer, parameter :: ADAPTIVE_INDICATOR=1,ADAPTIVE_ANALYTIC_MAJORANT=2,ADAPTIVE_CERTIFIED=3

   type adaptive_electrostatics_options
      integer :: degree=3,error_mode=ADAPTIVE_ANALYTIC_MAJORANT
      double precision :: tolerances(3)=[1.0d-6,1.0d-5,1.0d-4]
      integer :: max_depth=3,max_anchors=600,max_overlap=8
      double precision :: max_prepare_seconds=30.0d0,max_memory_mb=256.0d0
      ! An explicit fixed domain is necessary for comparisons across calls.
      ! It does not freeze query-dependent refinement selection: derivative
      ! tests must also retain the same selected cover/anchors and density,
      ! or exercise evaluate_electronic_cover directly on a frozen cover.
      ! A batch outside a valid fixed box selects whole-batch direct fallback.
      logical :: fixed_domain=.false.
      double precision :: lower(3)=0.0d0,upper(3)=0.0d0
      ! Indicator mode replaces support boxes by their centers and scales
      ! the resulting point majorants. It has NO continuous bound claim.
      double precision :: indicator_factor=2.0d0
      ! Explicit CUDA-query selection; anchors and majorants remain CPU.
      ! Unsupported builds return a backend error, without a CPU fallback.
      logical :: gpu_queries=.false.
   end type adaptive_electrostatics_options

   type adaptive_electrostatics_report
      logical :: fallback=.true.,approximation_accepted=.false.,certified=.false.
      logical :: numerical_budget_known=.false.,continuous_truncation_formula=.false.
      logical :: underflow_detected=.false.,density_refreshed=.true.
      integer :: depth=0,anchors=0,bound_count=0,max_overlap=0,error_mode=0
      integer(int64) :: prefactor_underflow_count=0,boys_underflow_count=0
      double precision :: preparation_seconds=0.0d0,query_seconds=0.0d0
      double precision :: estimated_work_bytes=0.0d0
      double precision :: requested_tolerances(3)=0.0d0,truncation_budget(3)=0.0d0
      double precision :: total_error_budget(3)=0.0d0
      character(len=64) :: reason='not prepared'
      character(len=16) :: anchor_backend='cpu',query_backend='cpu'
   end type adaptive_electrostatics_report
contains
   subroutine evaluate_adaptive_electronic(nb,na,maxprim,nc,ang,centers,expo,coeff,atoms,density, &
         npoints,points,options,v,e,g,report,ierr)
      !---------------------------------------------------------------------!
      ! Resource-bounded, opt-in reference. Each call rebuilds all density-!
      ! dependent values: there is deliberately no reusable electronic   !
      ! cache to become stale across SCFs/frames. A uniform octree level   !
      ! gives a balanced fixed cover and at most eight overlapping patches.!
      ! Acceptance concerns the globally blended Taylor truncation budget.!
      ! Anchor, Boys, reduction and ordinary-FP errors remain UNKNOWN;    !
      ! the total error budget is NaN, never a fabricated zero. Certified !
      ! requests fail. Caps/coverage failures request WHOLE-BATCH direct   !
      ! fallback; outputs stay unchanged until an approximation succeeds. !
      ! This raw routine returns electronic V/E/G only. Public API adds   !
      ! classical charges directly or executes the existing direct route. !
      !---------------------------------------------------------------------!
      implicit none
      integer, intent(in) :: nb,na,maxprim,npoints,nc(nb),ang(3,nb),centers(nb)
      double precision, intent(in) :: expo(maxprim,nb),coeff(maxprim,nb),atoms(3,na),density(nb,nb)
      double precision, intent(in) :: points(:,:)
      type(adaptive_electrostatics_options), intent(in) :: options
      double precision, intent(inout) :: v(:),e(:,:),g(:,:,:)
      type(adaptive_electrostatics_report), intent(out) :: report
      integer, intent(out) :: ierr
      double precision, allocatable :: anchors(:,:),va(:),ea(:,:),ga(:,:,:),ha(:,:,:,:)
      double precision, allocatable :: majorants(:,:),vwork(:),ework(:,:),gwork(:,:,:),budgets(:,:)
      integer, allocatable :: owned(:)
      double precision :: lower(3),upper(3),width(3),middle(3),lo(3,8),hi(3,8),bounds(35,8)
      double precision :: estimated_bytes,elapsed
      integer :: depth,nside,nanchors,ix,iy,iz,index,start,last,count,status,overlap
      integer(int64) :: start_tick,query_tick,stop_tick,rate
      type(electrostatic_jet_status) :: jet_status

      report=adaptive_electrostatics_report()
      report%total_error_budget=ieee_value(0.0d0,ieee_quiet_nan)
      report%truncation_budget=ieee_value(0.0d0,ieee_quiet_nan)
      report%requested_tolerances=options%tolerances
      report%error_mode=options%error_mode
      report%continuous_truncation_formula=options%error_mode==ADAPTIVE_ANALYTIC_MAJORANT
      ierr=58
      if (options%error_mode==ADAPTIVE_CERTIFIED) then
         report%reason='certified arithmetic and anchor budgets unavailable'
         ierr=59
         return
      endif
      if (options%error_mode/=ADAPTIVE_INDICATOR.and.options%error_mode/=ADAPTIVE_ANALYTIC_MAJORANT) return
      if (options%degree/=2.and.options%degree/=3) return
      if (.not.all(ieee_is_finite(options%tolerances)).or.any(options%tolerances<=0)) return
      if (options%max_depth<0.or.options%max_depth>6.or.options%max_anchors<1) return
      if (options%max_overlap<1.or.options%max_overlap>8) return
      if (.not.ieee_is_finite(options%max_prepare_seconds).or.options%max_prepare_seconds<=0) return
      if (.not.ieee_is_finite(options%max_memory_mb).or.options%max_memory_mb<=0) return
      if (.not.ieee_is_finite(options%indicator_factor).or.options%indicator_factor<1) return
#if !defined(CUDA) && !defined(CUDA_MPIV)
      if (options%gpu_queries) then
         report%reason='CUDA partition queries unavailable in this build'
         ierr=56;return
      endif
#endif
      ierr=46
      if (npoints<1.or.npoints>huge(npoints)/40.or.nb<1.or.na<1.or.maxprim<1) return
      if (any(shape(points)/=[3,npoints]).or.size(v)/=npoints) return
      if (any(shape(e)/=[3,npoints]).or.any(shape(g)/=[3,3,npoints])) return
      ierr=55
      if (.not.all(ieee_is_finite(points))) return
      if (options%fixed_domain) then
         lower=options%lower;upper=options%upper
         if (.not.all(ieee_is_finite(lower)).or..not.all(ieee_is_finite(upper))) return
         if (any(upper<=lower)) return
         if (any(minval(points,dim=2)<lower).or.any(maxval(points,dim=2)>upper)) then
            report%reason='queries outside fixed domain';ierr=0;return
         endif
      else
         lower=minval(points,dim=2);upper=maxval(points,dim=2)
         middle=0.5d0*lower+0.5d0*upper
         ! A nonzero domain also covers one-point and planar workloads.
         width=max(upper-lower,0.5d0)
         lower=middle-0.5d0*width;upper=middle+0.5d0*width
      endif
      call system_clock(start_tick,rate)
      ierr=56
      if (rate<=0) return
      ! Bound work counts every attempted level. Time caps are checked after
      ! each batch of <=16 anchors / <=8 cells, never in an integral kernel.
      estimated_bytes=adaptive_work_bytes(npoints,1,nb,options%gpu_queries)
      report%estimated_work_bytes=estimated_bytes
      if (estimated_bytes>options%max_memory_mb*1024.0d0**2) then
         report%reason='work memory cap';ierr=0;return
      endif
      allocate(owned(nb),vwork(npoints),ework(3,npoints),gwork(3,3,npoints),budgets(3,npoints),stat=status)
      if (status/=0) return
      owned=1
      ierr=0
      do depth=0,options%max_depth
         nside=2**depth;nanchors=nside**3
         if (nanchors>options%max_anchors-report%anchors) then
            report%reason='anchor cap';exit
         endif
         estimated_bytes=adaptive_work_bytes(npoints,nanchors,nb,options%gpu_queries)
         report%estimated_work_bytes=max(report%estimated_work_bytes,estimated_bytes)
         if (estimated_bytes>options%max_memory_mb*1024.0d0**2) then
            report%reason='work memory cap';exit
         endif
         allocate(anchors(3,nanchors),va(nanchors),ea(3,nanchors),ga(3,3,nanchors), &
            ha(3,3,3,nanchors),majorants(2,nanchors),stat=status)
         if (status/=0) then
            ierr=56;return
         endif
         width=(upper-lower)/nside
         index=0
         do ix=0,nside-1
            do iy=0,nside-1
               do iz=0,nside-1
                  index=index+1
                  anchors(:,index)=lower+([real(ix,8),real(iy,8),real(iz,8)]+0.5d0)*width
               enddo
            enddo
         enddo
         va=0;ea=0;ga=0;ha=0;majorants=0
         do start=1,nanchors,16
            last=min(start+15,nanchors);count=last-start+1
            call electronic_jet_cpu(nb,na,maxprim,nc,ang,centers,expo,coeff,atoms,density, &
               count,anchors(:,start:last),owned,va(start:last),ea(:,start:last), &
               ga(:,:,start:last),ha(:,:,:,start:last),jet_status,ierr)
            call accumulate_status(report,jet_status)
            if (ierr/=0) return
            report%anchors=report%anchors+count
            if (time_exceeded()) exit
         enddo
         if (time_exceeded()) exit
         do start=1,nanchors,8
            last=min(start+7,nanchors);count=last-start+1
            if (options%error_mode==ADAPTIVE_INDICATOR) then
               lo(:,1:count)=anchors(:,start:last);hi(:,1:count)=lo(:,1:count)
            else
               do index=1,count
                  lo(:,index)=anchors(:,start+index-1)-width
                  hi(:,index)=anchors(:,start+index-1)+width
               enddo
            endif
            bounds=0
            call electronic_cell_majorants_cpu(nb,na,maxprim,nc,ang,centers,expo,coeff,atoms,density, &
               count,lo(:,1:count),hi(:,1:count),owned,bounds(:,1:count),jet_status,ierr)
            call accumulate_status(report,jet_status)
            if (ierr/=0) return
            do index=1,count
               majorants(:,start+index-1)=[jet_rank_norm(bounds(:,index),3),jet_rank_norm(bounds(:,index),4)]
            enddo
            report%bound_count=report%bound_count+count
            if (time_exceeded()) exit
         enddo
         if (time_exceeded()) exit
         if (options%error_mode==ADAPTIVE_INDICATOR) majorants=options%indicator_factor*majorants
         call system_clock(query_tick)
         if (options%gpu_queries) then
            report%query_backend='cuda'
            call evaluate_electronic_cover_gpu(lower,upper,nside,options%degree,va,ea,ga,ha,majorants, &
               points,vwork,ework,gwork,budgets,overlap,ierr)
         else
            call evaluate_electronic_cover(lower,upper,nside,options%degree,va,ea,ga,ha,majorants, &
               points,vwork,ework,gwork,budgets,overlap,ierr)
         endif
         call system_clock(stop_tick)
         report%query_seconds=report%query_seconds+real(stop_tick-query_tick,8)/real(rate,8)
         if (ierr==56) return
         if (ierr/=0) then
            report%reason='cover failure';ierr=0;exit
         endif
         report%depth=depth;report%max_overlap=overlap
         report%truncation_budget=maxval(budgets,dim=2)
         if (overlap>options%max_overlap) then
            report%reason='overlap cap';exit
         endif
         if (all(report%truncation_budget<=options%tolerances)) then
            report%fallback=.false.;report%approximation_accepted=.true.
            report%reason='accepted truncation budget; numerical error unknown'
            v=vwork;e=ework;g=gwork
            exit
         endif
         report%reason='truncation budget not met within depth cap'
         deallocate(anchors,va,ea,ga,ha,majorants)
      enddo
      call system_clock(stop_tick)
      report%preparation_seconds=max(0.0d0,real(stop_tick-start_tick,8)/real(rate,8)-report%query_seconds)
   contains
      logical function time_exceeded() result(exceeded)
         implicit none
         integer(int64) :: now
         call system_clock(now)
         elapsed=real(now-start_tick,8)/real(rate,8)-report%query_seconds
         exceeded=elapsed>options%max_prepare_seconds
         if (exceeded) report%reason='preparation time cap (checked between batches)'
      end function time_exceeded
   end subroutine evaluate_adaptive_electronic

   double precision function adaptive_work_bytes(npoints,nanchors,nb,gpu_queries) result(bytes)
      implicit none
      integer, intent(in) :: npoints,nanchors,nb
      logical, intent(in) :: gpu_queries
      ! Conservative live working-set estimate, including the public API's
      ! full-D copy and transactional output arrays, controller/PoU buffers,
      ! anchors and batch integral scratch. The resident QUICK job, caller
      ! inputs/outputs and runtime libraries are not additional method work.
      bytes=8.0d0*(45.0d0*npoints+45.0d0*nanchors+real(nb,8)*nb+35*8+4*286*16)+4.0d0*nb
      ! CUDA uploads42 scalar anchor channels and3 coordinate channels per
      ! point; outputs16 channels, plus two device status integers. The
      ! controller never supplies optional numerical anchor-error arrays.
      if (gpu_queries) bytes=bytes+8.0d0*(19.0d0*npoints+42.0d0*nanchors)+8.0d0
   end function adaptive_work_bytes

   subroutine accumulate_status(report,status)
      implicit none
      type(adaptive_electrostatics_report), intent(inout) :: report
      type(electrostatic_jet_status), intent(in) :: status
      report%underflow_detected=report%underflow_detected.or.status%underflow_detected
      report%prefactor_underflow_count=report%prefactor_underflow_count+status%prefactor_underflow_count
      report%boys_underflow_count=report%boys_underflow_count+status%boys_underflow_count
   end subroutine accumulate_status
end module quick_adaptive_electrostatics_module
