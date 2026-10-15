#include "util.fh"
!
! (C) Copyright 2026 QUICK contributors. Mozilla Public License, v. 2.0.
! Strict named source conventions, boundary tokens and default reset.
!
program quick_ddx_source_parser_convergence_driver
   use quick_method_module, only: quick_method_type, init, read, check
   use quick_ddx_backend_module, only: quick_ddx_available
   implicit none
   type(quick_method_type) :: method
   integer :: ierr, unit, checks, failures
   checks=0
   failures=0
   open(newunit=unit,status='scratch',action='readwrite')
   call options('HF DDX=PCM',quick_ddx_available)
   call assert(method%ddx_source==1,'omitted source remains REGULAR')
   call options('HF DDX=PCM DDX_SOURCE=REGULAR',quick_ddx_available)
   call assert(method%ddx_source==1,'explicit REGULAR')
   call options('HF DDX=COSMO DDX_SOURCE=COULOMB',quick_ddx_available)
   call assert(method%ddx_source==2,'explicit COULOMB')
   call options('hf ddx=pcm ddx_source=coulomb',quick_ddx_available)
   call assert(method%ddx_source==2,'case normalization')
   call options('HF DDX_SOURCE = COULOMB DDX=PCM',quick_ddx_available)
   call assert(method%ddx_source==2,'source before model and spaced equals')
   call options('HF DDX=PCM DDX_SOURCE'//achar(9)//'='//achar(9)//'REGULAR',quick_ddx_available)
   call options('HF DDX=PCM NOT_DDX_SOURCE=COULOMB MY_DDX_SOURCE=COULOMB',quick_ddx_available)
   call assert(method%ddx_source==1,'prefix boundaries ignored')
   call options('HF DDX=PCM DDX_SOURCE_EXTRA=COULOMB',quick_ddx_available)
   call assert(method%ddx_source==1,'suffix boundary ignored')
   call options('HF DDX_SOURCE=COULOMB',.false.)
   call options('HF DDX_SOURCE=REGULAR',.false.)
   call options('HF DDX=PCM DDX_SOURCE',.false.)
   call options('HF DDX=PCM DDX_SOURCE=',.false.)
   call options('HF DDX=PCM DDX_SOURCE = ',.false.)
   call options('HF DDX=PCM DDX_SOURCE COULOMB',.false.)
   call options('HF DDX=PCM DDX_SOURCE=2',.false.)
   call options('HF DDX=PCM DDX_SOURCE=NaN',.false.)
   call options('HF DDX=PCM DDX_SOURCE=COULOMBX',.false.)
   call options('HF DDX=PCM DDX_SOURCE=REGULAR=COULOMB',.false.)
   call options('HF DDX=PCM DDX_SOURCE=REGULAR,COULOMB',.false.)
   call options('HF DDX=PCM DDX_SOURCE=COULOMB DDX_SOURCE=REGULAR',.false.)
   call options('HF DDX=PCM DDX_SOURCE=COULOMB DDX_SOURCE=COULOMB',.false.)
   call options('HF DDX=PCM DDX_SOURCE=REGULAR DDX_SOURCE=REGULAR',.false.)
   call options('HF',.true.)
   call assert(.not.method%ddx.and.method%ddx_source==1,'new-job ddX disabled and source default restored')
   method%ddx=.true.
   method%ddx_source=0
   ierr=0
   call check(method,unit,ierr)
   call assert(ierr/=0,'direct invalid source component rejected')
   close(unit)
   write(*,'(A,I0,A,I0)') 'source_parser_checks ',checks,' failures ',failures
   if(failures/=0) stop 1
contains
   subroutine options(card,success)
      implicit none
      character(len=*),intent(in)::card
      logical,intent(in)::success
      character(len=300)::keyword
      keyword=card
      ierr=0
      call init(method,ierr)
      call read(method,keyword,ierr)
      if(ierr==0) call check(method,unit,ierr)
      call assert((ierr==0).eqv.success,card)
   end subroutine options
   subroutine assert(passed,label)
      implicit none
      logical,intent(in)::passed
      character(len=*),intent(in)::label
      checks=checks+1
      if(.not.passed) failures=failures+1
      write(*,'(L1,1X,A)') passed,label
   end subroutine assert
end program quick_ddx_source_parser_convergence_driver
