!
!	read_job_and_atom.f90
!	new_quick
!
!	Created by Yipu Miao on 2/18/11.
!	Copyright 2011 University of Florida. All rights reserved.

#include "util.fh"

subroutine read_job_and_atom(ierr)
   
   ! this subroutine is to read job and atoms
   
   use quick_molspec_module
   use quick_method_module
   use quick_mpi_module
   use quick_files_module
   use quick_calculated_module
   use quick_ecp_module
   use quick_api_module
   use quick_exception_module

#ifdef CEW
   use quick_cew_module, only: quick_cew, print
#endif

   implicit none
   
   ! Instant variables
   integer i,j,k
   integer istrt
   character(len=200) :: keyWD
   character(len=20) :: tempstring
   integer, intent(inout) :: ierr

   if (master) then

      ! AG 03/05/2007
      itolecp=0

      ! Open input file
      call PrtAct(iOutFile,"Read Job And Atom")

      if(quick_api%apiMode .and. quick_api%hasKeywd) then 
        keyWD=quick_api%Keywd
      else
        SAFE_CALL(quick_open(infile,inFileName,'O','F','W',.true.,ierr))
        read (inFile,'(A200)') keyWD
      endif

      call upcase(keyWD,200)
      write(iOutFile,'("  KEYWORD=",a)') keyWD

      ! These interfaces,"read","check" and "print" are from quick_method_module
      SAFE_CALL(read(quick_method,keyWD,ierr))     ! read method from Keyword
      call read(quick_qm_struct,keyWD)  ! read qm struct from Keyword
      call check(quick_method,iOutFile,ierr) ! check the correctness
      call print(quick_method,iOutFile,ierr) ! then print method
    
      ! read basis file
      SAFE_CALL(read_basis_file(keywd,ierr))
      call print_basis_file(iOutFile)
      if (quick_method%ecp) call print_ecp_file(iOutFile)
      
      ! If PDB flag is on, then call readPDB to read PDB file and 
      ! rewrite input file so that there will be no difference between
      ! PDB input and normal quick input file

      if (quick_method%PDB) call readPDB(infile)

      ! read molecule specfication. Note this is step 1 for molspec reading
      ! and this is mainly to read atom number, atom kind and external charge number. 
      call read(quick_molspec,inFile, isTemplate, quick_api%hasKeywd, quick_api%Keywd, ierr)
      !SAFE_CALL(check(quick_molspec, ierr))

      if( .not. (quick_api%apiMode .and. quick_api%hasKeywd )) close(inFile)

      ! ECP integrals prescreening  -Alessandro GENONI 03/05/2007
      if ((index(keywd,'TOL_ECPINT=') /= 0).and.quick_method%ecp) then
         istrt = index(keywd,'TOL_ECPINT=')+10
         call rdinum(keywd,istrt,itolecp,ierr)
         tolecp=2.30258d+00*itolecp
         thrshecp=10.0d0**(-1.0d0*itolecp)
         write(iOutFile,'("THRESHOLD FOR ECP-INTEGRALS PRESCREENING = ", &
               & E15.5)')thrshecp
      end if


      ! Alessandro GENONI 03/05/2007
      if ((itolecp == 0).and.quick_method%ecp) then
         itolecp=12
         tolecp=2.30258d+00*itolecp
         thrshecp=10.0d0**(-1.0d0*itolecp)
         write(iOutFile,'("THRESHOLD FOR ECP-INTEGRALS PRESCREENING = ", &
               & E15.5,"  (DEFAULT)")')thrshecp
      end if

#ifdef CEW
      SAFE_CALL(print(quick_cew, iOutFile, ierr))
#endif

      call PrtAct(iOutFile,"Finish reading job")

      close(infile)
   endif


#ifdef MPIV
   !-------------------MPI/ALL NODES---------------------------------------
   if (bMPI) then 
     call mpi_setup_job(ierr) ! communicates nodes and pass job specs to all nodes
     CHECK_ERROR(ierr)
   endif
   !-------------------MPI/ALL NODES---------------------------------------
#endif

end
