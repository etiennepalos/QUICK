#include "util.fh"
!
!	getEnergy.f90
!	new_quick
!
!	Created by Yipu Miao on 3/4/11.
!	Copyright 2011 University of Florida. All rights reserved.
!
!   written by Ed Brothers. 08/15/02
!   This subroutine calculates and ouptus the energy.
!
subroutine getEnergy(isGuess, ierr)
   use allMod
   use quick_gridpoints_module
   use quick_scf_module
   use quick_uscf_module, only: uscf
   use quick_overlap_module, only: fullx
   use quick_dftd3_module, only: calculateDFTD3 
   use quick_exception_module
   use quick_mbx_module, only: quick_mbx_active, quick_mbx_refresh_energy, quick_mbx_get_energy_terms
#ifdef CEW
   use quick_cew_module, only : quick_cew
#endif
#ifdef MPIV
   use mpi
#endif

   implicit none

   double precision :: distance
   double precision :: field_dot_r
   double precision, external :: rootSquare
   integer i,j
   logical, intent(in) :: isGuess
   integer, intent(inout) :: ierr
   logical :: verbose
   double precision :: mbx_energy_au, mbx_external_energy_au, mbx_additive_energy_au
   double precision :: mbx_perm_total_au, mbx_perm_classical_au, mbx_perm_electronic_au
   double precision :: mbx_ind_total_au, mbx_ind_classical_au, mbx_ind_electronic_au
   double precision :: mbx_external_half_residual_au

   verbose = .true.
   if ( isGuess .and. (.not. quick_method%writeSAD) ) verbose = .false.

    !Form the exchange-correlation quadrature if DFT is requested
    if ( ( quick_method%DFT &
#ifdef CEW
        .or. quick_cew%use_cew &
#endif
        ) .and. .not. isGuess ) then

        if (master) call PrtAct(ioutfile,"Begin XC Quadrature Formation")

        call form_dft_grid(quick_dft_grid, quick_xcg_tmp)

        if (master) call print_grid_info(quick_dft_grid)

        if (master) call PrtAct(ioutfile,"End XC Quadrature Formation")
    endif

   if (master) then
      if (verbose) call PrtAct(ioutfile,"Begin Energy Calculation")
      ! Build a transformation matrix X and overlap matrix
      call fullX

      ! if it's a div-con calculate, construct Div & Con matrices, Overlap,X, and PDC
      !if (quick_method%DivCon) then
      !   call DivideS
      !   call DivideX
      !   call PDCDivided
      !endif

      !Classical Nuclear-Nuclear interaction energy
      quick_qm_struct%Ecore=0.d0      ! atom-extcharge and atom-atom replusion
      quick_qm_struct%ECharge=0d0     ! extcharge-extcharge interaction


      if (natom > 1) then
         !                    qi*qj
         ! E=sigma(i,j=1,n)----------
         !                   |ri-rj|
         do I=1,natom+quick_molspec%nextatom
            do J=I+1,natom+quick_molspec%nextatom
               if(i<=natom .and. j<=natom)then                     ! the atom to atom replusion
                  distance = rootSquare(xyz(1:3,i), xyz(1:3,j), 3)
                  quick_qm_struct%Ecore = quick_molspec%chg(I)*quick_molspec%chg(J)/distance+quick_qm_struct%Ecore
                  elseif(i<=natom .and. j>natom)then                  ! the atom to external point charge replusion
                  distance = rootSquare(xyz(1:3,i), quick_molspec%extxyz(1:3,j-natom), 3)
                  quick_qm_struct%Ecore = quick_molspec%chg(I)*quick_molspec%extchg(J-natom)/distance+quick_qm_struct%Ecore
                  elseif(i>natom .and. j>natom)then                   ! external to external point charge repulsion
                  distance = rootSquare(quick_molspec%extxyz(1:3,i-natom), quick_molspec%extxyz(1:3,j-natom), 3)
                  quick_qm_struct%ECharge = quick_qm_struct%ECharge + &

                        quick_molspec%extchg(I-natom)*quick_molspec%extchg(J-natom)/distance
               endif
            enddo
         enddo
      endif

      ! A uniform external electric field contributes the classical
      ! charge-field term to Ecore.  The electronic counterpart is added
      ! as a one-electron operator in quick_oei_module.
      if (quick_method%external_efield) then
         do I=1,natom
            field_dot_r = quick_method%external_efield_vector(1) &
               *(xyz(1,I)-quick_method%external_efield_origin(1)) &
               + quick_method%external_efield_vector(2) &
               *(xyz(2,I)-quick_method%external_efield_origin(2)) &
               + quick_method%external_efield_vector(3) &
               *(xyz(3,I)-quick_method%external_efield_origin(3))
            quick_qm_struct%Ecore = quick_qm_struct%Ecore - quick_molspec%chg(I)*field_dot_r
         enddo

         do I=1,quick_molspec%nextatom
            field_dot_r = quick_method%external_efield_vector(1) &
               *(quick_molspec%extxyz(1,I)-quick_method%external_efield_origin(1)) &
               + quick_method%external_efield_vector(2) &
               *(quick_molspec%extxyz(2,I)-quick_method%external_efield_origin(2)) &
               + quick_method%external_efield_vector(3) &
               *(quick_molspec%extxyz(3,I)-quick_method%external_efield_origin(3))
            quick_qm_struct%Ecore = quick_qm_struct%Ecore - quick_molspec%extchg(I)*field_dot_r
         enddo
      endif
   endif
   ! Converge the density matrix.
#ifdef MPIV
   !-------------- MPI / ALL NODES ----------------------------------
   if (bMPI) then
      quick_qm_struct%NBSuse => NBSuse

      call MPI_BCAST(NBSuse,1,mpi_integer,0,MPI_COMM_WORLD,mpierror)

      if(.not. master) call allocate_quick_qm_struct_fullx(quick_qm_struct)

      call MPI_BCAST(quick_qm_struct%s,nbasis*nbasis,mpi_double_precision,0,MPI_COMM_WORLD,mpierror)
      call MPI_BCAST(quick_qm_struct%x,nbasis*NBSuse,mpi_double_precision,0,MPI_COMM_WORLD,mpierror)
      call MPI_BCAST(quick_qm_struct%Ecore,1,mpi_double_precision,0,MPI_COMM_WORLD,mpierror)
   endif
   !-------------- END MPI / ALL NODES ------------------------------
#endif
   ! scf process to obtain converged density matrix and coeffecient so that we can
   ! process to next step, the energy compuation. the restrited system will call scf and
   ! unrestred system will call uscf. the logical variable failed indicated failed convergence.
   ! convergence criteria can be set in the job or default value.
   if (quick_method%UNRST) then
      if(isGuess) then
        !call uscf_sad(verbose,ierr)
        !call uscf_sad_new(ierr)
      else
        call uscf(ierr)       ! unrestricted system
      endif
   else
      call scf(ierr)        ! restricted system
   endif

   if (quick_mbx_active()) then
      call quick_mbx_refresh_energy(ierr)
      if (ierr /= 0) return
   endif

   !--------------- MPI/MASTER --------------------------
   if (master) then

      ! Fisrt, it is PB model, we need calculate the energy for PB Sol.
      !
      ! Blocked by Yipu Miao
      !
      if(quick_method%PBSOL)then
         if (quick_method%UNRST) then
            !       if (quick_method%HF) call UHFEnergy
            !       if (quick_method%DFT) call uDFTEnergy
            !        if (quick_method%SEDFT) call uSEDFTEnergy
         else
            !        if (quick_method%HF) call HFEnergy
            !        if (quick_method%DFT) call DFTenergy
            !        if (quick_method%SEDFT) call SEDFTenergy
         endif
      endif

      ! Now that we have a converged density matrix, it is time to
      ! calculate the energy.  It equals to the summation of different
      ! parts: electronic energy, core-core repulsion, and some other energy
      ! for specific job
      quick_qm_struct%Eelvac=quick_qm_struct%Eel
      if (quick_method%extcharges) then
         quick_qm_struct%Etot = quick_qm_struct%Etot + quick_qm_struct%ECharge
      endif
      quick_qm_struct%Etot = quick_qm_struct%Eel + quick_qm_struct%Ecore

      ! calculate emperical dispersion correction 
      if(quick_method%edisp) then
         SAFE_CALL(calculateDFTD3(ierr))
         quick_qm_struct%Etot=quick_qm_struct%Etot+quick_qm_struct%Edisp
      endif

      mbx_energy_au = 0.0d0
      mbx_external_energy_au = 0.0d0
      mbx_additive_energy_au = 0.0d0
      mbx_perm_total_au = 0.0d0
      mbx_perm_classical_au = 0.0d0
      mbx_perm_electronic_au = 0.0d0
      mbx_ind_total_au = 0.0d0
      mbx_ind_classical_au = 0.0d0
      mbx_ind_electronic_au = 0.0d0
      mbx_external_half_residual_au = 0.0d0
      if (quick_mbx_active()) then
         call quick_mbx_get_energy_terms(mbx_energy_au,mbx_external_energy_au,mbx_additive_energy_au, &
            mbx_perm_total_au,mbx_perm_classical_au,mbx_perm_electronic_au,mbx_ind_total_au, &
            mbx_ind_classical_au,mbx_ind_electronic_au,mbx_external_half_residual_au)
         quick_qm_struct%Etot = quick_qm_struct%Etot + mbx_additive_energy_au
      endif

      if (ioutfile.ne.0 .and. verbose) then
         write (ioutfile,'(" ELECTRONIC ENERGY    = ",F16.9)') quick_qm_struct%Eel
         write (ioutfile,'(" CORE_CORE REPULSION  = ",F16.9)') quick_qm_struct%Ecore
         if(quick_method%edisp) then
            write (ioutfile,'(" DISPERSION CORRECTION  = ",F16.9)') quick_qm_struct%Edisp
         endif
         if (quick_method%extcharges) then
            write (ioutfile,'(" EXT CHARGE REPULSION = ",F16.9)') quick_qm_struct%ECharge
         endif
         if (quick_mbx_active()) then
            write (ioutfile,'(" QUICK-MBX MBX TOTAL       = ",F16.9)') mbx_energy_au
            write (ioutfile,'(" QUICK-MBX ADDITIVE        = ",F16.9)') mbx_additive_energy_au
            write (ioutfile,'(" QUICK-MBX PERM TOTAL      = ",F16.9)') mbx_perm_total_au
            write (ioutfile,'(" QUICK-MBX PERM CLASSICAL  = ",F16.9)') mbx_perm_classical_au
            write (ioutfile,'(" QUICK-MBX PERM ELECTRONIC = ",F16.9)') mbx_perm_electronic_au
            write (ioutfile,'(" QUICK-MBX IND TOTAL       = ",F16.9)') mbx_ind_total_au
            write (ioutfile,'(" QUICK-MBX IND CLASSICAL   = ",F16.9)') mbx_ind_classical_au
            write (ioutfile,'(" QUICK-MBX IND ELECTRONIC  = ",F16.9)') mbx_ind_electronic_au
            write (ioutfile,'(" QUICK-MBX MBX EXT PERM/2  = ",F16.9)') mbx_external_energy_au
            write (ioutfile,'(" QUICK-MBX EXT PERM CHECK  = ",ES16.6)') mbx_external_half_residual_au
            write (ioutfile,'(" QUICK-MBX UNITS hartree; QUICK EFG convention dE_i/dC_j")')
         endif
         write (ioutfile,'(" TOTAL ENERGY         = ",F16.9)') quick_qm_struct%Etot
         call prtact(ioutfile,"End Energy calculation")
         call flush(ioutfile)
      endif
   endif
   !--------------- END MPI/MASTER ----------------------

#if defined(MPIV) || defined(MPIV_GPU)
  call MPI_BCAST(quick_qm_struct%Etot, 1, mpi_double_precision,0,MPI_COMM_WORLD,mpierror) 
#endif

end subroutine getenergy
