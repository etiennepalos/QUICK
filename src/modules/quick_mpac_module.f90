#include "util.fh"
!---------------------------------------------------------------------!
! MPAC/HFAC energy module.                                            !
!                                                                     !
! This module evaluates closed-shell RHF adiabatic-connection          !
! correlation models that use canonical or scaled opposite-spin MP2,   !
! exact HF exchange, and density-grid strong-interaction ingredients.  !
! The MP2 tensor                                                       !
! transformation is shared with quick_mbpt_module; the density terms   !
! use the same pruned SG grid and AO density routines used by QUICK DFT.!
!---------------------------------------------------------------------!

module quick_mpac_module

   use quick_mbpt_module, only: mbpt_energy_type, transform_ao_to_iajb, &
      compute_mbpt_energies

   implicit none
   private

   public :: calculate_mpac

   double precision, parameter :: MPAC_PI = 3.1415926535897932384626433832795d0
   double precision, parameter :: MPAC_A_PC = -1.451d0
   double precision, parameter :: MPAC_B_PC = 5.317d-3
   double precision, parameter :: HFAC_A_HF = -1.44423075d0
   double precision, parameter :: HFAC_C_HF = 2.8687d0
   double precision, parameter :: HFAC_MU_EL_GE2 = 0.399d0
   double precision, parameter :: HFAC_MU_HALF_GE2 = 1.601d0
   double precision, parameter :: HFAC_GGA_C_EL = 20.0d0
   double precision, parameter :: HFAC_GGA_C_HALF = 14.0d0
   double precision, parameter :: HFAC_SPIN_BETA = 1.05d0

   type mpac_grid_terms_type
      double precision :: rho_4_3 = 0.0d0
      double precision :: grad_square_over_rho_4_3 = 0.0d0
      double precision :: rho_3_2 = 0.0d0
      double precision :: grad_square_over_rho_7_6 = 0.0d0
      double precision :: hfac_e_el = 0.0d0
      double precision :: hfac_w_half = 0.0d0
      double precision :: hfac_w_three_quarter = 0.0d0
   end type mpac_grid_terms_type

contains

   subroutine calculate_mpac()

      use quick_basis_module, only: nbasis
      use quick_calculated_module, only: quick_qm_struct
      use quick_eri_cshell_module, only: getCshellEriTensor
      use quick_files_module, only: ioutfile
      use quick_gridpoints_module, only: quick_dft_grid, quick_xcg_tmp, &
         form_dft_grid, deform_dft_grid, print_grid_info
      use quick_method_module, only: quick_method
      use quick_molspec_module, only: quick_molspec
      use quick_mpi_module, only: master
#ifdef MPIV
      use mpi
      use quick_mpi_module, only: bMPI, mpierror
#endif
      implicit none

      double precision, allocatable :: eri_ao(:,:,:,:)
      double precision, allocatable :: eri_iajb(:,:,:,:)
      double precision :: e_x_hf, e_spl2, e_os_spl2, e_mpac25
      double precision :: e_os_mpac25, e_hfac24, e_ueg_ihf
      double precision :: pc_energy, hfac_w_inf
      integer :: nocc, nvir, nmo
      type(mbpt_energy_type) :: mbpt_energies
      type(mpac_grid_terms_type) :: grid_terms

      nocc = quick_molspec%nelec/2
      nmo = quick_qm_struct%NBSuse
      nvir = nmo - nocc

      if (master) call PrtAct(ioutfile,"Begin MPAC/HFAC Calculation")

      if (quick_method%UNRST .or. mod(quick_molspec%nelec,2).ne.0) then
         if (master) call PrtErr(ioutfile,"MPAC/HFAC currently supports closed-shell RHF references only.")
         call quick_exit(ioutfile,1)
      endif

      if (nvir.lt.1) then
         if (master) call PrtErr(ioutfile,"MPAC/HFAC requires at least one virtual orbital.")
         call quick_exit(ioutfile,1)
      endif

      ! Build the QUICK molecular grid on demand.  MPAC25 uses the PC
      ! density model; HFAC24 also needs W_1/2 and nuclear cusp terms.
      call form_dft_grid(quick_dft_grid, quick_xcg_tmp)
      if (master) call print_grid_info(quick_dft_grid)

      call compute_mpac_grid_terms(grid_terms)

      if (master) then
         grid_terms%hfac_w_three_quarter = nuclear_cusp_w_three_quarter()
      endif

      call deform_dft_grid(quick_dft_grid)

      allocate(eri_ao(nbasis,nbasis,nbasis,nbasis))
      call getCshellEriTensor(eri_ao)

      if (master) then
         allocate(eri_iajb(nocc,nvir,nocc,nvir))
         call transform_ao_to_iajb(eri_ao,eri_iajb,nocc,nvir)
         call compute_mbpt_energies(eri_iajb,nocc,nvir,mbpt_energies)

         e_x_hf = exact_exchange_energy(eri_ao)
         pc_energy = pc_model(grid_terms)
         hfac_w_inf = grid_terms%hfac_e_el + e_x_hf

         call print_mpac_ingredients(mbpt_energies, grid_terms, e_x_hf, &
            pc_energy, hfac_w_inf)

         if (quick_method%SPL2) then
            e_spl2 = spl2_correlation(e_x_hf, mbpt_energies%canonical, &
               grid_terms)
            write (iOutFile,'("SPL2 CORRELATION             =",F20.12)') e_spl2
            write (iOutFile,'("ESPL2                        =",F20.12)') &
               quick_qm_struct%Etot + e_spl2
         endif

         if (quick_method%OSSPL2) then
            e_os_spl2 = os_spl2_correlation(e_x_hf, mbpt_energies%opposite_spin, &
               grid_terms)
            write (iOutFile,'("OS-SPL2 CORRELATION          =",F20.12)') e_os_spl2
            write (iOutFile,'("EOS-SPL2                     =",F20.12)') &
               quick_qm_struct%Etot + e_os_spl2
         endif

         if (quick_method%MPAC25) then
            e_mpac25 = mpac25_correlation(e_x_hf, mbpt_energies%canonical, &
               grid_terms)
            write (iOutFile,'("MPAC25 CORRELATION           =",F20.12)') e_mpac25
            write (iOutFile,'("EMPAC25                      =",F20.12)') &
               quick_qm_struct%Etot + e_mpac25
         endif

         if (quick_method%OSMPAC25) then
            e_os_mpac25 = mpac25_correlation(e_x_hf, &
               quick_method%os_mpac25_scale*mbpt_energies%opposite_spin, &
               grid_terms)
            write (iOutFile,'("OS-MPAC25 CORRELATION        =",F20.12)') e_os_mpac25
            write (iOutFile,'("EOS-MPAC25                   =",F20.12)') &
               quick_qm_struct%Etot + e_os_mpac25
         endif

         if (quick_method%HFAC24) then
            call validate_hfac_inputs(hfac_w_inf, grid_terms%hfac_w_half, &
               grid_terms%hfac_w_three_quarter)
            e_ueg_ihf = ueg_ihf_correlation(hfac_w_inf, &
               grid_terms%hfac_w_half, grid_terms%hfac_w_three_quarter)
            e_hfac24 = hfac24_correlation(mbpt_energies%canonical, hfac_w_inf, &
               grid_terms%hfac_w_half, grid_terms%hfac_w_three_quarter, e_ueg_ihf)
            write (iOutFile,'("HFAC24 UEGIHF CORRELATION    =",F20.12)') e_ueg_ihf
            write (iOutFile,'("HFAC24 CORRELATION           =",F20.12)') e_hfac24
            write (iOutFile,'("EHFAC24                      =",F20.12)') &
               quick_qm_struct%Etot + e_hfac24
         endif

         quick_qm_struct%EMP2 = mbpt_energies%canonical
         deallocate(eri_iajb)
      endif

      deallocate(eri_ao)

#ifdef MPIV
      if (bMPI) call MPI_BARRIER(MPI_COMM_WORLD,mpierror)
#endif

      if (master) call PrtAct(ioutfile,"End MPAC/HFAC Calculation")

   end subroutine calculate_mpac

   subroutine compute_mpac_grid_terms(terms)

      use quick_basis_module, only: nbasis, phixiao, dphidxxiao, dphidyxiao, &
         dphidzxiao
      use quick_gridpoints_module, only: quick_dft_grid
      use quick_method_module, only: quick_method
      use quick_mpi_module, only: master
#ifdef MPIV
      use mpi
      use quick_mpi_module, only: bMPI, mpierror, mpirank
#endif
      implicit none

      type(mpac_grid_terms_type), intent(out) :: terms

      double precision :: densitya, densityb, density, grad_square, grad_norm
      double precision :: gax, gay, gaz, gbx, gby, gbz
      double precision :: gridx, gridy, gridz, phi, dphidx, dphidy, dphidz
      double precision :: reduced_gradient, f_el, f_half, weight
      double precision :: local_values(6), global_values(6)
      integer :: ibin, igp, ibas, icount
      integer :: ibin_first, ibin_last

      terms = mpac_grid_terms_type()
      local_values(:) = 0.0d0
      global_values(:) = 0.0d0

#ifdef MPIV
      if (bMPI) then
         ibin_first = quick_dft_grid%igridptll(mpirank+1)
         ibin_last = quick_dft_grid%igridptul(mpirank+1)
      else
         ibin_first = 1
         ibin_last = quick_dft_grid%nbins
      endif
#else
      ibin_first = 1
      ibin_last = quick_dft_grid%nbins
#endif

      do ibin = ibin_first, ibin_last
         igp = quick_dft_grid%bin_counter(ibin)+1
         do while (igp < quick_dft_grid%bin_counter(ibin+1)+1)
            weight = quick_dft_grid%gridb_weight(igp)

            if (weight.ge.quick_method%DMCutoff) then
               gridx = quick_dft_grid%gridxb(igp)
               gridy = quick_dft_grid%gridyb(igp)
               gridz = quick_dft_grid%gridzb(igp)

               icount = quick_dft_grid%basf_counter(ibin)+1
               do while (icount < quick_dft_grid%basf_counter(ibin+1)+1)
                  ibas = quick_dft_grid%basf(icount)+1
                  call pteval_new_imp(gridx,gridy,gridz,phi,dphidx,dphidy, &
                     dphidz,ibas,icount)
                  phixiao(ibas) = phi
                  dphidxxiao(ibas) = dphidx
                  dphidyxiao(ibas) = dphidy
                  dphidzxiao(ibas) = dphidz
                  icount = icount + 1
               enddo

               call denspt_new_imp(gridx,gridy,gridz,densitya,densityb,gax,gay, &
                  gaz,gbx,gby,gbz,ibin)

               density = densitya + densityb
               if (density.gt.0.0d0) then
                  grad_square = (gax+gbx)**2 + (gay+gby)**2 + (gaz+gbz)**2
                  local_values(1) = local_values(1) + weight*density**(4.0d0/3.0d0)
                  local_values(3) = local_values(3) + weight*density**(3.0d0/2.0d0)

                  if (density.gt.quick_method%mpac_rho_trunc) then
                     local_values(2) = local_values(2) + weight*grad_square &
                        / density**(4.0d0/3.0d0)
                     local_values(4) = local_values(4) + weight*grad_square &
                        / density**(7.0d0/6.0d0)

                     grad_norm = dsqrt(grad_square)
                     reduced_gradient = grad_norm/(2.0d0*(3.0d0*MPAC_PI**2) &
                        **(1.0d0/3.0d0)*density**(4.0d0/3.0d0))
                     f_el = hfac_enhancement_factor(reduced_gradient, &
                        HFAC_MU_EL_GE2, HFAC_GGA_C_EL)
                     f_half = hfac_enhancement_factor(reduced_gradient, &
                        HFAC_MU_HALF_GE2, HFAC_GGA_C_HALF)

                     local_values(5) = local_values(5) + HFAC_A_HF*weight &
                        * density**(4.0d0/3.0d0)*f_el
                     local_values(6) = local_values(6) + HFAC_C_HF*weight &
                        * density**(3.0d0/2.0d0)*f_half
                  endif
               endif
            endif

            igp = igp + 1
         enddo
      enddo

#ifdef MPIV
      if (bMPI) then
         call MPI_REDUCE(local_values,global_values,6,mpi_double_precision, &
            MPI_SUM,0,MPI_COMM_WORLD,mpierror)
         if (master) call assign_grid_terms(terms,global_values)
      else
         call assign_grid_terms(terms,local_values)
      endif
#else
      call assign_grid_terms(terms,local_values)
#endif

   end subroutine compute_mpac_grid_terms

   subroutine assign_grid_terms(terms,values)

      implicit none

      type(mpac_grid_terms_type), intent(out) :: terms
      double precision, intent(in) :: values(6)

      terms%rho_4_3 = values(1)
      terms%grad_square_over_rho_4_3 = values(2)
      terms%rho_3_2 = values(3)
      terms%grad_square_over_rho_7_6 = values(4)
      terms%hfac_e_el = values(5)
      terms%hfac_w_half = values(6)
      terms%hfac_w_three_quarter = 0.0d0

   end subroutine assign_grid_terms

   double precision function exact_exchange_energy(eri_ao)

      use quick_calculated_module, only: quick_qm_struct
      implicit none

      double precision, intent(in) :: eri_ao(:,:,:,:)
      double precision :: d_munu, d_lamsig
      integer :: mu, nu, lambda, sigma, nao

      nao = size(eri_ao,1)
      exact_exchange_energy = 0.0d0

      do mu=1,nao
         do nu=1,nao
            d_munu = quick_qm_struct%dense(mu,nu)
            do lambda=1,nao
               do sigma=1,nao
                  d_lamsig = quick_qm_struct%dense(lambda,sigma)
                  exact_exchange_energy = exact_exchange_energy &
                     - 0.25d0*d_munu*d_lamsig*eri_ao(mu,lambda,nu,sigma)
               enddo
            enddo
         enddo
      enddo

   end function exact_exchange_energy

   double precision function pc_model(terms)

      implicit none

      type(mpac_grid_terms_type), intent(in) :: terms

      pc_model = MPAC_A_PC*terms%rho_4_3 &
         + MPAC_B_PC*terms%grad_square_over_rho_4_3

   end function pc_model

   double precision function spl2_correlation(e_x_hf,e_mp2_corr,terms)

      use quick_method_module, only: quick_method
      implicit none

      double precision, intent(in) :: e_x_hf, e_mp2_corr
      type(mpac_grid_terms_type), intent(in) :: terms

      spl2_correlation = spl2_correlation_with_params(e_x_hf,e_mp2_corr,terms, &
         quick_method%spl2_b2, quick_method%spl2_m2, &
         quick_method%spl2_alpha, quick_method%spl2_beta)

   end function spl2_correlation

   double precision function os_spl2_correlation(e_x_hf,e_mp2_os,terms)

      use quick_method_module, only: quick_method
      implicit none

      double precision, intent(in) :: e_x_hf, e_mp2_os
      type(mpac_grid_terms_type), intent(in) :: terms

      os_spl2_correlation = spl2_correlation_with_params(e_x_hf, &
         quick_method%os_spl2_scale*e_mp2_os, terms, &
         quick_method%os_spl2_b2, quick_method%os_spl2_m2, &
         quick_method%os_spl2_alpha, quick_method%os_spl2_beta)

   end function os_spl2_correlation

   double precision function spl2_correlation_with_params(e_x_hf,e_mp2_corr, &
      terms,b2,m2,alpha,beta)

      implicit none

      double precision, intent(in) :: e_x_hf, e_mp2_corr
      double precision, intent(in) :: b2, m2, alpha, beta
      type(mpac_grid_terms_type), intent(in) :: terms

      double precision :: w0, wd0, winf, root_arg
      double precision :: denominator, first_term, second_term

      w0 = e_x_hf
      wd0 = 2.0d0*e_mp2_corr
      winf = alpha*pc_model(terms) + beta*w0

      root_arg = (m2 + b2*m2 + w0 - 2.0d0*wd0 - winf)/(m2 + w0 - winf)
      first_term = winf - (2.0d0*(-1.0d0 + dsqrt(1.0d0 + b2))*m2)/b2
      denominator = b2*m2 - 2.0d0*wd0
      second_term = 2.0d0*(-1.0d0 + dsqrt(root_arg)) &
         *(m2 + w0 - winf)**2/denominator
      spl2_correlation_with_params = first_term + second_term - w0

   end function spl2_correlation_with_params

   double precision function mpac25_correlation(e_x_hf,e_mp2_corr,terms)

      use quick_method_module, only: quick_method
      implicit none

      double precision, intent(in) :: e_x_hf, e_mp2_corr
      type(mpac_grid_terms_type), intent(in) :: terms

      double precision :: w0, wd0, winf, denom, numerator_factor
      double precision :: denominator_factor

      w0 = e_x_hf
      wd0 = 2.0d0*e_mp2_corr
      winf = quick_method%mpac25_alpha*pc_model(terms) &
         + quick_method%mpac25_beta*w0

      denom = -2.0d0*wd0 + winf*quick_method%mpac25_d2**4
      numerator_factor = 1.0d0 + 2.0d0 &
         * (wd0 - winf*quick_method%mpac25_d1**2)/denom
      denominator_factor = dsqrt(1.0d0 + quick_method%mpac25_d1**2)
      denominator_factor = denominator_factor + 2.0d0 &
         * (1.0d0 + quick_method%mpac25_d2**4)**0.25d0 &
         * (wd0 - winf*quick_method%mpac25_d1**2)/denom

      mpac25_correlation = winf - (winf*numerator_factor)/denominator_factor

   end function mpac25_correlation

   double precision function hfac_enhancement_factor(s,mu,c)

      implicit none

      double precision, intent(in) :: s, mu, c
      double precision :: kappa_s

      kappa_s = c/(1.0d0 + s*s)
      hfac_enhancement_factor = 1.0d0 + kappa_s &
         - kappa_s/(1.0d0 + mu*s*s/kappa_s)

   end function hfac_enhancement_factor

   double precision function nuclear_cusp_w_three_quarter()

      use quick_method_module, only: quick_method
      use quick_molspec_module, only: natom, quick_molspec, xyz
      implicit none

      double precision :: rho, prefactor
      integer :: iatom

      prefactor = (4.0d0*MPAC_PI)**0.25d0/4.0d0
      nuclear_cusp_w_three_quarter = 0.0d0

      do iatom=1,natom
         if (quick_molspec%chg(iatom).gt.0.0d0) then
            rho = total_density_at_point(xyz(1,iatom),xyz(2,iatom),xyz(3,iatom))
            rho = max(rho, quick_method%mpac_rho_trunc)
            nuclear_cusp_w_three_quarter = nuclear_cusp_w_three_quarter &
               + prefactor*quick_molspec%chg(iatom)*rho**0.25d0 &
               * epsilon_three_quarter(0.0d0)
         endif
      enddo

   end function nuclear_cusp_w_three_quarter

   double precision function total_density_at_point(gridx,gridy,gridz)

      use quick_basis_module, only: nbasis
      use quick_calculated_module, only: quick_qm_struct
      implicit none

      double precision, intent(in) :: gridx, gridy, gridz
      double precision :: phi(nbasis), dphidx, dphidy, dphidz
      double precision :: density_half
      integer :: ibas, jbas

      do ibas=1,nbasis
         call pteval(gridx,gridy,gridz,phi(ibas),dphidx,dphidy,dphidz,ibas)
      enddo

      density_half = 0.0d0
      do ibas=1,nbasis
         density_half = density_half &
            + 0.5d0*quick_qm_struct%dense(ibas,ibas)*phi(ibas)*phi(ibas)
         do jbas=ibas+1,nbasis
            density_half = density_half &
               + quick_qm_struct%dense(jbas,ibas)*phi(ibas)*phi(jbas)
         enddo
      enddo

      total_density_at_point = 2.0d0*density_half

   end function total_density_at_point

   double precision function epsilon_three_quarter(zeta)

      implicit none

      double precision, intent(in) :: zeta
      double precision :: zeta_clipped, sigma

      zeta_clipped = max(-1.0d0,min(1.0d0,zeta))
      sigma = 0.5d0*(1.0d0 + zeta_clipped*zeta_clipped)
      epsilon_three_quarter = -2.002d0 - 1.588d0*sigma + 0.394d0*sigma*sigma

   end function epsilon_three_quarter

   subroutine validate_hfac_inputs(w_inf,w_half,w_three_quarter)

      use quick_files_module, only: ioutfile
      implicit none

      double precision, intent(in) :: w_inf, w_half, w_three_quarter

      if (w_inf.ge.0.0d0) then
         call PrtErr(ioutfile,"HFAC24 expects a negative W_c,infty.")
         call quick_exit(ioutfile,1)
      endif

      if (w_half.le.0.0d0) then
         call PrtErr(ioutfile,"HFAC24 expects a positive W_1/2.")
         call quick_exit(ioutfile,1)
      endif

      if (-w_three_quarter/w_half.lt.0.0d0) then
         call PrtErr(ioutfile,"HFAC24 expects -W_3/4/W_1/2 to be nonnegative.")
         call quick_exit(ioutfile,1)
      endif

   end subroutine validate_hfac_inputs

   double precision function ueg_ihf_integrand(alpha,w_inf,w_half,w_three_quarter)

      use quick_method_module, only: quick_method
      implicit none

      double precision, intent(in) :: alpha, w_inf, w_half, w_three_quarter
      double precision :: q, z, g, b, c, root, quarter, denom

      q = -w_inf/w_half
      z = -w_three_quarter/w_half

      if (z.eq.0.0d0) then
         g = 0.0d0
      else
         denom = -q*z*z + 8.0d0*quick_method%hfac24_d1 + 8.0d0
         g = (2.0d0*dsqrt(2.0d0*q*(1.0d0+quick_method%hfac24_d1)) &
            + q*z)*(1.0d0+quick_method%hfac24_d2)*z/denom
      endif

      b = -w_inf*(1.0d0 + quick_method%hfac24_d2 + g)**2 &
         /(1.0d0 + quick_method%hfac24_d1)
      c = 0.25d0*q*q*(1.0d0 + quick_method%hfac24_d2 + g)**4 &
         /(1.0d0 + quick_method%hfac24_d1)**2
      root = dsqrt(alpha*c + 1.0d0)
      quarter = (alpha*c + 1.0d0)**0.25d0

      ueg_ihf_integrand = w_inf + b*(2.0d0 + c*alpha &
         + 2.0d0*quick_method%hfac24_d1*root) &
         /(2.0d0*root*(quick_method%hfac24_d2 + g*quarter + root)**2)

   end function ueg_ihf_integrand

   double precision function ueg_ihf_correlation(w_inf,w_half,w_three_quarter)

      use quick_method_module, only: quick_method
      implicit none

      double precision, intent(in) :: w_inf, w_half, w_three_quarter
      double precision, allocatable :: alpha(:), weights(:)
      integer :: iq

      allocate(alpha(quick_method%hfac24_nquad))
      allocate(weights(quick_method%hfac24_nquad))
      call gauss_legendre_01(quick_method%hfac24_nquad,alpha,weights)

      ueg_ihf_correlation = 0.0d0
      do iq=1,quick_method%hfac24_nquad
         ueg_ihf_correlation = ueg_ihf_correlation + weights(iq) &
            * ueg_ihf_integrand(alpha(iq),w_inf,w_half,w_three_quarter)
      enddo

      deallocate(alpha)
      deallocate(weights)

   end function ueg_ihf_correlation

   double precision function hfac24_correlation(e_mp2_corr,w_inf,w_half, &
      w_three_quarter,e_ueg_ihf)

      use quick_method_module, only: quick_method
      implicit none

      double precision, intent(in) :: e_mp2_corr, w_inf, w_half
      double precision, intent(in) :: w_three_quarter, e_ueg_ihf
      double precision, allocatable :: alpha(:), weights(:)
      double precision :: w_ueg, mp2_line, exponent, denom, switch
      integer :: iq

      if (e_mp2_corr.eq.0.0d0) then
         hfac24_correlation = 0.0d0
         return
      endif

      allocate(alpha(quick_method%hfac24_nquad))
      allocate(weights(quick_method%hfac24_nquad))
      call gauss_legendre_01(quick_method%hfac24_nquad,alpha,weights)

      hfac24_correlation = 0.0d0
      do iq=1,quick_method%hfac24_nquad
         w_ueg = ueg_ihf_integrand(alpha(iq),w_inf,w_half,w_three_quarter)
         mp2_line = 2.0d0*alpha(iq)*e_mp2_corr
         exponent = max(-700.0d0,min(700.0d0,1000.0d0*(w_ueg-mp2_line)))
         denom = 1.0d0 - 2.0d0*alpha(iq)*e_mp2_corr*dexp(exponent)
         switch = erfc(quick_method%hfac24_kappa*alpha(iq)*e_mp2_corr &
            / e_ueg_ihf)/denom
         hfac24_correlation = hfac24_correlation + weights(iq) &
            * (mp2_line*switch + w_ueg*(1.0d0-switch))
      enddo

      deallocate(alpha)
      deallocate(weights)

   end function hfac24_correlation

   subroutine gauss_legendre_01(n,x,w)

      implicit none

      integer, intent(in) :: n
      double precision, intent(out) :: x(n), w(n)
      double precision :: z, z_old, p1, p2, p3, pp
      integer :: i, j, m

      m = (n+1)/2
      do i=1,m
         z = dcos(MPAC_PI*(dble(i)-0.25d0)/(dble(n)+0.5d0))

         do
            p1 = 1.0d0
            p2 = 0.0d0
            do j=1,n
               p3 = p2
               p2 = p1
               p1 = ((2.0d0*dble(j)-1.0d0)*z*p2 &
                  - (dble(j)-1.0d0)*p3)/dble(j)
            enddo

            pp = dble(n)*(z*p1-p2)/(z*z-1.0d0)
            z_old = z
            z = z_old - p1/pp
            if (dabs(z-z_old).lt.1.0d-14) exit
         enddo

         x(i) = 0.5d0*(1.0d0-z)
         x(n+1-i) = 0.5d0*(1.0d0+z)
         w(i) = 1.0d0/((1.0d0-z*z)*pp*pp)
         w(n+1-i) = w(i)
      enddo

   end subroutine gauss_legendre_01

   subroutine print_mpac_ingredients(mbpt_energies,terms,e_x_hf,pc_energy, &
      hfac_w_inf)

      use quick_files_module, only: ioutfile
      implicit none

      type(mbpt_energy_type), intent(in) :: mbpt_energies
      type(mpac_grid_terms_type), intent(in) :: terms
      double precision, intent(in) :: e_x_hf, pc_energy, hfac_w_inf

      write (iOutFile,'("MPAC MP2 OPPOSITE-SPIN      =",F20.12)') &
         mbpt_energies%opposite_spin
      write (iOutFile,'("MPAC MP2 SAME-SPIN          =",F20.12)') &
         mbpt_energies%same_spin
      write (iOutFile,'("MPAC MP2 CORRELATION        =",F20.12)') &
         mbpt_energies%canonical
      write (iOutFile,'("MPAC EXACT EXCHANGE         =",F20.12)') e_x_hf
      write (iOutFile,'("MPAC RHO^(4/3)              =",F20.12)') terms%rho_4_3
      write (iOutFile,'("MPAC GRAD2/RHO^(4/3)        =",F20.12)') &
         terms%grad_square_over_rho_4_3
      write (iOutFile,'("MPAC PC MODEL               =",F20.12)') pc_energy
      write (iOutFile,'("HFAC RHO^(3/2)              =",F20.12)') terms%rho_3_2
      write (iOutFile,'("HFAC GRAD2/RHO^(7/6)        =",F20.12)') &
         terms%grad_square_over_rho_7_6
      write (iOutFile,'("HFAC E_EL^GGA               =",F20.12)') terms%hfac_e_el
      write (iOutFile,'("HFAC W_C,INF                =",F20.12)') hfac_w_inf
      write (iOutFile,'("HFAC W_1/2                  =",F20.12)') terms%hfac_w_half
      write (iOutFile,'("HFAC W_3/4                  =",F20.12)') &
         terms%hfac_w_three_quarter

   end subroutine print_mpac_ingredients

end module quick_mpac_module
