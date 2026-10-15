! (C) Copyright 2026 QUICK contributors
! All rights reserved.
!
! This Source Code Form is subject to the terms of the Mozilla Public
! License, v. 2.0. If a copy of the MPL was not distributed with this
! file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include "util.fh"

module quick_solvation_grid_module
   use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
   use, intrinsic :: iso_fortran_env, only: int64
   use quick_gridpoints_module, only: ld0006, ld0014, ld0026, ld0038, ld0050, ld0074, &
      ld0086, ld0110, ld0146, ld0170, ld0194, ld0230, ld0266, ld0302, ld0350, ld0434, &
      ld0590, ld0770, ld0974, ld1202, ld1454, ld1730, ld2030, ld2354, ld2702, ld3074, &
      ld3470, ld3890, ld4334, ld4802, ld5294, ld5810
   implicit none
   private

   integer, parameter, public :: solvation_grid_default_nrad = 75
   integer, parameter, public :: solvation_grid_default_nang = 302
   integer, parameter :: grid_error = 53
   double precision, parameter :: four_pi = 12.566370614359172953850573533118d0
   double precision, parameter :: switch_width = 0.64d0

   ! This volume quadrature is independent of the XC and OEPROP surface grids.
   ! Coordinates, offsets, and the optional radial scales are in bohr.
   ! base_weight contains the radial Jacobian and the full solid-angle weight.
   ! weight = base_weight * partition. No density-dependent pruning is applied.
   type, public :: quick_solvation_grid_type
      integer :: npoints = 0
      integer :: nrad = 0
      integer :: nang = 0
      double precision, allocatable :: points(:, :), offsets(:, :)
      integer, allocatable :: parent(:)
      double precision, allocatable :: base_weight(:), partition(:), weight(:)
      double precision, allocatable :: atom_coords(:, :), atom_distances(:, :)
      logical, allocatable :: partition_constant(:)
   end type quick_solvation_grid_type

   public :: build_solvation_grid, destroy_solvation_grid, solvation_partition
   public :: solvation_grid_partition_derivative
   public :: solvation_angular_grid_supported

contains

   subroutine build_solvation_grid(natom, coords, nrad, nang, grid, ierr, radial_scale)
      implicit none
      integer, intent(in) :: natom, nrad, nang
      double precision, intent(in) :: coords(3, natom)
      type(quick_solvation_grid_type), intent(inout) :: grid
      integer, intent(out) :: ierr
      double precision, intent(in), optional :: radial_scale(natom)
      double precision, allocatable :: ax(:), ay(:), az(:), aw(:), rx(:), rw(:)
      double precision :: radius, radial_weight, scale, part
      integer :: iatom, irad, iang, ipoint, nreturned, alloc_status
      integer(int64) :: npoints_long

      ierr = 0
      call destroy_solvation_grid(grid)
      if (natom < 1 .or. nrad < 1 .or. .not. solvation_angular_grid_supported(nang)) then
         ierr = grid_error
         return
      endif
      if (.not. all(ieee_is_finite(coords))) then
         ierr = grid_error
         return
      endif
      if (present(radial_scale)) then
         if (.not. all(ieee_is_finite(radial_scale)) .or. any(radial_scale <= 0.d0)) then
            ierr = grid_error
            return
         endif
      endif
      npoints_long = int(natom, int64) * int(nrad, int64) * int(nang, int64)
      if (npoints_long > int(huge(grid%npoints), int64)) then
         ierr = grid_error
         return
      endif
      allocate(ax(nang), ay(nang), az(nang), aw(nang), rx(nrad), rw(nrad), stat=alloc_status)
      if (alloc_status /= 0) then
         ierr = grid_error
         return
      endif
      call angular_grid(nang, ax, ay, az, aw, nreturned)
      if (nreturned /= nang) then
         ierr = grid_error
         return
      endif
      call gauss_legendre(nrad, rx, rw, ierr)
      if (ierr /= 0) return

      grid%npoints = int(npoints_long)
      grid%nrad = nrad
      grid%nang = nang
      allocate(grid%points(3, grid%npoints), grid%offsets(3, grid%npoints), &
         grid%parent(grid%npoints), grid%base_weight(grid%npoints), &
         grid%partition(grid%npoints), grid%weight(grid%npoints), grid%atom_coords(3,natom), &
         grid%atom_distances(natom,natom), grid%partition_constant(grid%npoints), stat=alloc_status)
      if (alloc_status /= 0) then
         call destroy_solvation_grid(grid)
         ierr = grid_error
         return
      endif
      grid%atom_coords = coords
      call prepare_partition_geometry(natom,coords,grid%atom_distances,ierr)
      if (ierr /= 0) then
         call destroy_solvation_grid(grid)
         return
      endif

      ipoint = 0
      do iatom = 1, natom
         ! The map covers [0,infinity). The scale is a fixed integration
         ! parameter, not a geometry-dependent cavity radius.
         scale = 1.d0
         if (present(radial_scale)) scale = radial_scale(iatom)
         do irad = 1, nrad
            radius = scale * (1.d0 + rx(irad)) / (1.d0 - rx(irad))
            radial_weight = rw(irad) * radius**2 * 2.d0 * scale / (1.d0 - rx(irad))**2
            do iang = 1, nang
               ipoint = ipoint + 1
               grid%offsets(:, ipoint) = radius * [ax(iang), ay(iang), az(iang)]
               grid%points(:, ipoint) = coords(:, iatom) + grid%offsets(:, ipoint)
               grid%parent(ipoint) = iatom
               grid%base_weight(ipoint) = radial_weight * four_pi * aw(iang)
               call partition_worker(natom,coords,grid%atom_distances,grid%points(:,ipoint), &
                  iatom,part,ierr,constant=grid%partition_constant(ipoint))
               if (ierr /= 0) then
                  call destroy_solvation_grid(grid)
                  return
               endif
               grid%partition(ipoint) = part
               grid%weight(ipoint) = grid%base_weight(ipoint) * part
            enddo
         enddo
      enddo
      if (.not. all(ieee_is_finite(grid%points)) .or. .not. all(ieee_is_finite(grid%weight))) then
         call destroy_solvation_grid(grid)
         ierr = grid_error
      endif
   end subroutine build_solvation_grid


   subroutine destroy_solvation_grid(grid)
      implicit none
      type(quick_solvation_grid_type), intent(inout) :: grid

      if (allocated(grid%points)) deallocate(grid%points)
      if (allocated(grid%offsets)) deallocate(grid%offsets)
      if (allocated(grid%parent)) deallocate(grid%parent)
      if (allocated(grid%base_weight)) deallocate(grid%base_weight)
      if (allocated(grid%partition)) deallocate(grid%partition)
      if (allocated(grid%weight)) deallocate(grid%weight)
      if (allocated(grid%atom_coords)) deallocate(grid%atom_coords)
      if (allocated(grid%atom_distances)) deallocate(grid%atom_distances)
      if (allocated(grid%partition_constant)) deallocate(grid%partition_constant)
      grid%npoints = 0
      grid%nrad = 0
      grid%nang = 0
   end subroutine destroy_solvation_grid


   ! derivative(:,A) is the TOTAL nuclear derivative of the parent's
   ! partition at point = R_parent + fixed_offset. It includes the point's
   ! motion with its parent. Thus sum_A derivative(:,A) = 0, up to roundoff.
   ! Only O(natom) scratch is used; no all-points/all-atoms tensor is stored.
   ! Logarithmic products avoid underflow for large numbers of spheres.
   subroutine solvation_partition(natom, coords, point, iparent, part, ierr, derivative)
      implicit none
      integer, intent(in) :: natom, iparent
      double precision, intent(in) :: coords(3, natom), point(3)
      double precision, intent(out) :: part
      integer, intent(out) :: ierr
      double precision, intent(out), optional :: derivative(3, natom)
      double precision :: atom_distances(natom,natom)

      part = 0.d0
      if (present(derivative)) derivative = 0.d0
      call prepare_partition_geometry(natom,coords,atom_distances,ierr)
      if (ierr /= 0) return
      call partition_worker(natom,coords,atom_distances,point,iparent,part,ierr,derivative)
   end subroutine solvation_partition


   ! Use only with the geometry that constructed grid. This path reuses its
   ! validated interatomic distances and exact constant-partition flags.
   subroutine solvation_grid_partition_derivative(grid,natom,coords,ipoint,derivative,ierr)
      implicit none
      type(quick_solvation_grid_type), intent(in) :: grid
      integer, intent(in) :: natom,ipoint
      double precision, intent(in) :: coords(3,natom)
      double precision, intent(out) :: derivative(3,natom)
      integer, intent(out) :: ierr
      double precision :: part

      derivative = 0.d0
      ierr = grid_error
      if (ipoint < 1 .or. ipoint > grid%npoints) return
      if (.not.allocated(grid%atom_coords) .or. .not.allocated(grid%atom_distances) .or. &
         .not.allocated(grid%partition_constant)) return
      if (size(grid%atom_coords,2) /= natom) return
      if (any(grid%atom_coords /= coords)) return
      ierr = 0
      if (grid%partition_constant(ipoint)) return
      call partition_worker(natom,coords,grid%atom_distances,grid%points(:,ipoint), &
         grid%parent(ipoint),part,ierr,derivative)
   end subroutine solvation_grid_partition_derivative


   subroutine prepare_partition_geometry(natom,coords,atom_distances,ierr)
      implicit none
      integer, intent(in) :: natom
      double precision, intent(in) :: coords(3,natom)
      double precision, intent(out) :: atom_distances(natom,natom)
      integer, intent(out) :: ierr
      integer :: iatom,jatom
      double precision :: rij

      ierr = grid_error
      if (natom < 1) return
      if (.not.all(ieee_is_finite(coords))) return
      atom_distances = 0.d0
      do iatom = 1,natom
         do jatom = iatom+1,natom
            rij = norm2(coords(:,iatom)-coords(:,jatom))
            if (.not.ieee_is_finite(rij) .or. rij <= 1.d-12) return
            atom_distances(iatom,jatom) = rij
            atom_distances(jatom,iatom) = rij
         enddo
      enddo
      ierr = 0
   end subroutine prepare_partition_geometry


   subroutine partition_worker(natom,coords,atom_distances,point,iparent,part,ierr,derivative,constant)
      implicit none
      integer, intent(in) :: natom,iparent
      double precision, intent(in) :: coords(3,natom),atom_distances(natom,natom),point(3)
      double precision, intent(out) :: part
      integer, intent(out) :: ierr
      double precision, intent(out), optional :: derivative(3,natom)
      logical, intent(out), optional :: constant
      double precision :: distances(natom), units(3, natom), logw(natom), normalized(natom)
      double precision :: dlogw(3, natom), parent_dlogw(3, natom), mean_dlogw(3, natom)
      double precision :: delta(3), between(3), rij, mu, switch_value, switch_deriv
      double precision :: dmu_i(3), dmu_j(3), dmu_point(3), factor, sumw, maxlog
      double precision :: parent_distance,core_radius,parent_logw
      integer :: iatom, jatom
      logical :: parent_one

      ierr = 0
      part = 0.d0
      if (present(derivative)) derivative = 0.d0
      if (present(constant)) constant = .false.
      if (natom < 1 .or. iparent < 1 .or. iparent > natom) then
         ierr = grid_error
         return
      endif
      if (.not. all(ieee_is_finite(point))) then
         ierr = grid_error
         return
      endif
      if (natom == 1) then
         part = 1.d0
         if (present(constant)) constant = .true.
         return
      endif
      parent_distance = norm2(point-coords(:,iparent))
      core_radius = 0.5d0*(1.d0-switch_width) * &
         minval(atom_distances(:,iparent),mask=atom_distances(:,iparent)>0.d0)
      ! Triangle inequality gives mu_parent,j <= -switch_width for every
      ! other atom in this region. All competing weights are exactly zero.
      if (parent_distance < core_radius) then
         part = 1.d0
         if (present(constant)) constant = .true.
         return
      endif
      do iatom = 1, natom
         delta = point - coords(:, iatom)
         distances(iatom) = norm2(delta)
         if (present(derivative)) then
            units(:, iatom) = 0.d0
            if (distances(iatom) > 0.d0) units(:, iatom) = delta / distances(iatom)
         endif
      enddo
      if (.not. all(ieee_is_finite(distances))) then
         ierr = grid_error
         return
      endif

      ! Evaluate the requested numerator first. A zero plateau has a zero
      ! derivative, so neither the denominator nor its gradient is needed.
      parent_one = .true.
      parent_logw = 0.d0
      do jatom = 1,natom
         if (jatom == iparent) cycle
         mu = (distances(iparent)-distances(jatom))/atom_distances(iparent,jatom)
         call partition_switch(mu,switch_value,switch_deriv)
         if (switch_value == 0.d0) then
            if (present(constant)) constant = .true.
            return
         endif
         if (mu > -switch_width) parent_one = .false.
         parent_logw = parent_logw + log(switch_value)
      enddo
      if (parent_one) then
         part = 1.d0
         if (present(constant)) constant = .true.
         return
      endif
      logw = -huge(1.d0)
      logw(iparent) = parent_logw
      do iatom = 1, natom
         if (iatom == iparent) cycle
         ! Any atom killed by its pair with the parent has exactly zero
         ! normalized weight. This test preserves the original row order.
         mu = (distances(iatom)-distances(iparent))/atom_distances(iatom,iparent)
         if (mu >= switch_width) cycle
         logw(iatom) = 0.d0
         do jatom = 1, natom
            if (jatom == iatom) cycle
            rij = atom_distances(iatom,jatom)
            mu = (distances(iatom) - distances(jatom)) / rij
            call partition_switch(mu, switch_value, switch_deriv)
            if (switch_value == 0.d0) then
               logw(iatom) = -huge(1.d0)
               exit
            endif
            logw(iatom) = logw(iatom) + log(switch_value)
         enddo
      enddo
      maxlog = maxval(logw)
      if (maxlog == -huge(1.d0)) then
         ierr = grid_error
         return
      endif
      normalized = exp(logw - maxlog)
      sumw = sum(normalized)
      normalized = normalized / sumw
      part = normalized(iparent)
      if (.not. ieee_is_finite(part)) then
         ierr = grid_error
         return
      endif
      if (.not. present(derivative) .or. part == 0.d0) return

      parent_dlogw = 0.d0
      mean_dlogw = 0.d0
      do iatom = 1, natom
         if (normalized(iatom) == 0.d0) cycle
         dlogw = 0.d0
         do jatom = 1, natom
            if (jatom == iatom) cycle
            between = coords(:, iatom) - coords(:, jatom)
            rij = atom_distances(iatom,jatom)
            mu = (distances(iatom) - distances(jatom)) / rij
            call partition_switch(mu, switch_value, switch_deriv)
            if (switch_deriv == 0.d0) cycle
            factor = switch_deriv / switch_value
            dmu_i = -units(:, iatom) / rij - mu * between / rij**2
            dmu_j = units(:, jatom) / rij + mu * between / rij**2
            dmu_point = (units(:, iatom) - units(:, jatom)) / rij
            dlogw(:, iatom) = dlogw(:, iatom) + factor * dmu_i
            dlogw(:, jatom) = dlogw(:, jatom) + factor * dmu_j
            dlogw(:, iparent) = dlogw(:, iparent) + factor * dmu_point
         enddo
         mean_dlogw = mean_dlogw + normalized(iatom) * dlogw
         if (iatom == iparent) parent_dlogw = dlogw
      enddo
      derivative = part * (parent_dlogw - mean_dlogw)
      if (.not. all(ieee_is_finite(derivative))) ierr = grid_error
   end subroutine partition_worker


   pure subroutine partition_switch(mu, value, derivative)
      implicit none
      double precision, intent(in) :: mu
      double precision, intent(out) :: value, derivative
      double precision :: t, abs_t, small_value

      if (mu <= -switch_width) then
         value = 1.d0
         derivative = 0.d0
      else if (mu >= switch_width) then
         value = 0.d0
         derivative = 0.d0
      else
         t = mu / switch_width
         abs_t = abs(t)
         ! Factoring the seventh-degree Stratmann switch retains positive
         ! tiny weights near the edge without subtracting nearly equal terms.
         small_value = (1.d0 - abs_t)**4 * &
            (16.d0 + abs_t * (29.d0 + abs_t * (20.d0 + 5.d0 * abs_t))) / 32.d0
         value = small_value
         if (t < 0.d0) value = 1.d0 - small_value
         derivative = -35.d0 * (1.d0 - t*t)**3 / (32.d0 * switch_width)
      endif
   end subroutine partition_switch


   subroutine gauss_legendre(n, nodes, weights, ierr)
      implicit none
      integer, intent(in) :: n
      double precision, intent(out) :: nodes(n), weights(n)
      integer, intent(out) :: ierr
      double precision :: z, previous, p0, p1, p2, derivative
      integer :: i, j, iteration

      ierr = 0
      do i = 1, (n + 1) / 2
         z = cos(0.25d0 * four_pi * (dble(i) - 0.25d0) / (dble(n) + 0.5d0))
         do iteration = 1, 100
            p0 = 1.d0
            p1 = z
            do j = 2, n
               p2 = ((2.d0*dble(j) - 1.d0)*z*p1 - (dble(j) - 1.d0)*p0) / dble(j)
               p0 = p1
               p1 = p2
            enddo
            derivative = dble(n) * (z*p1 - p0) / (z*z - 1.d0)
            previous = z
            z = previous - p1 / derivative
            if (abs(z - previous) <= 4.d0*epsilon(1.d0)) exit
         enddo
         if (iteration > 100) then
            ierr = grid_error
            return
         endif
         ! Re-evaluate the derivative at the accepted root for its weight.
         p0 = 1.d0
         p1 = z
         do j = 2, n
            p2 = ((2.d0*dble(j) - 1.d0)*z*p1 - (dble(j) - 1.d0)*p0) / dble(j)
            p0 = p1
            p1 = p2
         enddo
         derivative = dble(n) * (z*p1 - p0) / (z*z - 1.d0)
         nodes(i) = -z
         nodes(n + 1 - i) = z
         weights(i) = 2.d0 / ((1.d0 - z*z) * derivative**2)
         weights(n + 1 - i) = weights(i)
      enddo
   end subroutine gauss_legendre


   pure logical function solvation_angular_grid_supported(nang)
      implicit none
      integer, intent(in) :: nang
      select case(nang)
      case(6,14,26,38,50,74,86,110,146,170,194,230,266,302,350,434,590,770,974, &
         1202,1454,1730,2030,2354,2702,3074,3470,3890,4334,4802,5294,5810)
         solvation_angular_grid_supported = .true.
      case default
         solvation_angular_grid_supported = .false.
      end select
   end function solvation_angular_grid_supported


   subroutine angular_grid(nang, x, y, z, w, nreturned)
      implicit none
      integer, intent(in) :: nang
      double precision, intent(out) :: x(nang), y(nang), z(nang), w(nang)
      integer, intent(out) :: nreturned

      nreturned = 0
      select case(nang)
      case(6)
         call ld0006(x,y,z,w,nreturned)
      case(14)
         call ld0014(x,y,z,w,nreturned)
      case(26)
         call ld0026(x,y,z,w,nreturned)
      case(38)
         call ld0038(x,y,z,w,nreturned)
      case(50)
         call ld0050(x,y,z,w,nreturned)
      case(74)
         call ld0074(x,y,z,w,nreturned)
      case(86)
         call ld0086(x,y,z,w,nreturned)
      case(110)
         call ld0110(x,y,z,w,nreturned)
      case(146)
         call ld0146(x,y,z,w,nreturned)
      case(170)
         call ld0170(x,y,z,w,nreturned)
      case(194)
         call ld0194(x,y,z,w,nreturned)
      case(230)
         call ld0230(x,y,z,w,nreturned)
      case(266)
         call ld0266(x,y,z,w,nreturned)
      case(302)
         call ld0302(x,y,z,w,nreturned)
      case(350)
         call ld0350(x,y,z,w,nreturned)
      case(434)
         call ld0434(x,y,z,w,nreturned)
      case(590)
         call ld0590(x,y,z,w,nreturned)
      case(770)
         call ld0770(x,y,z,w,nreturned)
      case(974)
         call ld0974(x,y,z,w,nreturned)
      case(1202)
         call ld1202(x,y,z,w,nreturned)
      case(1454)
         call ld1454(x,y,z,w,nreturned)
      case(1730)
         call ld1730(x,y,z,w,nreturned)
      case(2030)
         call ld2030(x,y,z,w,nreturned)
      case(2354)
         call ld2354(x,y,z,w,nreturned)
      case(2702)
         call ld2702(x,y,z,w,nreturned)
      case(3074)
         call ld3074(x,y,z,w,nreturned)
      case(3470)
         call ld3470(x,y,z,w,nreturned)
      case(3890)
         call ld3890(x,y,z,w,nreturned)
      case(4334)
         call ld4334(x,y,z,w,nreturned)
      case(4802)
         call ld4802(x,y,z,w,nreturned)
      case(5294)
         call ld5294(x,y,z,w,nreturned)
      case(5810)
         call ld5810(x,y,z,w,nreturned)
      end select
   end subroutine angular_grid

end module quick_solvation_grid_module
