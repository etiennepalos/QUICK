#!/bin/bash
# (C) Copyright 2026 QUICK contributors
# All rights reserved.
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0.

#SBATCH --job-name=quick-oeprop-bomd
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --cpus-per-task=1
#SBATCH --time=24:00:00
#SBATCH --mem=48G
#SBATCH --output=quick-oeprop-bomd.%j.out

set -euo pipefail

: "${QUICK_ROOT:?Set QUICK_ROOT to the installed QUICK prefix}"
source "${QUICK_ROOT}/quick.rc"

DRIVER="${QUICK_BOMD_DRIVER:-${QUICK_ROOT}/bin/test-api-oeprop-bomd.MPI}"
CONTROL="${1:-microdroplet.nml}"

srun --ntasks="${SLURM_NTASKS}" "${DRIVER}" "${CONTROL}"
