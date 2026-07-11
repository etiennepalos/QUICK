/******************************************************************************
 * (C) Copyright 2026 QUICK contributors
 * All rights reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0.
 ******************************************************************************/

#include "bblock/system.h"

#include <algorithm>
#include <array>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

struct Subsystem {
    std::unique_ptr<bblock::System> system;
    std::vector<size_t> atoms;
};

struct CrossResidual {
    size_t natoms = 0;
    std::array<Subsystem, 3> systems; // full, QM shadow, MB subsystem
};

std::unique_ptr<CrossResidual> residual;

Subsystem make_subsystem(const std::vector<size_t>& monomers, const std::vector<size_t>& atom_offsets,
                         const int* nat_monomers, double* coords, char** atom_names, char** monomer_names,
                         char* json_file) {
    Subsystem result;
    result.system.reset(new bblock::System());

    for (const size_t imon : monomers) {
        const size_t first = atom_offsets[imon];
        const size_t count = static_cast<size_t>(nat_monomers[imon]);
        std::vector<double> xyz(3 * count);
        std::vector<std::string> names(count);
        std::copy(coords + 3 * first, coords + 3 * (first + count), xyz.begin());
        for (size_t i = 0; i < count; ++i) {
            names[i] = atom_names[first + i];
            result.atoms.push_back(first + i);
        }
        result.system->AddMonomer(xyz, names, monomer_names[imon]);
    }

    result.system->Initialize();
    result.system->SetUpFromJson(json_file);
    return result;
}

void evaluate_components(Subsystem& subsystem, const double* full_xyz, std::array<double, 3>& energies,
                         std::array<std::vector<double>, 3>& gradients) {
    std::vector<double> xyz(3 * subsystem.atoms.size());
    for (size_t i = 0; i < subsystem.atoms.size(); ++i) {
        std::copy(full_xyz + 3 * subsystem.atoms[i], full_xyz + 3 * subsystem.atoms[i] + 3, xyz.begin() + 3 * i);
    }
    subsystem.system->SetRealXyz(xyz);

    energies[0] = subsystem.system->TwoBodyEnergy(true);
    gradients[0] = subsystem.system->GetRealGrads();
    energies[1] = subsystem.system->ThreeBodyEnergy(true);
    gradients[1] = subsystem.system->GetRealGrads();
    energies[2] = subsystem.system->Dispersion(true);
    gradients[2] = subsystem.system->GetRealGrads();
    for (const auto& gradient : gradients) {
        if (gradient.size() != xyz.size())
            throw std::runtime_error("MBX returned an unexpected gradient size");
    }
}

} // namespace

extern "C" {

void quick_mbx_cross_residual_initialize_(double* coords, int* nat_monomers, char** atom_names, char** monomer_names,
                                          int* regions, int* nmon, char* json_file, int* ierr) {
    if (!ierr)
        return;
    *ierr = 0;
    try {
        if (!coords || !nat_monomers || !atom_names || !monomer_names || !regions || !nmon || !json_file) {
            throw std::invalid_argument("null cross-residual initialization argument");
        }
        if (*nmon < 2)
            throw std::invalid_argument("at least two monomers are required");

        std::vector<size_t> atom_offsets(static_cast<size_t>(*nmon) + 1, 0);
        std::array<std::vector<size_t>, 3> monomer_groups;
        for (int imon = 0; imon < *nmon; ++imon) {
            if (!monomer_names[imon] || nat_monomers[imon] <= 0 || (regions[imon] != 0 && regions[imon] != 1)) {
                throw std::invalid_argument("invalid monomer size or region label");
            }
            atom_offsets[imon + 1] = atom_offsets[imon] + static_cast<size_t>(nat_monomers[imon]);
            monomer_groups[0].push_back(static_cast<size_t>(imon));
            monomer_groups[regions[imon] + 1].push_back(static_cast<size_t>(imon));
        }
        if (monomer_groups[1].empty() || monomer_groups[2].empty()) {
            throw std::invalid_argument("both QM-shadow and MB monomers are required");
        }
        for (size_t i = 0; i < atom_offsets.back(); ++i) {
            if (!atom_names[i])
                throw std::invalid_argument("null atom name");
        }

        std::unique_ptr<CrossResidual> next(new CrossResidual());
        next->natoms = atom_offsets.back();
        for (size_t group = 0; group < next->systems.size(); ++group) {
            next->systems[group] = make_subsystem(monomer_groups[group], atom_offsets, nat_monomers, coords, atom_names,
                                                  monomer_names, json_file);
        }
        residual = std::move(next);
    } catch (const std::exception& error) {
        std::cerr << "QUICK-MBX cross-residual initialization failed: " << error.what() << '\n';
        residual.reset();
        *ierr = 54;
    } catch (...) {
        std::cerr << "QUICK-MBX cross-residual initialization failed with an unknown exception\n";
        residual.reset();
        *ierr = 54;
    }
}

void quick_mbx_cross_residual_evaluate_(double* coords, int* natoms, double* energies, double* gradients, int* ierr) {
    if (!ierr)
        return;
    *ierr = 0;
    if (!coords || !natoms || !energies || !gradients || *natoms <= 0) {
        std::cerr << "QUICK-MBX cross-residual evaluation received invalid arguments\n";
        *ierr = 54;
        return;
    }
    std::fill(energies, energies + 3, 0.0);
    std::fill(gradients, gradients + 9 * static_cast<size_t>(*natoms), 0.0);
    try {
        if (!residual || static_cast<size_t>(*natoms) != residual->natoms) {
            throw std::invalid_argument("the residual system is not initialized for this atom count");
        }

        for (size_t group = 0; group < residual->systems.size(); ++group) {
            std::array<double, 3> group_energies{};
            std::array<std::vector<double>, 3> group_gradients;
            evaluate_components(residual->systems[group], coords, group_energies, group_gradients);
            const double sign = group == 0 ? 1.0 : -1.0;
            for (size_t component = 0; component < 3; ++component) {
                energies[component] += sign * group_energies[component];
                for (size_t i = 0; i < residual->systems[group].atoms.size(); ++i) {
                    const size_t global_atom = residual->systems[group].atoms[i];
                    for (size_t axis = 0; axis < 3; ++axis) {
                        gradients[component * 3 * residual->natoms + 3 * global_atom + axis] +=
                            sign * group_gradients[component][3 * i + axis];
                    }
                }
            }
        }
    } catch (const std::exception& error) {
        std::cerr << "QUICK-MBX cross-residual evaluation failed: " << error.what() << '\n';
        *ierr = 54;
    } catch (...) {
        std::cerr << "QUICK-MBX cross-residual evaluation failed with an unknown exception\n";
        *ierr = 54;
    }
}

void quick_mbx_cross_residual_finalize_() { residual.reset(); }

} // extern "C"
