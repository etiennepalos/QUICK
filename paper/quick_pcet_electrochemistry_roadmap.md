# QUICK Electrochemical PCET Roadmap

## Scope decision

Proton-coupled electron transfer (PCET) is a strong application target for
the trajectory-resolved electrostatics and polarizable QM/MB-MM work, but it
is not another OEPROP kernel. ESP, EFIELD, and EFG describe the instantaneous
electrostatic environment; a nonadiabatic PCET rate additionally requires
state-specific proton potentials, electronic coupling, reaction free energy,
reorganization energy, and a donor-acceptor distance distribution.

The immediate development target should therefore be a reproducible bridge
from QUICK energies, forces, trajectories, and local electrostatics to an
external PCET rate-theory layer. It should not vendor a rate package or claim
that local fields alone determine a PCET rate.

## Repository survey

The following public repositories from Kai Cui were inspected at the listed
scratch revisions on 2026-07-11.

| Repository | Role | Required upstream data | Fit for QUICK | Integration note |
|---|---|---|---|---|
| [pyPCET](https://github.com/kcui1995/pyPCET), `56716e9` | Vibronically nonadiabatic PCET rates, KIEs, and nonadiabaticity analysis | Reactant/product proton potentials, `DeltaG`, `Lambda`, `Vel`, temperature, and optionally a distribution over donor-acceptor distance | Highest priority downstream consumer | GPLv3, no package metadata or automated tests in the inspected snapshot; keep external and exchange documented data files |
| [aseMECP](https://github.com/kcui1995/aseMECP), `fe96c35` | Penalty-function crossing optimization using two ASE energy/force calculators and a duplicated transferring proton | Two stable diabatic-state calculators with analytic forces | Useful only after a QUICK ASE adapter and reliable state definitions exist | Single-proton prototype; no license file in the inspected snapshot, so do not copy or vendor without permission |
| [pyPCEnT](https://github.com/kcui1995/pyPCEnT), `1e783d7` | Nonadiabatic proton-coupled energy-transfer rates | Ground/reactant/product proton potentials, donor-emission and acceptor-absorption line shapes, convolution kernel, and electronic coupling | Later photochemical/excited-state branch | GPLv3; QUICK currently lacks the excited-state and spectral machinery needed to generate its principal inputs |
| [PCET-pH-kinetic-model](https://github.com/kcui1995/PCET-pH-kinetic-model), `fddfed1` | Four-channel homogeneous or electrochemical pH-dependent kinetics with Butler-Volmer or Marcus-Gerischer options | Acid/association constants, redox potentials, and channel rate constants | Downstream interpretation after microscopic rates are available | MIT; equations are channel-specific rather than a general network engine |
| [Rate-equation-integrators](https://github.com/kcui1995/Rate-equation-integrators), `847f01f` | Small system-specific concentration kinetics | Precomputed rate and equilibrium constants | Low priority for QUICK itself | MIT; useful as an application template, not an electronic-structure dependency |

The most relevant theoretical reference is Hutchison, Cui, Zhong, and
Hammes-Schiffer, *J. Chem. Phys.* **2025**, 163, 091501,
DOI: [10.1063/5.0284337](https://doi.org/10.1063/5.0284337). It identifies
the required microscopic inputs as inner- and outer-sphere reorganization
energies, diabatic proton potential-energy profiles, electronic coupling,
reaction free energy, and the donor-acceptor distance distribution.

## Mathematical data contract

For proton donor D, acceptor A, and transferring proton H, define

\[
R_{\mathrm{DA}}=|\mathbf R_{\mathrm A}-\mathbf R_{\mathrm D}|,
\qquad
\xi_{\mathrm H}=|\mathbf R_{\mathrm H}-\mathbf R_{\mathrm D}|
-|\mathbf R_{\mathrm H}-\mathbf R_{\mathrm A}|.
\]

At fixed heavy-atom/environment coordinates \(\mathbf Q\), the electronic
structure layer must provide two state-specific proton surfaces

\[
U_{\mathrm R}(\xi_{\mathrm H};R_{\mathrm{DA}},\mathbf Q),
\qquad
U_{\mathrm P}(\xi_{\mathrm H};R_{\mathrm{DA}},\mathbf Q),
\]

and an electronic coupling
\(V_{\mathrm{el}}(\xi_{\mathrm H},R_{\mathrm{DA}},\mathbf Q)\). The
trajectory layer should also provide the vertical state-energy gap

\[
\Delta U(\mathbf Q)=U_{\mathrm P}(\mathbf Q)-U_{\mathrm R}(\mathbf Q).
\]

The pyPCET rate layer then combines proton vibrational populations and
overlaps with \(V_{\mathrm{el}}\), \(\Delta G^0\), and \(\lambda\). A useful
schematic form is

\[
k_{\mathrm{PCET}}=
\frac{2\pi}{\hbar}|V_{\mathrm{el}}|^2
\sum_{\mu\nu}P_\mu |S_{\mu\nu}|^2
\frac{\exp[-(\Delta G^0_{\mu\nu}+\lambda)^2/(4\lambda k_{\mathrm B}T)]}
{\sqrt{4\pi\lambda k_{\mathrm B}T}},
\]

followed, where needed, by averaging over \(R_{\mathrm{DA}}\), electrode
states, solvent configurations, and applied potential.

QUICK OEPROP adds complementary environment descriptors. For a unit vector
\(\widehat{\mathbf R}_{\mathrm{DA}}\) and a nonsingular probe point
\(\mathbf C\), record

\[
E_{\parallel}(\mathbf C)=
\widehat{\mathbf R}_{\mathrm{DA}}\cdot\mathbf E(\mathbf C),
\qquad
G_{\parallel\parallel}(\mathbf C)=
\widehat{\mathbf R}_{\mathrm{DA}}^{\mathsf T}
\mathbf G(\mathbf C)
\widehat{\mathbf R}_{\mathrm{DA}},
\]

and an off-nuclear potential difference

\[
\Delta V_{\mathrm{DA}}=V(\mathbf C_{\mathrm A})-V(\mathbf C_{\mathrm D}).
\]

Probe points exactly on nuclei require an explicit self-source exclusion
contract; relying on a distance cutoff to remove a singular nuclear term is
not a publishable definition.

## What QUICK can provide now

- RHF/UHF/DFT energies and analytic molecular gradients through the API.
- Restartable CPU BOMD and same-density ESP/EFIELD/EFG evaluation.
- Fixed, molecule-attached, atom-offset, and density-surface probes.
- Uniform external electric fields for energies and post-SCF properties.
- External point charges/dipoles and a polarizable QUICK-MBX energy/force
  path on the corresponding development branches.
- Separate serial and MPI reference implementations suitable for regression
  data before GPU acceleration.

These capabilities are already enough to compute structural/electrostatic
PCET descriptors and to scan simple redox/protonation states that differ by
total electron number. They are not enough for general same-electron-number
diabatic states or ab initio PCET rates.

## Missing enabling methods

1. **QUICK ASE calculator.** Provide `energy` in eV and `forces` in eV/Angstrom,
   retain state-specific input, and run each persistent state in its own
   worker process because QUICK's current API owns global calculation state.
2. **Diabatic-state control.** A simple charge/multiplicity change is useful
   for a first electrochemical benchmark, but general PCET needs constrained
   DFT or another localization/state-tracking method. A representative CDFT
   functional is
   \[
   \mathcal L[D,\eta]=E_{\mathrm{KS}}[D]
   +\eta\{\operatorname{Tr}(DW)-N^*\},
   \]
   with a documented population operator \(W\), converged multiplier
   \(\eta\), analytic gradients, and state-collapse diagnostics.
3. **Electronic coupling.** QUICK does not yet provide a validated
   \(V_{\mathrm{el}}\) route such as CDFT-CI, GMH, or FCD. Accepting an
   externally computed coupling is suitable for the first workflow test;
   an internal coupling method is a separate method-development milestone.
4. **State-specific free energies.** Obtain \(\Delta G^0\), \(\lambda\), and
   \(P(R_{\mathrm{DA}})\) from explicit sampling or validated thermodynamic
   cycles. QUICK currently has no implicit solvent, umbrella-sampling, or
   path-integral driver.
5. **Electrochemical boundary conditions.** The existing uniform field is a
   useful perturbation but is not a constant-potential electrode. A general
   external scalar potential would require
   \[
   V^{\mathrm{ext}}_{\alpha\beta}
   =\int\chi_\alpha(\mathbf r)v_{\mathrm{ext}}(\mathbf r)
   \chi_\beta(\mathbf r)\,d\mathbf r
   \]
   and consistent nuclear and analytic-gradient terms. Periodic electrodes,
   electrode DOS, and grand-canonical/constant-potential DFT remain external
   or long-term capabilities.
6. **Finite-field gradients.** Analytic gradients under the existing uniform
   external field should be completed before field-biased MECP optimization.

## Staged implementation plan

### Stage A: trajectory PCET descriptors

- Add donor, acceptor, and transferring-proton atom selections to the OEPROP
  trajectory driver.
- Emit \(R_{\mathrm{DA}}\), \(\xi_{\mathrm H}\),
  \(E_{\parallel}\), \(G_{\parallel\parallel}\), and off-nuclear
  \(\Delta V_{\mathrm{DA}}\).
- Add a source-exclusion mask or explicitly displaced probes for atom-site
  electrostatics.
- Validate rotational covariance and finite-difference derivatives of the
  projected observables.

This stage belongs naturally in Paper 3 and does not require PCET rate
theory.

### Stage B: ASE bridge and proton-surface exporter

- Implement an external `QuickCalculator` with file-based reference mode and
  a persistent-process API mode.
- Validate QUICK/ASE energies and forces for closed-shell, open-shell,
  external-field, and QUICK-MBX cases.
- Add constrained \(R_{\mathrm{DA}}\) optimizations and scans over
  \(\xi_{\mathrm H}\).
- Export pyPCET-compatible tables in Angstrom/eV with complete method,
  charge, multiplicity, field, and environment metadata.

The first wiring benchmark should reproduce a checked pyPCET example from
its supplied proton curves. The first QUICK-generated benchmark should use a
small electrochemical molecular PCET model whose reactant/product states
differ by total electron number, avoiding an unvalidated localization
assumption.

### Stage C: polarizable state surfaces and gap sampling

- Re-polarize MBX independently for reactant and product states at every
  geometry; never reuse induced dipoles across electronic states.
- Validate state-specific QUICK-MBX gradients against finite differences.
- Sample \(\Delta U\), local electrostatics, and \(R_{\mathrm{DA}}\) in both
  ensembles to estimate reorganization and structural distributions.
- Test whether \(E_{\parallel}\), \(G_{\parallel\parallel}\), and
  \(\Delta V_{\mathrm{DA}}\) predict the vertical energy gap beyond purely
  structural coordinates.

This is the most direct PCET application of Paper 2's polarizable force
machinery.

### Stage D: diabatic crossing and rates

- Implement and validate CDFT/state tracking before general use of a crossing
  optimizer.
- Use aseMECP only as an external prototype after licensing is clarified, or
  implement an independently documented crossing optimizer from the
  literature.
- Add an internal or externally validated electronic-coupling route.
- Feed \(U_{\mathrm R}\), \(U_{\mathrm P}\), \(V_{\mathrm{el}}\),
  \(\Delta G^0\), \(\lambda\), and \(P(R)\) to pyPCET and reproduce a
  published rate/KIE benchmark.
- Apply the MIT pH kinetic model only after microscopic channel rates and
  thermodynamic constants are independently justified.

### Stage E: PCEnT and excited states

Defer pyPCEnT integration until QUICK has a validated excited-state method,
state-specific gradients, transition energies/couplings, and line-shape
inputs. This is a separate photochemical project, not a near-term extension
of ground-state OEPROP.

## Paper boundaries

- **Paper 1:** keep ESP/EFIELD/EFG algorithms, validation, GPU performance,
  and on-the-fly capability. PCET may appear only as motivation.
- **Paper 2:** mention polarizable reactant/product surfaces and energy-gap
  sampling as a future application. Do not delay the QUICK-MBX methods paper
  for CDFT or PCET rates.
- **Paper 3:** add PCET-aware local descriptors and one small
  proton-transfer/electrochemical trajectory application if the data are
  available. Do not claim a rate without all pyPCET inputs.
- **Dedicated later paper:** QUICK-to-ASE diabatic surfaces, polarizable
  energy-gap sampling, and nonadiabatic electrochemical PCET rates/KIEs.

## Acceptance criteria

- QUICK direct and ASE calculator energies/forces agree at every test point.
- Reactant and product states retain the intended charge/spin/localization
  throughout scans and optimizations.
- Analytic state gradients and crossing-objective forces agree with finite
  differences.
- Proton-potential tables are smooth, reproducible, and converged with
  electronic method, basis, scan spacing, and proton grid.
- pyPCET rates reproduce a supplied reference before QUICK-generated surfaces
  are interpreted.
- Rates/KIEs are converged with proton grid, vibrational-state count,
  donor-acceptor sampling, electrode-energy integration, and environment
  sampling.
- Applied-field calculations distinguish a finite molecular field model from
  a genuine constant-potential electrode treatment.

