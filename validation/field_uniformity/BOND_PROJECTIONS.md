# Fixed-probe water bond-direction projections

`bond_projections.py` reads accepted frozen-density exports offline. It runs no
SCF, native query, GPU kernel or molecular dynamics. It validates the repeated
O,H,H grouping and each grouped H's nearest oxygen, then associates every
original probe with its explicitly recorded nearest oxygen. Probe/O distances,
the association gap and atom indices remain in the output. A probe is **not**
relabelled as an atom or bond center.

```sh
python3 validation/field_uniformity/bond_projections.py \
  --manifest /accepted/analysis-input/manifest.json \
  --geometry validation/solvation/fixtures/water27_h2o20es.xyz \
  --native-input /accepted/native/input.dat \
  --model /exact/prepared/model.json \
  --native-report /accepted/report.json \
  --bisector --output /new/bond-projections
```

For each nearest water, axes point from O toward each H. `--bisector` adds the
normalized sum of the two O-to-H unit axes. At the **unchanged original probe**
the tool reports `E.a`, `a.G.a`, `H[a,a,a]`, and the perpendicular field norm.
The sign convention is `E=-grad(V)`, `G=grad(E)=-Hess(V)` and
`H=grad(G)=-third(V)`. These are observation-point derivatives at fixed nuclei
and density, not derivatives of a relaxed molecule. The optional G/H STF and
trace-source projections sum back to the full projections; the trace-source
parts are algebraic tensor decompositions, not separate environmental fields.

Every already supplied displaced E vector is projected onto the **same fixed
axis**. Its displacement directions remain the original finite sample set;
they are not replaced by new bond-aligned observations. Constant, linear and
quadratic Taylor predictions give signed `prediction-direct` residuals,
absolute projected errors and the full vector error norm. The projected error
cannot exceed that norm; every retained sample is checked. No continuous
directional/sphere bound, vibrational frequency or Stark-response calibration
is inferred.

`probe_water_associations.csv`, `center_axis_projections.csv`,
`sample_axis_residuals.csv` and `radius_axis_summary.csv` retain the complete
data. `summary.json` records the exact accepted-report/data/source hashes,
native/model/XYZ geometry identity, density identity, units, interpretation and
unchanged-input checks. The accepted native report must certify its fixed
density/state, and its retained hashes must match the supplied input tables.

Independent math validation runs with every application analysis. It uses an
analytic quartic scalar potential, directional finite differences at two
steps through the third derivative, proper rotations, axis-reversal parity,
malformed/nonunit axis rejection, STF/source reconstruction and projection
norm inequalities. It can also run alone:

```sh
python3 validation/field_uniformity/bond_projections.py \
  --self-test --output /new/projection-math-checks
```
