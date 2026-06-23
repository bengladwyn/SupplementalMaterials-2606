# SupplementalMaterials-2606.14853
Supplemental material for *The
limits of lattice inflation: a cautionary tale* (Barker, Gladwyn, Zell), http://arxiv.org/abs/2606.14853

### Models

Three benchmark models are used throughout, named consistently across `Analytics`, `CLmodels`, `CLPlotting`, `CLOutput` and `ModifiedCLOutput`:

- `lphi4` - slow-roll (SR) case
- `pbh` - ultra-slow roll (USR) model
- `tanh2` - preheating/reheating example.

### Contents

- `Analytics` - Scripts that calculate the analytic background/perturbation evolution for each model and produce the input parameters fed into CosmoLattice.
- `CLmodels` - The CosmoLattice model (`.h`) and parameter (`parameter-files/*.in`) files used to run the lattice simulations.
- `CLPlotting` - Notebooks that plot the CosmoLattice output (`CLOutput`, `ModifiedCLOutput`) against the analytic expectations from `Analytics`.
- `CLOutput` - Raw CosmoLattice evolution output for the three models.
- `ModifiedCLOutput` - CosmoLattice evolution output for the three models including 1st-order metric perturbations.
- `PerformantMS` - Standalone reproduction kit for the Mukhanov-Sasaki solver benchmark of the appendix. See [PerformantMS/README.md](PerformantMS/README.md).

### CosmoLattice with 1st order perturbations

The `ModifiedCLOutput` data was produced with a modified fork of CosmoLattice that implements the 1st-order metric perturbations in the evolution of the fields. That fork is still being prepared for release; in the meantime, its output is already included here under `ModifiedCLOutput`.