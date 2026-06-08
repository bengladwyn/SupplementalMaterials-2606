# PerformantMS

Standalone reproduction kit for the Pareto figure in Appendix E of *The
limits of lattice inflation: a cautionary tale* (Barker, Gladwyn, Zell).

The kit runs **nine** ODE solvers against the Mukhanov-Sasaki equation
in the USR pbh background at a single comoving k mode, sweeping ten
tolerance levels per solver per equation, and renders the resulting
wall-time vs relative-error trade-off as a single-panel plot.

## Contents

| file | role |
|---|---|
| `pareto.py` | orchestrator + Python scipy/SUNDIALS workers + Pareto rendering |
| `ms_bench.jl` | minimal Julia driver for the three OrdinaryDiffEq.jl methods |
| `Project.toml`, `Manifest.toml` | Julia environment lock |
| `README.md` | this file |

No imports from elsewhere in the parent repository.  The kit only
depends on the Python / Julia / LaTeX stacks listed below.

## Requirements

**Python**
- 3.10+
- `numpy`, `scipy`, `matplotlib`, `sksundae`

**Julia**
- 1.10+
- packages: `OrdinaryDiffEq`, `BSplineKit`, `NPZ` (instantiated by
  running once: `julia --project=. -e 'import Pkg; Pkg.instantiate()'`)

**LaTeX**
- any TeX install with `amsmath`, `amssymb`, `underscore`

## Usage

```bash
# One-time: install Julia packages
julia --project=. -e 'import Pkg; Pkg.instantiate()'

# Reproduce the appendix figure (default k = 1.004e10 GeV ~ 4.1e-9 Mp).
python pareto.py
```

Wall time on a 128-core workstation is ~30 s.  The script writes
`pareto.pdf` next to itself.

### Options

```bash
python pareto.py --k <GeV>          # pick a different k mode
python pareto.py --out <path>       # output PDF path
python pareto.py --sysimage <path>  # use a Julia sysimage.so for faster
                                    # startup (optional, ~3 s savings)
```

## What the plot shows

Each curve = one (solver, MS variant) at ten rtol values from `1e-3`
(loose) to `1e-12` (tight).  Markers are joined for visual continuity;
each marker is one ODE solve.  Style convention:

- **solid line, filled marker** — full MS equation (with metric perts)
- **dashed line, open marker** — deformed MS (without metric perts)

The ground-truth value used to compute the relative error on the x-axis
is the mean of the three Julia methods' tightest-rtol results
(`compute_ground_truth` in `pareto.py`).  Python and SUNDIALS solvers do
not contribute to the reference — they are samples to be measured
against it.

The visual Pareto frontier is the lower-left envelope of all curves: at
any chosen relative error, the curve closest to the bottom is the
cheapest method that delivers that accuracy.  In this benchmark the
front is owned by the Julia stiff-RK family (`RadauIIA5`, `QNDF`,
`Rodas5P`), which run roughly 10x-100x faster than the scipy/SUNDIALS
solvers at any matched accuracy.

## Notes

`pareto.py` and `ms_bench.jl` are stripped-down forks of the production
files `MinimalPowerSpectra/ms_benchmark/{ms_benchmark.py, ms_solver.py,
ms_benchmark.jl}` in the upstream repository.  The production code
additionally handles multi-k spectrum extraction, time-series capture,
multiple plot panels, sysimage detection, spline round-trip validation,
and HTCondor / systemd-scope wrappers for cluster use.  None of that is
needed to reproduce the appendix figure; this kit drops it all.
