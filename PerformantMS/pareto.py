#!/usr/bin/env python3
"""Standalone single-k MS-equation solver Pareto benchmark.

Reproduces the appendix Pareto figure of LatticeLetter (Barker, Gladwyn,
Zell) at a single k value.  Nine ODE solvers (scipy.solve_ivp x 4,
SUNDIALS CVODE x 2, Julia OrdinaryDiffEq.jl x 3) are run over a 10-point
rtol sweep against both the full Mukhanov-Sasaki equation and its
deformed (no-metric-perturbation) variant.  The resulting wall-time vs
relative-error trade-off is rendered as a single-panel figure.

This script and the companion `ms_bench.jl` constitute a self-contained
reproduction kit: no imports from the rest of the CLInflation tree, no
historical data dependency.

Requirements:
    Python:  numpy, scipy, matplotlib, sksundae
    Julia:   1.10+, with OrdinaryDiffEq + BSplineKit + NPZ instantiated
             (`julia --project=. -e 'import Pkg; Pkg.instantiate()'`)
    LaTeX:   any TeX install with amsmath, amssymb, underscore

Usage:
    python pareto.py [--k VALUE_IN_GEV] [--out PATH]
"""
from __future__ import annotations

import argparse
import os
import shutil
import signal
import subprocess
import tempfile
import time as _time
from concurrent.futures import ProcessPoolExecutor, as_completed
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import optimize
from scipy.interpolate import UnivariateSpline, BSpline
from scipy.integrate import solve_ivp

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['text.latex.preamble'] = r'\usepackage{amsmath,amssymb}\usepackage{underscore}'
import matplotlib.pyplot as plt


Mp = 2.435e18  # reduced Planck mass in GeV

# -------------------------- pbh (USR) model --------------------------
# Inflaton potential V(phi), and the Einstein-frame field h.  Parameters
# are the same ones the paper uses for the Pareto figure.
c2, c3, phi0, lam, xi = 0.03, 0.075, 0.97, 6.6e-11, 0.10483


def V_phi(phi):
    return lam / (24 * (1 + xi * phi**2)**2) * (
        3 * phi**4 + xi**2 * phi0**4 * phi**4
        - 8 * (1 + c3) * phi0 * phi**3
        + 2 * (1 + c2) * (3 + xi * phi0**2) * phi0**2 * phi**2)


def _hofphi(phi):
    # Einstein-frame field redefinition for non-minimal coupling.
    return (np.sqrt((1 + 6 * xi) / xi)
            * np.arcsinh(phi * np.sqrt(xi * (1 + 6 * xi)))
            - np.sqrt(6) * np.arctanh(
                (np.sqrt(6) * xi * phi) / np.sqrt(1 + xi * (1 + 6 * xi) * phi**2)))


def _find_phi_for_h(h):
    for seed in np.linspace(1.0, 12.0, 100):
        sol = optimize.root(lambda p, h: _hofphi(p) - h, seed, args=(h,))
        if sol.success:
            return sol.x[0]
    return None


def V_h(h):
    return V_phi(_find_phi_for_h(h))


def h_derivatives(x, _Ne, Vh, Vh_grad):
    return [x[1], -3*x[1] + x[1]**3/2 - (3 - x[1]**2/2) * Vh_grad(x[0]) / Vh(x[0])]


# -------------------------- Background --------------------------

def build_background():
    """Return spline interpolants + (NStart, Nend, k_Ne, Ne_k).

    Mirrors the production setup in ms_benchmark.py (which itself follows
    pbh.py exactly): integrate the Einstein-frame inflaton background
    starting at h0=11 with slow-roll IC, then construct k(N_e) = a H.
    """
    print("Building V(h) interpolation...")
    h_grid = np.linspace(0, 14, 1000)
    Vh = UnivariateSpline(h_grid, [V_h(h) for h in h_grid], k=4, s=0)
    Vh_grad = Vh.derivative()

    h0  = 11
    H0  = np.sqrt(Vh(h0) / 3)
    dh0 = -Vh_grad(h0) / (3 * H0)

    print("Integrating background phi(N_e)...")
    NeArr = np.arange(0, 200, 1e-3)
    h_event = lambda Ne, x: x[0]
    h_event.terminal, h_event.direction = True, -1
    sol = solve_ivp(lambda Ne, x: h_derivatives(x, Ne, Vh, Vh_grad),
                    (NeArr[0], NeArr[-1]), [h0, dh0], t_eval=NeArr,
                    method='DOP853', atol=1e-12, rtol=1e-12, events=h_event)
    phi_arr, dphi_arr, NeArr = sol.y[0], sol.y[1], sol.t

    dphi_spl = UnivariateSpline(NeArr, dphi_arr, k=1, s=0)
    eps_arr  = dphi_arr**2 / 2
    eps_spl  = UnivariateSpline(NeArr, eps_arr, k=1, s=0)
    eta_arr  = eps_arr - 0.5 * eps_spl.derivative()(NeArr) / eps_arr
    eta_spl  = UnivariateSpline(NeArr, eta_arr, k=1, s=0)
    H_arr    = np.where(Vh(phi_arr) / (3 - eps_arr) > 0,
                        np.sqrt(Vh(phi_arr) / (3 - eps_arr)), 0)
    H_spl    = UnivariateSpline(NeArr, H_arr, k=1, s=0)

    Nend  = min(optimize.brentq(lambda Ne: eps_spl(Ne) - 1.0, 0, NeArr[-1]),
                NeArr[-1])
    NStart = 74

    def k_Ne(Ne):
        return H_spl(Ne) * np.exp(Ne - NStart) * Mp

    x_full = np.append(np.arange(0, Nend, 1e-4), Nend)
    k_full = k_Ne(x_full)
    x_mono = x_full[:np.argmax(k_full) + 1]
    k_mono = k_full[:np.argmax(k_full) + 1]
    k_spl  = UnivariateSpline(x_mono, k_mono, k=1, s=0)

    def Ne_k(ks):
        return np.array([optimize.brentq(lambda Ne: k_spl(Ne) - kk, 0, x_mono[-1])
                         for kk in ks])

    print(f"  Nend = {Nend:.2f}, NStart = {NStart}, "
          f"k_max = {k_mono[-1]/Mp:.3e} Mp")
    return eps_spl, dphi_spl, eta_spl, H_spl, k_Ne, Ne_k, NStart, Nend


# -------------------------- Spline IO --------------------------

def _spline_to_arrays(spl):
    t, c, k = spl._eval_args
    return (np.array(t), np.array(c), int(k))


def _spline_from_arrays(t, c, k):
    return BSpline(t, c, k)


# -------------------------- Solver wrappers --------------------------

def _solve_ivp_endpoint(k, Ni, Nf, k_Ne_fn, eps_spl, dphi_spl,
                        Foo, NStart, method, rtol, atol):
    """scipy.solve_ivp -> final P(k).  Two integrations: real + imag rotation."""
    ki = k_Ne_fn(Ni)
    scaling = 1e4
    tspan = [Ni, Nf]

    def f(Ne, y):
        u, up = y
        eps = eps_spl(Ne)
        foo = Foo(Ne, k)
        return [up, -(1 - eps) * up - foo * u]

    R0 = np.array([1.0 / scaling, 0.0])
    I0 = np.array([0.0, -k / ki / scaling])

    sol_R = solve_ivp(f, tspan, R0, method=method, rtol=rtol, atol=atol,
                      dense_output=False)
    sol_I = solve_ivp(f, tspan, I0, method=method, rtol=rtol, atol=atol,
                      dense_output=False)
    return _power_spectrum(k, Nf, NStart, dphi_spl,
                           sol_R.y[0, -1], sol_I.y[0, -1], scaling)


def _cvode_endpoint(k, Ni, Nf, k_Ne_fn, eps_spl, dphi_spl,
                    Foo, NStart, method, rtol, atol):
    """SUNDIALS CVODE (via sksundae) -> final P(k).  Two integrations."""
    from sksundae.cvode import CVODE
    ki = k_Ne_fn(Ni)
    scaling = 1e4
    tspan = np.array([Ni, Nf])

    def f(Ne, y, yp):
        u, up = y
        eps = eps_spl(Ne)
        foo = Foo(Ne, k)
        yp[0] = up
        yp[1] = -(1 - eps) * up - foo * u

    R0 = np.array([1.0 / scaling, 0.0])
    I0 = np.array([0.0, -k / ki / scaling])
    sR_end = CVODE(f, method=method, rtol=rtol, atol=atol, max_num_steps=500000).solve(tspan, R0).y[-1, 0]
    sI_end = CVODE(f, method=method, rtol=rtol, atol=atol, max_num_steps=500000).solve(tspan, I0).y[-1, 0]
    return _power_spectrum(k, Nf, NStart, dphi_spl, sR_end, sI_end, scaling)


def _power_spectrum(k, Nf, NStart, dphi_spl, R_end, I_end, scaling):
    u_sq = scaling**2 / (2 * k) * (R_end**2 + I_end**2)
    a_Nf = np.exp(Nf - NStart)
    z    = -a_Nf * dphi_spl(Nf) * Mp
    if z == 0:
        z = 1e-30
    return (k**3) * u_sq / (2 * np.pi**2 * z**2)


# -------------------------- Python pool worker --------------------------

class _Timeout(Exception): pass
def _alarm(signum, frame): raise _Timeout()


def _python_worker(args):
    """Sweep all rtols for one (method, k) over both equations.

    Loosest rtol first; once a solve times out at the per-rtol limit, all
    tighter rtols are auto-marked as failures (they would be slower).
    """
    RTOL_TIMEOUT = 1  # seconds
    (k, Ni, Nf, NStart, method, rtols, atol_fac,
     eps_arr, dphi_arr, eta_arr, hub_arr) = args

    eps_spl = _spline_from_arrays(*eps_arr)
    dphi_spl = _spline_from_arrays(*dphi_arr)
    eta_spl = _spline_from_arrays(*eta_arr)
    hub_spl = _spline_from_arrays(*hub_arr)
    eps_d = eps_spl.derivative()
    eta_d = eta_spl.derivative()

    def k_Ne(Ne):
        return hub_spl(Ne) * np.exp(Ne - NStart) * Mp

    def _foo(Ne, k_val, mod=False):
        kc = k_Ne(Ne)
        eps, et = float(eps_spl(Ne)), float(eta_spl(Ne))
        epsp, etp = float(eps_d(Ne)), float(eta_d(Ne))
        base = (k_val / kc)**2 + (1 + eps - et)*(et - 2) - (epsp - etp)
        return base + 2*eps*(3 + eps - 2*et) if mod else base

    sweep = sorted(rtols, reverse=True)
    results = []
    old = signal.signal(signal.SIGALRM, _alarm)
    try:
        for eq, mod in [("MS", False), ("MS_mod", True)]:
            for i, rt in enumerate(sweep):
                atol = rt * atol_fac
                signal.alarm(RTOL_TIMEOUT)
                try:
                    t0 = _time.perf_counter()
                    Foo_fn = lambda Ne, k_val, _mod=mod: _foo(Ne, k_val, _mod)
                    if method.startswith("CVODE_"):
                        P = _cvode_endpoint(k, Ni, Nf, k_Ne, eps_spl, dphi_spl,
                                            Foo_fn, NStart,
                                            method.split("_", 1)[1], rt, atol)
                    else:
                        P = _solve_ivp_endpoint(k, Ni, Nf, k_Ne, eps_spl, dphi_spl,
                                                Foo_fn, NStart, method, rt, atol)
                    dt = _time.perf_counter() - t0
                    ok = not (np.isnan(P) or np.isinf(P) or P <= 0)
                    results.append((method, rt, float(P) if ok else np.nan,
                                    dt, ok, eq))
                except _Timeout:
                    dt = _time.perf_counter() - t0
                    results.append((method, rt, np.nan, dt, False, eq))
                    # Skip remaining (tighter) rtols.
                    for rt_rem in sweep[i+1:]:
                        results.append((method, rt_rem, np.nan, 0.0, False, eq))
                    break
                finally:
                    signal.alarm(0)
    finally:
        signal.signal(signal.SIGALRM, old)
    return results


# -------------------------- Julia subprocess --------------------------

def _julia_worker(args):
    """One Julia subprocess handles all (Julia method x rtol x equation)
    for a single k.  Communicates via two NPZ files.
    """
    (k, Ni, Nf, NStart, julia_methods, rtols, atol_fac,
     eps_arr, dphi_arr, eta_arr, hub_arr, sysimage) = args

    julia_bin = shutil.which("julia")
    if julia_bin is None:
        return []

    here = Path(__file__).resolve().parent
    bench_jl = here / "ms_bench.jl"
    jl_names = [m.split("_", 1)[1] for m in julia_methods]  # strip "JL_"
    methods_str = ",".join(jl_names)

    with tempfile.TemporaryDirectory() as td:
        inp = os.path.join(td, "in.npz")
        out = os.path.join(td, "out.npz")
        np.savez(inp,
                 eps_t=eps_arr[0],  eps_c=eps_arr[1],  eps_k=np.float64(eps_arr[2]),
                 dphi_t=dphi_arr[0], dphi_c=dphi_arr[1], dphi_k=np.float64(dphi_arr[2]),
                 eta_t=eta_arr[0],  eta_c=eta_arr[1],  eta_k=np.float64(eta_arr[2]),
                 hub_t=hub_arr[0],  hub_c=hub_arr[1],  hub_k=np.float64(hub_arr[2]),
                 k_phys=np.float64(k), Ni=np.float64(Ni), Nf=np.float64(Nf),
                 NStart=np.float64(NStart),
                 methods_str=np.frombuffer(methods_str.encode("ascii"), dtype=np.uint8),
                 rtols=np.array(rtols, dtype=np.float64),
                 atol_factor=np.float64(atol_fac))

        cmd = [julia_bin]
        if sysimage and os.path.exists(sysimage):
            cmd.append(f"-J{sysimage}")
        cmd += [f"--project={here}", str(bench_jl), inp, out]
        env = os.environ.copy()
        env["OPENBLAS_NUM_THREADS"] = "1"
        env["MKL_NUM_THREADS"] = "1"

        try:
            r = subprocess.run(cmd, capture_output=True, text=True,
                               timeout=300, env=env)
            if r.returncode != 0:
                return []
            d = np.load(out)
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return []

    results = []
    for eq, suffix in [("MS", ""), ("MS_mod", "_mod")]:
        Pf, wt, sc = d[f"P_finals{suffix}"], d[f"wall_times{suffix}"], d[f"success{suffix}"]
        idx = 0
        for m in julia_methods:
            for rt in rtols:
                P  = Pf[idx]
                ok = sc[idx] > 0.5
                results.append((m, rt, P if ok else np.nan, wt[idx], ok, eq))
                idx += 1
    return results


# -------------------------- Ground truth --------------------------

def compute_ground_truth(results):
    """Consensus P_truth = mean of each Julia method's tightest-rtol value.

    Python solvers are explicitly excluded from contributing to the
    reference because they ran only at default tolerance and would
    contaminate the truth with their inherent looseness.  If no Julia
    results survived, fall back to averaging all methods' tightest-rtol
    values (graceful degradation).
    """
    jl, alt = defaultdict(list), defaultdict(list)
    for m, rt, P, dt, ok in results:
        if not ok or np.isnan(P) or np.isinf(P) or P <= 0:
            continue
        alt[m].append((rt, P))
        if m.startswith("JL_"):
            jl[m].append((rt, P))
    chosen = jl if jl else alt
    tightest = [min(v, key=lambda r: r[0])[1] for v in chosen.values()]
    return float(np.mean(tightest)) if tightest else None


# -------------------------- Plotting --------------------------

def _display_name(m):
    if m.startswith("JL_"):
        return r"$\mathrm{Julia\_" + m[3:] + r"}$"
    if m.startswith("CVODE_"):
        return r"$\mathrm{SUNDIALS\_" + m[6:] + r"}$"
    return r"$\mathrm{Python\_" + m + r"}$"


def plot_pareto(k, results, P_ms, P_mod, methods, out_pdf, title=None):
    by_me = defaultdict(list)
    for entry in results:
        m, rt, P, dt, ok = entry[:5]
        eq = entry[5]
        if ok and not (np.isnan(P) or np.isinf(P) or P <= 0):
            by_me[(m, eq)].append((rt, P, dt))

    colors  = {m: f"C{i % 10}" for i, m in enumerate(methods)}
    markers = {m: ['o','s','^','D','v','P','X','h','*'][i % 9] for i, m in enumerate(methods)}
    style = {"MS": dict(ls='-', fillstyle='full'),
             "MS_mod": dict(ls='--', fillstyle='none')}

    fig, ax = plt.subplots(figsize=(5, 4.5))
    for m in methods:
        for eq in ("MS", "MS_mod"):
            truth = P_ms if eq == "MS" else P_mod
            data = by_me.get((m, eq), [])
            if truth is None or not data:
                continue
            data.sort(key=lambda x: x[0], reverse=True)
            rts, Ps, ts = zip(*data)
            err = np.abs(np.array(Ps) - truth) / truth
            err = np.where(err == 0, 1e-16, err)
            ax.plot(err, ts,
                    marker=markers[m], color=colors[m],
                    label=_display_name(m) if eq == "MS" else None,
                    markersize=5, ls=style[eq]['ls'], fillstyle=style[eq]['fillstyle'])

    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlabel(r'$|\mathcal{P}_{\mathcal{R}}(k;\, N_e^{\mathrm{end}}) - \mathcal{P}_{\mathcal{R}}^{\mathrm{truth}}| / \mathcal{P}_{\mathcal{R}}^{\mathrm{truth}}$')
    ax.set_ylabel(r'$\mathrm{Wall\ time\ (s)}$')
    if title:
        ax.set_title(title)
    else:
        ax.set_title(rf'$k = {k/Mp:.3e}\ M_\mathrm{{pl}}$')
    ax.legend(fontsize=7, loc='best', ncol=2)
    fig.tight_layout()
    fig.savefig(out_pdf)
    print(f"Wrote {out_pdf}")


# -------------------------- Main --------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--k', type=float, default=1.004e10,
                        help='Single k value in GeV (default 1.004e10)')
    parser.add_argument('--out', type=str,
                        default=str(Path(__file__).resolve().parent / "pareto.pdf"))
    parser.add_argument('--sysimage', type=str, default=None,
                        help='Optional Julia sysimage.so for fast startup')
    parser.add_argument('--title', type=str, default=None,
                        help='Plot title (typically the hardware string). '
                             'Defaults to "$k = ...\\,M_pl$".')
    args = parser.parse_args()

    (eps_spl, dphi_spl, eta_spl, H_spl,
     k_Ne, Ne_k, NStart, Nend) = build_background()

    eps_arr  = _spline_to_arrays(eps_spl)
    dphi_arr = _spline_to_arrays(dphi_spl)
    eta_arr  = _spline_to_arrays(eta_spl)
    hub_arr  = _spline_to_arrays(H_spl)

    # Per-k integration window: start at 1/20 horizon (matches pbh.py);
    # stop at horizon-cross + 7 e-folds (matches the current production
    # spectrum-generation convention).
    k_start = args.k / 20
    Ni = 0.0 if k_start <= k_Ne(0.0) else float(Ne_k([k_start])[0])
    Nf = min(Nend, float(Ne_k([args.k])[0]) + 7.0)
    print(f"k = {args.k:.3e} GeV ({args.k/Mp:.3e} Mp), "
          f"Ni = {Ni:.2f}, Nf = {Nf:.2f}")

    python_methods = ["Radau", "BDF", "LSODA", "DOP853",
                      "CVODE_BDF", "CVODE_Adams"]
    julia_methods  = ["JL_RadauIIA5", "JL_QNDF", "JL_Rodas5P"]
    all_methods    = python_methods + julia_methods
    rtols          = list(np.logspace(-3, -12, 10))
    atol_fac       = 1e-3

    print(f"Running {len(python_methods)} Python + {len(julia_methods)} Julia methods "
          f"x {len(rtols)} rtols ...")

    n_cpu = os.cpu_count() or 4
    results = []
    with ProcessPoolExecutor(max_workers=n_cpu) as py_pool:
        py_futs = [py_pool.submit(_python_worker,
                                  (args.k, Ni, Nf, NStart, m, rtols, atol_fac,
                                   eps_arr, dphi_arr, eta_arr, hub_arr))
                   for m in python_methods]
        with ProcessPoolExecutor(max_workers=1) as jl_pool:
            jl_fut = jl_pool.submit(_julia_worker,
                                    (args.k, Ni, Nf, NStart,
                                     julia_methods, rtols, atol_fac,
                                     eps_arr, dphi_arr, eta_arr, hub_arr,
                                     args.sysimage))
            for f in as_completed(py_futs + [jl_fut]):
                results.extend(f.result())

    # Split MS / MS_mod for per-equation ground truth.
    ms     = [(m, rt, P, dt, ok) for m, rt, P, dt, ok, eq in results if eq == "MS"]
    ms_mod = [(m, rt, P, dt, ok) for m, rt, P, dt, ok, eq in results if eq == "MS_mod"]
    P_truth_ms  = compute_ground_truth(ms)     if ms     else None
    P_truth_mod = compute_ground_truth(ms_mod) if ms_mod else None

    n_ok = sum(1 for r in results if r[4])
    print(f"{n_ok}/{len(results)} ok.  "
          f"P_truth_MS = {P_truth_ms!r}, P_truth_mod = {P_truth_mod!r}")

    plot_pareto(args.k, results, P_truth_ms, P_truth_mod, all_methods, args.out,
                title=args.title)


if __name__ == '__main__':
    main()
