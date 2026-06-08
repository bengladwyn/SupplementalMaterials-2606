#!/usr/bin/env julia
#
# ms_bench.jl - Minimal Julia driver for the LatticeLetter appendix.
#
# Runs three OrdinaryDiffEq.jl stiff solvers (RadauIIA5, QNDF, Rodas5P) at
# a single k value, sweeping a list of rtols against both the full MS
# equation and its deformed variant.  Communicates with the Python caller
# via two NPZ files (input + output).
#
# Usage:  julia --project=. ms_bench.jl input.npz output.npz
#
# Input keys:
#   eps_t/eps_c/eps_k, dphi_t/dphi_c/dphi_k,
#   eta_t/eta_c/eta_k, hub_t/hub_c/hub_k         - scipy BSpline tck arrays
#   k_phys, Ni, Nf, NStart                       - per-mode integration window
#   methods_str (uint8 array of "name,name,...") - solvers to test
#   rtols (Float64 array), atol_factor (Float64) - tolerance sweep
#
# Output keys (per equation): P_finals{_mod}, wall_times{_mod}, success{_mod}.

using NPZ
using BSplineKit
using OrdinaryDiffEq

const Mp = 2.435e18

function reconstruct_spline(t::Vector{Float64}, c::Vector{Float64}, k::Int)
    # scipy BSpline(t, c, k) stores the full knot vector with boundary
    # repeated k+1 times.  BSplineKit's BSplineBasis(order, breakpoints)
    # expects unique breakpoints; it adds its own augmentation.
    order = k + 1
    breakpoints = unique(t)
    B = BSplineBasis(BSplineOrder(order), breakpoints)
    n = length(B)
    return Spline(B, c[1:n])
end

function get_solver(name::AbstractString)
    solvers = Dict(
        "RadauIIA5" => RadauIIA5(),
        "QNDF"      => QNDF(),
        "Rodas5P"   => Rodas5P(),
    )
    return solvers[name]
end

function compute_P(k_phys, Nf, NStart, dphi_spl, scaling, R_end, I_end)
    u_squared = (scaling^2) / (2 * k_phys) * (R_end^2 + I_end^2)
    a_Nf = exp(Nf - NStart)
    z = -a_Nf * dphi_spl(Nf) * Mp
    return (k_phys^3) * u_squared / (2 * pi^2 * z^2)
end

function run_one_eq(f!, method_names, rtols, atol_fac,
                    R0, I0, tspan,
                    k_phys, Nf, NStart, dphi_spl, scaling)
    n_jobs = length(method_names) * length(rtols)
    P_finals   = Vector{Float64}(undef, n_jobs)
    wall_times = Vector{Float64}(undef, n_jobs)
    success    = Vector{Float64}(undef, n_jobs)

    idx = 1
    for mname in method_names
        solver = get_solver(strip(String(mname)))
        # Per-solver JIT warmup (cheap rtol so the timed loop is hot).
        solve(ODEProblem(f!, R0, tspan), solver; reltol=1e-3, abstol=1e-6,
              maxiters=5000, save_everystep=false)

        for rt in rtols
            at = rt * atol_fac
            try
                dt_wall = @elapsed sol_R = solve(ODEProblem(f!, R0, tspan), solver;
                    reltol=rt, abstol=at, maxiters=1_000_000, save_everystep=false)
                dt_wall += @elapsed sol_I = solve(ODEProblem(f!, I0, tspan), solver;
                    reltol=rt, abstol=at, maxiters=1_000_000, save_everystep=false)

                if sol_R.retcode != ReturnCode.Success || sol_I.retcode != ReturnCode.Success
                    P_finals[idx] = NaN; wall_times[idx] = dt_wall; success[idx] = 0.0
                else
                    P = compute_P(k_phys, Nf, NStart, dphi_spl, scaling,
                                  sol_R.u[end][1], sol_I.u[end][1])
                    if isnan(P) || isinf(P) || P <= 0
                        P_finals[idx] = NaN; wall_times[idx] = dt_wall; success[idx] = 0.0
                    else
                        P_finals[idx] = P; wall_times[idx] = dt_wall; success[idx] = 1.0
                    end
                end
            catch
                P_finals[idx] = NaN; wall_times[idx] = 0.0; success[idx] = 0.0
            end
            idx += 1
        end
    end
    return P_finals, wall_times, success
end

function main()
    length(ARGS) == 2 || (println(stderr, "Usage: julia ms_bench.jl input.npz output.npz"); exit(1))
    data = npzread(ARGS[1])

    eps_spl  = reconstruct_spline(data["eps_t"],  data["eps_c"],  Int(data["eps_k"]))
    dphi_spl = reconstruct_spline(data["dphi_t"], data["dphi_c"], Int(data["dphi_k"]))
    eta_spl  = reconstruct_spline(data["eta_t"],  data["eta_c"],  Int(data["eta_k"]))
    hub_spl  = reconstruct_spline(data["hub_t"],  data["hub_c"],  Int(data["hub_k"]))
    eps_deriv = Derivative(1) * eps_spl
    eta_deriv = Derivative(1) * eta_spl

    k_phys   = Float64(data["k_phys"])
    Ni, Nf   = Float64(data["Ni"]),  Float64(data["Nf"])
    NStart   = Float64(data["NStart"])
    rtols    = Float64.(data["rtols"])
    atol_fac = Float64(data["atol_factor"])

    method_names = split(String(UInt8.(data["methods_str"])), ",")

    scaling = 1e4
    ki = hub_spl(Ni) * exp(Ni - NStart) * Mp
    R0 = [1.0 / scaling, 0.0]
    I0 = [0.0, -k_phys / ki / scaling]

    k_Ne(Ne) = hub_spl(Ne) * exp(Ne - NStart) * Mp

    function f!(du, u, p, Ne)
        u_, up_ = u
        ev, etv = eps_spl(Ne), eta_spl(Ne)
        epsp, etp = eps_deriv(Ne), eta_deriv(Ne)
        kc = k_Ne(Ne)
        foo = (k_phys/kc)^2 + (1+ev-etv)*(etv-2) - (epsp - etp)
        du[1] = up_;  du[2] = -(1 - ev)*up_ - foo*u_;  nothing
    end
    function f_mod!(du, u, p, Ne)
        u_, up_ = u
        ev, etv = eps_spl(Ne), eta_spl(Ne)
        epsp, etp = eps_deriv(Ne), eta_deriv(Ne)
        kc = k_Ne(Ne)
        foo = (k_phys/kc)^2 + (1+ev-etv)*(etv-2) - (epsp - etp) + 2*ev*(3 + ev - 2*etv)
        du[1] = up_;  du[2] = -(1 - ev)*up_ - foo*u_;  nothing
    end

    tspan = (Ni, Nf)

    # Cheap JIT warmup so timing loops are not inflated by compilation.
    solve(ODEProblem(f!,     R0, tspan), RadauIIA5(); reltol=1e-3, abstol=1e-6, maxiters=5000, save_everystep=false)
    solve(ODEProblem(f_mod!, R0, tspan), RadauIIA5(); reltol=1e-3, abstol=1e-6, maxiters=5000, save_everystep=false)

    Pf,    wt,    sc    = run_one_eq(f!,     method_names, rtols, atol_fac,
                                     R0, I0, tspan, k_phys, Nf, NStart, dphi_spl, scaling)
    Pfmod, wtmod, scmod = run_one_eq(f_mod!, method_names, rtols, atol_fac,
                                     R0, I0, tspan, k_phys, Nf, NStart, dphi_spl, scaling)

    npzwrite(ARGS[2], Dict(
        "P_finals" => Pf, "wall_times" => wt, "success" => sc,
        "P_finals_mod" => Pfmod, "wall_times_mod" => wtmod, "success_mod" => scmod,
    ))
end

main()
