import matplotlib.pyplot as plt
import matplotlib
from scipy import optimize
import numpy as np
from scipy.interpolate import UnivariateSpline
from scipy.integrate import solve_ivp, cumulative_trapezoid
from dask.distributed import LocalCluster, Client, progress
from pathlib import Path
import os
import sys

matplotlib.use("Agg")
matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.rcParams['font.family'] = 'serif'

def V_phi(phi):
    return lam0 * phi**4 / 4

def dV_dphi(phi):
    return lam0 * phi**3

def epsilon_SR(phi):
    return 0.5 * (dV_dphi(phi) / V_phi(phi))**2

# Define an event function to stop when phi crosses zero
def phi_crosses_zero(Ne, x):
    return x[0]  # This triggers when phi crosses zero

def phi_derivatives(x, Ne, V_phi, dV_dphi):
    return [x[1], -3 * x[1] + x[1]**3 / 2 - (3 - (x[1]**2 / 2)) * dV_dphi(x[0])/V_phi(x[0])]

def V2(phi,q):
    return lam0 * (phi**4 / 4 + q**4 / 4)

def dV2_dphi(phi,q):
    return lam0 * phi**3

def dV2_dq(phi,q):
    return lam0 * q**3

def twofield_derivatives(Ne, x):
    y, dy, q, dq = x

    # precompute quantities
    U = V2(y, q)
    Uy = dV2_dphi(y, q)
    Uq = dV2_dq(y, q)

    dy2 = dy**2
    dq2 = dq**2

    pref = 3 - 0.5*(dy2 + dq2)

    # --- y'' equation ---
    ydd = (
        -3*dy
        + 0.5*(dy2 + dq2)*(dy)
        - pref*( Uy/U - (1/6)*(Uq/U)*dy*dq )
    )

    # --- q'' equation ---
    qdd = (
        -3*dq
        + 0.5*(dy2 + dq2)*(dq)
        - pref*( -(1/6)*(Uy/U)*dy*dq + Uq/U )
    )

    return [dy, ydd, dq, qdd]

def singlefield_plus_rad_derivs(Ne, state):
    y, dy, rho_rad = state

    # keep rho_rad non-negative for safety
    rho_rad = max(rho_rad, 0.0)

    # compute V and its derivative
    Vy = dV_dphi(y)
    Vy_val = V_phi(y)

    # compute H^2 from Friedmann including rho_rad
    H2 = (2.0 * Vy_val + 2.0 * rho_rad) / (6.0 - dy**2)
    H = np.sqrt(np.maximum(H2, 0.0))

    # field equation in N_e time
    ddy = - (3.0 - 0.5 * dy**2) * dy - Vy / (H**2)

    # rho_rad evolution
    drho_rad = -4.0 * rho_rad

    return [dy, ddy, drho_rad]



def solve_ms_for_k(k, Ni, Nf, k_Ne, epsilon_Ne_interp, dphi_dNe_interp, Foo, NStart):
    """Solve MS equation for a single k value. Returns (k, P)."""

    #print(f"Starting MS solver for k={k:.2e}")

    Mp= 2.435e18 
    ki = k_Ne(Ni)

    # Scaling for numerical stability
    scaling = 1e4

    # MS equation: f'' + (1 - epsilon) f' + Foo * f = 0
    def ms_ode(Ne, y):
        f, fp = y
        eps = epsilon_Ne_interp(Ne)
        foo = Foo(Ne, k)
        fpp = -(1 - eps) * fp - foo * f
        return [fp, fpp]
    

    # Original (unrotated) real and imaginary initial vectors
    R0 = np.array([1.0 / scaling, 0.0])              # real part: f, f'
    I0 = np.array([0.0, -k / ki / scaling])         # imag part: f, f'

    # Solve for rotated "real" part (corresponds to real part of complex solution)
    
    sol_real = solve_ivp(
        ms_ode,
        [Ni, Nf],
        R0,
        method="Radau",
        rtol=1e-8,
        atol=1e-10
    )
    R_final = sol_real.y[0, -1]

    # Solve for rotated "imaginary" part (corresponds to imag part of complex solution)
    
    sol_imag = solve_ivp(
        ms_ode,
        [Ni, Nf],
        I0,
        method="Radau",
        rtol=1e-8,
        atol=1e-10
    )
    I_final = sol_imag.y[0, -1]
    
    # Compute |u|^2
    u_squared = (scaling ** 2) / (2 * k) * (R_final ** 2 + I_final ** 2)

    # Scale factor: a(Ne) = (0.05 / HMpc(NCMB)) * exp(Ne - NCMB)
    a_Nf = np.exp(Nf - NStart)

    # z = -a * dphi/dNe * Mp * 1.56e38
    z = -a_Nf * dphi_dNe_interp(Nf) * Mp

    # Power spectrum: P = k^3 * |u|^2 / (2 * pi^2 * z^2)
    P = (k ** 3) * u_squared / (2 * np.pi ** 2 * z ** 2)
    P_phi = P * dphi_dNe_interp(Nf) ** 2 

    return (k, Nf, P, P_phi)

def solve_ms_for_k_timeseries(k, Ni, Nf, k_Ne, epsilon_Ne_interp, dphi_dNe_interp, Foo, NStart,
                              t_steps=2000, method="Radau", max_step=None):
    """Return full time-series of real and imag mode components f(Ne) and power P(Ne)."""
    Mp = 2.435e18
    ki = k_Ne(Ni)
    scaling = 1e4

    def ms_ode(Ne, y):
        f, fp = y
        eps = epsilon_Ne_interp(Ne)
        foo = Foo(Ne, k)
        fpp = -(1 - eps) * fp - foo * f
        return [fp, fpp]

    # Bunch-Davies-like initial vectors (real & imag)
    R0 = np.array([1.0 / scaling, 0.0])
    I0 = np.array([0.0, -k / ki / scaling])

    t_eval = np.linspace(Ni, Nf, t_steps)

    kwargs = dict(method=method, t_eval=t_eval) # ,rtol = 1e-8

    sol_real = solve_ivp(ms_ode, [Ni, Nf], R0, **kwargs)
    sol_imag = solve_ivp(ms_ode, [Ni, Nf], I0, **kwargs)

    Ne_arr = sol_real.t
    R_arr = sol_real.y[0]   # real part f(Ne)
    I_arr = sol_imag.y[0]   # imag part f(Ne)

    # Compute |u|^2 at each Ne
    u2_arr = (scaling**2) / (2 * k) * (R_arr**2 + I_arr**2)

    # Compute a(Ne), z(Ne) and P(Ne)
    a_arr = np.exp(Ne_arr - NStart)                      # a(N_e) = exp(N_e - NStart)
    dphi_dNe_arr = dphi_dNe_interp(Ne_arr)
    z_arr = - a_arr * dphi_dNe_arr * Mp                 # z(N_e)
    # Guard against zeros just in case:
    z_arr = np.where(z_arr == 0.0, 1e-30, z_arr)

    P_arr = (k**3) * u2_arr / (2 * np.pi**2 * z_arr**2)
    P_phi_arr = P_arr * dphi_dNe_arr**2

    # final P (same as previous P_final but explicit)
    P_final = P_arr[-1]
    P_phi_final = P_phi_arr[-1]

    return {
        "k": k,
        "Ne": Ne_arr,
        "R": R_arr,
        "I": I_arr,
        "u2": u2_arr,
        "a": a_arr,
        "z": z_arr,
        "dphi_dNe": dphi_dNe_arr,
        "P": P_arr,
        "P_phi": P_phi_arr,
        "P_final": P_final,
        "P_phi_final": P_phi_final,
    }


# --- MS job submission ---

def make_dask_client(nodes=16, batch = False):
    if batch:
        from dask_jobqueue.htcondor import HTCondorCluster
        cluster = HTCondorCluster(
            log_directory=Path().cwd() / ".condor_logs",
            cores=1,
            memory="16GB",
            disk="4GB",
            job_extra_directives={
                "stream_output": True,
                "stream_error": True,
                "should_transfer_files": True,
                "transfer_input_files": "lphi4Inflation.py",
            },
        )
        cluster.scale(jobs=nodes)
        client = Client(cluster)
        return client, cluster

    cluster = LocalCluster(
        processes=True,
        threads_per_worker=1,
        memory_limit="2GB",
        n_workers=8,
    )
    client = Client(cluster)
    return client, cluster

def MS_mode(Nf_min, Nf_max, k_val, M, k_Ne, Ne_k, epsilon_Ne_interp, dphi_dNe_interp, Foo, NStart, Nend):
    
    client, cluster = make_dask_client()

    # Create and submit tasks
    Nf_array = np.linspace(Nf_min, Nf_max, M)
    [k_Ne_future, epsilon_Ne_interp_future, dphi_dNe_interp_future, Foo_future] = client.scatter([k_Ne, epsilon_Ne_interp, dphi_dNe_interp, Foo], broadcast=True)
    futures = [client.submit(solve_ms_for_k, k_val, Nf_min, Nf, k_Ne_future, epsilon_Ne_interp_future, dphi_dNe_interp_future, Foo_future, NStart) for Nf in Nf_array]

    results = np.array(client.gather(futures))

    client.close()
    cluster.close()
    
    Nfs = results[:, 1]
    Ps = results[:, 2]
    P_phis = results[:, 3]

    return Nfs, Ps, P_phis

def MS_spectrum_CL(k_min, k_max, Ni, Nf, M, k_Ne, Ne_k, epsilon_Ne_interp, dphi_dNe_interp, Foo, NStart, Nend):
    
    client, cluster = make_dask_client()

    # Create and submit tasks
    k_array = np.geomspace(k_min, k_max, M)
    futures = [client.submit(solve_ms_for_k, k_val, Ni, Nf, k_Ne, epsilon_Ne_interp, dphi_dNe_interp, Foo, NStart ) for k_val in k_array]

    results = np.array(client.gather(futures))

    client.close()
    cluster.close()

    ks = results[:,0]
    Ps = results[:, 2]
    P_phis = results[:, 3]

    return ks, Ps, P_phis

def MS_spectrum(k_min, k_max, M, k_Ne, Ne_k, epsilon_Ne_interp, dphi_dNe_interp, Foo, NStart, Nend):
    
    client, cluster = make_dask_client()

    # Create and submit tasks
    k_array = np.geomspace(k_min, k_max, M)

    Ni_arr = Ne_k(k_array/20)
    
    if np.any(Ni_arr) < 0:
        print("Warning: Some Ni < 0, these are set to 0.")
    Ni_arr = np.where(Ni_arr < 0, 0, Ni_arr)

    Nf_arr = Ne_k(k_array) + 7
    Nf_arr = np.where(Nf_arr > Nend, Nend, Nf_arr)

    [k_Ne_future, epsilon_Ne_interp_future, dphi_dNe_interp_future, Foo_future] = client.scatter([k_Ne, epsilon_Ne_interp, dphi_dNe_interp, Foo], broadcast=True)
    futures = [client.submit(solve_ms_for_k, k_val, Ni, Nf, k_Ne_future, epsilon_Ne_interp_future, dphi_dNe_interp_future, Foo_future, NStart) for k_val, Ni, Nf in zip(k_array, Ni_arr, Nf_arr)]
    print(f"Submitted {M} MS jobs to HTCondor. Tracking progress:")
    progress(futures)
    results = np.array(client.gather(futures))

    client.close()
    cluster.close()

    ks = results[:,0]
    Ps = results[:, 2]
    P_phis = results[:, 3]

    return ks, Ps, P_phis

def ns(ks, Ps):
    lnk = np.log(ks)
    lnP = np.log(Ps)
    dlnP_dlnk = np.gradient(lnP, lnk)
    return 1 + dlnP_dlnk

def plot_background(dir, NeArray, phi_Ne, dphi_dNe,
                    epsilon_Ne, eta_Ne,
                    Hubble_Ne):
    
    os.makedirs(dir+'/plots/', exist_ok=True)

    if phi_Ne is not None:
        
        plt.figure()
        plt.plot(NeArray, phi_Ne, label=r'$\phi(N_e)$')
        plt.xlabel('e-folds $N_e$')
        plt.ylabel(r'$\phi$')
        plt.savefig(dir+'/plots/phi_Ne.pdf')
        plt.close()

    if dphi_dNe is not None:
        plt.figure()
        plt.plot(NeArray, dphi_dNe, label=r"$\phi'(N_e)$")
        plt.xlabel('e-folds $N_e$')
        plt.ylabel(r"$\phi'$")
        plt.savefig(dir+'/plots/dphi_dNe.pdf')
        plt.close()

    plt.figure()
    plt.plot(NeArray, epsilon_Ne, label=r'$\epsilon(N_e)$')
    plt.xlabel('e-folds $N_e$')
    plt.ylabel(r'$\epsilon$')
    plt.savefig(dir+'/plots/epsilon_Ne.pdf')
    plt.close()

    plt.figure()
    plt.plot(NeArray, eta_Ne, label=r'$\eta(N_e)$')
    plt.axhline(1.0, color='k', lw=0.5, ls='--', label=r'$\eta=1$')
    plt.xlabel('e-folds $N_e$')
    plt.ylabel(r'$\eta$')
    plt.savefig(dir+'/plots/eta_Ne.pdf')
    plt.close()

    if Hubble_Ne is not None:
        plt.figure()
        plt.plot(NeArray, Hubble_Ne, label=r'$H(N_e)$')
        plt.xlabel('e-folds $N_e$')
        plt.ylabel(r'$H$')
        plt.savefig(dir+'/plots/Hubble_Ne.pdf')
        plt.close()

    
def save_background(dir, NStart, Nend,
                    phi_Ne_interp, dphi_dNe_interp,
                    epsilon_Ne_interp, eta_Ne_interp,
                    Hubble_Ne_interp):
    
    Ne_vals = np.arange(NStart, Nend, 0.001)   # from NStart to Nend
    shifted_Ne = Ne_vals - NStart              # Shifted so that NStart corresponds to 0

    phi_vals        = phi_Ne_interp(Ne_vals)
    dphi_dt_vals   = dphi_dNe_interp(Ne_vals) * Hubble_Ne_interp(Ne_vals)
    epsilon_vals    = epsilon_Ne_interp(Ne_vals)
    eta_vals        = eta_Ne_interp(Ne_vals)
    H_vals          = Hubble_Ne_interp(Ne_vals)

    np.savetxt(dir+"/phi_vs_Ne.txt",
            np.column_stack([shifted_Ne, phi_vals]),
            header="Ne   phi(Ne)")

    np.savetxt(dir+"/dphidt_vs_Ne.txt",
            np.column_stack([shifted_Ne, dphi_dt_vals]),
            header="Ne   dphi/dt")

    np.savetxt(dir+"/epsilon_vs_Ne.txt",
            np.column_stack([shifted_Ne, epsilon_vals]),
            header="Ne   epsilon(Ne)")

    np.savetxt(dir+"/eta_vs_Ne.txt",
            np.column_stack([shifted_Ne, eta_vals]),
            header="Ne   eta(Ne)")

    np.savetxt(dir+"/H_vs_Ne.txt",
            np.column_stack([shifted_Ne, H_vals]),
            header="/Ne   H(Ne)")

def calculate_MS_spectrums(dir, Hubble_Ne_interp, dphi_dNe_interp,
                           epsilon_Ne_interp,eta_Ne_interp,NStart,Nend, WKB=False, k_low=1e16, k_high=1e19,k_val = 6e16, M=100):
    Mp= 2.435e18 
    def k_Ne(Ne):
        return Hubble_Ne_interp(Ne) * np.exp(Ne-NStart) * Mp
    
    x = np.arange(0,Nend,0.0001)
    Ne_k = UnivariateSpline(k_Ne(x),x,s=0)
    plt.figure()
    plt.plot(x,k_Ne(x))
    plt.xlabel('Ne')
    plt.ylabel('k(Ne)')
    plt.yscale('log')
    plt.savefig(dir+'/plots/k_Ne.pdf')
    plt.close()

    def epsilon_prime(Ne):
        return epsilon_Ne_interp.derivative()(Ne)

    def eta_prime(Ne):
        return eta_Ne_interp.derivative()(Ne)

    def Foo_mod(Ne, k_val):
        kc = k_Ne(Ne)
        eps = epsilon_Ne_interp(Ne)
        et = eta_Ne_interp(Ne)
        epsp = epsilon_prime(Ne)
        etp = eta_prime(Ne)
        return (k_val / kc) ** 2 + (1 + eps - et) * (et - 2) - (epsp - etp) + 2 * eps * (3 + eps - 2 * et)
        
    def Foo(Ne, k_val):
        kc = k_Ne(Ne)
        eps = epsilon_Ne_interp(Ne)
        et = eta_Ne_interp(Ne)
        epsp = epsilon_prime(Ne)
        etp = eta_prime(Ne)
        return (k_val / kc) ** 2 + (1 + eps - et) * (et - 2) - (epsp - etp)
    
    def P_SR(Ne):
        return Hubble_Ne_interp(Ne) ** 2 / (8 * np.pi ** 2 * epsilon_Ne_interp(Ne))

    # --- Power Spectrum ---
    
    ks, Ps, P_phis = MS_spectrum(k_low, k_high, M, k_Ne, Ne_k, epsilon_Ne_interp, dphi_dNe_interp, Foo, NStart, Nend)
    ks_mod, Ps_mod, P_phis_mod = MS_spectrum(k_low, k_high, M, k_Ne, Ne_k, epsilon_Ne_interp, dphi_dNe_interp, Foo_mod, NStart, Nend)
    
    plt.figure(figsize=[6.4, 4.8])
    plt.title(r'Power Spectrum $\Delta_{\mathcal{R},\text{Def}}(k)$')
    plt.plot(ks_mod/Mp, Ps_mod, label=r'$\Delta_{\mathcal{R},\text{Def}}(k)$')
    plt.plot(ks/Mp, Ps, label=r'$\Delta_\mathcal{R}(k)$')
    plt.plot(ks/Mp, P_SR(Ne_k(ks)),label=r'$\Delta_{\mathcal{R},\text{Slow Roll}}(k)$', linestyle="--", color='#d62728')
    plt.xlabel(r'$k$')
    plt.ylabel(r'$\Delta_\mathcal{R}(k)$')
    plt.xscale('log')
    plt.yscale('log')
    plt.legend()
    plt.tight_layout()
    plt.savefig(dir+'/plots/PS_k.pdf')
    plt.close()
    
    
    print(f"Power spectrum plot saved in {dir}/plots/PS_k.pdf")
    
    
    Nfs_mod, Pmodes_mod, Pphimodes_mod = MS_mode(NStart, NStart+15, k_val, M, k_Ne, Ne_k, epsilon_Ne_interp, dphi_dNe_interp, Foo_mod, NStart, Nend)
    Nfs, Pmodes, Pphimodes = MS_mode(NStart, NStart+15, k_val, M, k_Ne, Ne_k, epsilon_Ne_interp, dphi_dNe_interp, Foo, NStart, Nend)

    if NStart > 0:
        Nfs = Nfs - NStart
        Nfs_mod = Nfs_mod - NStart

    plt.figure()
    plt.title(r'Evolution of Modified MS Mode $\Delta_\mathcal{R}(k=2.46 \times 10^{-2}, N_e)$')
    #plt.plot(Nfs_mod, Pmodes_mod, label=r'$\Delta_{\mathcal{R},\text{Def}}(k=2.46 \times 10^{-2}, N_e)$')
    plt.plot(Nfs, Pmodes, label=r'$\Delta_\mathcal{R}(k=2.46 \times 10^{-2}, N_e)$')
    plt.xlabel(r'$N_e$')
    plt.ylabel(r'$\Delta_\mathcal{R}(k=2.46 \times 10^{-2}, N_e)$')
    plt.yscale('log')
    plt.legend()
    plt.tight_layout()
    plt.savefig(dir+'/plots/PS_Ne_k6e16.pdf')
    plt.close()

    print(f"Power spectrum mode saved in {dir}/plots/PS_Ne_k6e16.pdf")
    
    
    np.savetxt(dir+"/P_k.txt",
            np.column_stack([ks, Ps, P_phis, Ps_mod, P_phis_mod]),
            header="k    P(k)   Pphi(k)   P_mod(k)   Pphi_mod(k)")
    
    np.savetxt(dir+"/P_Ne_k6e16.txt",
               np.column_stack([Nfs, Pmodes, Pphimodes, Pmodes_mod, Pmodes_mod]),
                header="Ne    P(Ne)   Pphi(Ne)   P_mod(Ne)   Pphi_mod(Ne)")
    

    if WKB:
        Ne = np.arange(NStart, NStart+15, 1e-3)
        int_eps = cumulative_trapezoid(epsilon_Ne_interp(Ne), Ne, initial=0)
        u_grow = np.exp(-2 * (int_eps - int_eps[0]))
        plt.plot(Ne - NStart, np.abs(u_grow)**2,label="Late time WKB solution")

        
        plt.figure(figsize=(5, 3))
        myblue = '#005f87'
        myred = '#bc1e00'
        plt.title(r'Evolution of Modified MS Mode $\Delta_\mathcal{R}(k,N_e)$')
        plt.plot(Nfs, Pmodes, label=r'$\Delta_\mathcal{R}(k=2.46 \times 10^{-2}, N_e)$', color=myblue)
        plt.plot(Nfs_mod, Pmodes_mod, label=r'$\Delta_{\mathcal{R},\text{Def}}(k=2.46 \times 10^{-2}, N_e)$', color=myred)
        plt.plot(Ne - NStart, np.abs(u_grow)**2 * Pmodes_mod[-1] / u_grow[-1]**2,'k--',label="Late time WKB solution")
        plt.xlabel(r'$N_e$')
        plt.ylabel(r'$\Delta_\mathcal{R}(k=2.46 \times 10^{-2}, N_e)$')
        plt.yscale('log')
        plt.legend()
        plt.tight_layout()
        plt.savefig(dir+'/plots/lateTimeWKB.pdf')
        plt.close()
        
        plt.figure(figsize=(5, 4))
        plt.title(r'Power Spectrum $\Delta_{\mathcal{R},\text{Def}}(k)$')
        plt.plot(ks_mod/Mp, Ps_mod, label=r'$\Delta_{\mathcal{R},\text{Def}}(k)$')
        plt.plot(ks/Mp, Ps, label=r'$\Delta_\mathcal{R}(k)$')

        k_array = np.geomspace(k_low, k_high, M)
        corr = (Hubble_Ne_interp(Ne_k(k_array/20)) / Hubble_Ne_interp(Ne_k(k_array) + 7))**4
        plt.plot(ks/Mp, P_SR(Ne_k(ks)),label=r'$\Delta_{\mathcal{R},\text{Slow Roll}}(k)$', linestyle="--", color='#d62728')
        plt.xlabel(r'$k$')
        plt.ylabel(r'$\Delta_\mathcal{R}(k)$')
        plt.xscale('log')
        plt.yscale('log')
        plt.ylim(1e-6,2e-5)
        plt.legend()
        plt.tight_layout()
        plt.savefig(dir+'/plots/PS_k_corrected.pdf')
        plt.close()

        np.savetxt(dir+"/P_k_corrected.txt",
            np.column_stack([ks, Ps, P_phis, Ps_mod * corr, P_phis_mod]),
        header="k    P(k)   Pphi(k)   P_mod(k)   Pphi_mod(k)")

        sys.exit()
    
    
        

def main():
    
    ### --- lphi4 Inflation ---

    # Plot potential
    global lam0
    lam0 = 1e-9

    os.makedirs('plots/', exist_ok=True)

    x = np.linspace(0, 10, 1000)
    plt.plot(x, V_phi(x))
    plt.xlabel(r'$\phi$')
    plt.ylabel(r'$V(\phi)$')
    plt.savefig('plots/V_phi.pdf')
    plt.close()

    ## Initial conditions

    phi0 = 50
    dphi0 = - np.sqrt(2 * epsilon_SR(phi0))

    ## Background evolution

    maxN = 1000  # Maximum number of e-folds to evolve
    NeArray = np.arange(0, maxN,1e-3)

    # Event properties
    phi_crosses_zero.terminal = True  # Stop the solver when the event is triggered
    phi_crosses_zero.direction = -1   # Only trigger when h is decreasing (crossing zero from positive to negative)

    # Solve the system using solve_ivp with the event
    sol = solve_ivp(lambda Ne, x: phi_derivatives(x, Ne, V_phi, dV_dphi),
        (NeArray[0], NeArray[-1]),
        [phi0, dphi0],
        t_eval=NeArray,
        method='DOP853',
        atol=1e-12,    
        rtol=1e-12,    
        events=phi_crosses_zero)

    # Access the solution up to the stopping point
    phi_Ne = sol.y[0]
    dphi_dNe = sol.y[1]
    NeArray = sol.t
    
    ## Background quantities
    phi_Ne_interp = UnivariateSpline(NeArray, phi_Ne, k=1,s=0)
    dphi_dNe_interp = UnivariateSpline(NeArray, dphi_dNe, k=1,s=0)
    epsilon_Ne = dphi_dNe ** 2 / 2
    epsilon_Ne_interp = UnivariateSpline(NeArray, epsilon_Ne, k=1,s=0)
    eta_Ne = epsilon_Ne - 0.5 * epsilon_Ne_interp.derivative()(NeArray) / epsilon_Ne
    eta_Ne_interp = UnivariateSpline(NeArray, eta_Ne, k=1,s=0)
    Hubble_Ne = np.where(V_phi(phi_Ne)/(3-epsilon_Ne) > 0, np.sqrt(V_phi(phi_Ne) / (3 - epsilon_Ne)),0)
    Hubble_Ne_interp = UnivariateSpline(NeArray, Hubble_Ne, k=1,s=0)
    
    ## Identify the end of inflation and CMB scale exit
    Nend = optimize.brentq(lambda Ne: epsilon_Ne_interp(Ne) - 1.0, 0, NeArray[-1])
    print(f"End of inflation at N_e = {Nend:.2f}")

    NCMB = Nend - 55
    print(f"CMB scales exit the horizon at N_e = {NCMB:.2f}\n")

    ## Calculate inputs for CosmoLattice
    NStart = NCMB - 2
    global Mp
    Mp= 2.435e18  # Reduced Planck mass in GeV
    kh0 = Hubble_Ne_interp(NStart) * Mp
    CL_phi0 = phi_Ne_interp(NStart) * Mp
    omegaStar = np.sqrt(lam0) * phi_Ne_interp(NStart) * Mp
    CL_dphi_dNe0 = dphi_dNe_interp(NStart) * Hubble_Ne_interp(NStart) * Mp** 2

    print(f"--------------------------------")
    print("--- Inputs for CosmoLattice ---")
    print(f"--------------------------------\n")
    print(f"Choosing start at {NStart:.2f} e-folds")
    print(f"CL_lam0 = lam0 = {lam0}")
    print(f"CL_phi0 = fStar = {CL_phi0}")
    print(f"CL_dphi_dNe0 = {CL_dphi_dNe0}")
    print(f"omegaStar = {omegaStar}")
    print(f"Hubble at start {kh0}")
    print(f"To start IR mode 1 e-fold inside horizon: kIR = {kh0 * np.exp(-1) / omegaStar:5f}")
    print(f"--------------------------------\n")
    

    # Save background plots and data
    
    plot_background('lphi4Inflation', NeArray, phi_Ne, dphi_dNe,
                    epsilon_Ne, eta_Ne, Hubble_Ne)
    save_background('lphi4Inflation', NStart, Nend,
                    phi_Ne_interp, dphi_dNe_interp,
                    epsilon_Ne_interp, eta_Ne_interp,
                    Hubble_Ne_interp) 
    
    # Calculate MS spectrums
    calculate_MS_spectrums('lphi4Inflation', Hubble_Ne_interp, dphi_dNe_interp,
                           epsilon_Ne_interp,eta_Ne_interp,NStart,Nend, WKB=True)
    
    ### --- Resolve background with CL start point ---
    Ne_max = 30
    Ne_vals = np.linspace(0, Ne_max, 5000)

    y0      = phi_Ne_interp(NStart)          
    dy0     = dphi_dNe_interp(NStart)

    sol_CL = solve_ivp(lambda Ne, x: phi_derivatives(x, Ne, V_phi, dV_dphi),
        (Ne_vals[0], Ne_vals[-1]),
        [y0, dy0],
        t_eval=Ne_vals,
        method='DOP853',
        atol=1e-12,    
        rtol=1e-12,    
        events=phi_crosses_zero)
    
    ySol  = sol_CL.y[0]
    dySol = sol_CL.y[1]

    ySol_interp = UnivariateSpline(Ne_vals, ySol, k=1,s=0)
    dySol_interp = UnivariateSpline(Ne_vals, dySol, k=1,s=0)
    H_CL_vals = np.sqrt(2 * V_phi(ySol)/(6.0 - dySol**2))
    H_CL_interp = UnivariateSpline(Ne_vals, H_CL_vals, k=1,s=0)
    epsilon_CL = - H_CL_interp.derivative()(Ne_vals) / H_CL_vals
    epsilon_CL_interp = UnivariateSpline(Ne_vals, epsilon_CL, k=1,s=0)
    eta_CL = epsilon_CL - 0.5 * epsilon_CL_interp.derivative()(Ne_vals) / epsilon_CL
    eta_CL_interp = UnivariateSpline(Ne_vals, eta_CL, k=1,s=0)
    

    plot_background('lphi4InflationCL', Ne_vals, ySol, dySol,
                    epsilon_CL, eta_CL, H_CL_vals)
    
    save_background('lphi4InflationCL', 0, 30,
                    ySol_interp, dySol_interp,
                    epsilon_CL_interp, eta_CL_interp,
                    H_CL_interp)
    
    # Calculate MS spectrums
    calculate_MS_spectrums('lphi4InflationCL', H_CL_interp, dySol_interp,
                            epsilon_CL_interp,eta_CL_interp,0,10)


if __name__ == '__main__':
    main()