import matplotlib.pyplot as plt
from scipy import optimize
import numpy as np
from scipy.interpolate import UnivariateSpline
from scipy.integrate import solve_ivp, cumulative_trapezoid
from dask.distributed import LocalCluster, Client, progress
import os
import sys
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from Analytics.lphi4 import phi_derivatives, solve_ms_for_k, MS_mode, MS_spectrum, MS_spectrum_CL, ns, plot_background, save_background, calculate_MS_spectrums, solve_ms_for_k_timeseries

# Define model
c2 = 0.03
c3 = 0.075
phi0 = 0.97

# Paramaters to optimise xi
h0 = 11 # Initial value for inflaton
maxN = 90 # Maximum number of e-folds to calculate up to
lam = 6.6e-11
xi = 0.10483

def V_phi(phi):
        return lam / (24 * (1 + xi * phi**2)**2) * (
        3 * phi**4 + xi**2 * phi0**4 * phi**4 - 8 * (1 + c3) * phi0 * phi**3 +
        2 * (1 + c2) * (3 + xi * phi0**2) * phi0**2 * phi**2)

def hofphi(phi):
        return np.sqrt((1 + 6 * xi) / xi) * np.arcsinh(phi * np.sqrt(xi * (1 + 6 * xi))) - \
            np.sqrt(6) * np.arctanh((np.sqrt(6) * xi * phi) / np.sqrt(1 + xi * (1 + 6 * xi) * phi**2))

# Function to find the root
def root_function(phi, h):
    return hofphi(phi) - h

def find_phi_for_h(h):
    for phi_initial_guess in np.linspace(1.0, 12.0, 100):
        solution = optimize.root(lambda phi, h: root_function(phi, h), phi_initial_guess, args=(h,))
        if solution.success:
            return solution.x[0]
            break
        else:
            continue

# Define the potential V(h) in terms of h
def V_h(h):
    phi = find_phi_for_h(h)
    return V_phi(phi)

def h_derivatives(x, Ne, V_h_interp, V_h_grad_interp):
        return [x[1], -3 * x[1] + x[1]**3 / 2 - (3 - (x[1]**2 / 2)) * V_h_grad_interp(x[0])/V_h_interp(x[0])]


def main():
    
    ### --- lphi4 Inflation ---

    os.makedirs('pbh_CL/plots/', exist_ok=True)
    Mp = 2.435e18

    # Generate values for h between 0 and 2
    h_values = np.linspace(0, 14, 1000)
    V_h_values = [V_h(h) for h in h_values]

    # Create the interpolation function for V(h)
    V_h_interp = UnivariateSpline(h_values, V_h_values, k=4,s=0)

    # Generate values for h between 0 and 2 for plotting the gradient
    V_h_grad = V_h_interp.derivative()
    grad_V_h_values = V_h_grad(h_values)

    ##########################################
    # --- Input potential for Cosmolattice ---
    ##########################################

    V_h_grad2 = V_h_grad.derivative()

    # Evaluate on grid
    grad_V_h_values = V_h_grad(h_values)
    grad2_V_h_values = V_h_grad2(h_values)

    # Save to text file
    #with open("Vh_table.txt", "w") as f:
    #    f.write(f"{len(h_values)}\n")
    #    for h, V, dV, d2V in zip(h_values, V_h_values, grad_V_h_values, grad2_V_h_values):
    #        f.write(f"{h:.16e} {V:.16e} {dV:.16e} {d2V:.16e}\n")

    # Get 8th-order polynomial coefficients
    # V(h) approx p[0]*h^8 + p[1]*h^7 ... + p[8]
    poly = 20
    h_values_fit = np.linspace(0, 6, 1000)
    CLh0 = 3.73365369687465
    V_h_values_fit = V_h_interp(h_values_fit)
    p = np.polyfit(h_values_fit - CLh0, V_h_values_fit, poly)
    for i, coef in enumerate(p):
        print(f"a{poly-i} = {coef:.18e}")

    V_h_exact = V_h_values_fit
    V_h_poly = np.polyval(p, h_values_fit-CLh0)

    # 3. Calculate Difference (Residuals)
    residuals = V_h_exact - V_h_poly

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    # Top Plot: Potentials
    ax1.plot(h_values, V_h_exact, 'b-', linewidth=2, label=r'Exact $V(h)$ (Numerical)')
    ax1.plot(h_values, V_h_poly, 'r--', linewidth=2, label=r'Polynomial Fit (Order 8)')
    ax1.set_ylabel(r'$V(h)$')
    ax1.set_title('Potential Comparison: Exact vs. Polynomial Fit')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Bottom Plot: Residuals
    ax2.plot(h_values, residuals, 'g-', label=r'Residuals ($V_{exact} - V_{poly}$)')
    ax2.set_xlabel(r'Einstein Field $h$')
    ax2.set_ylabel(r'$\Delta V$')
    ax2.set_title('Accuracy Analysis (Residuals)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('pbh_CL/plots/potential_fit_and_residuals.pdf')
    plt.close()

    ##########################################
    
    # Solve the background evolution
    NeArray = np.arange(0,maxN,1e-3)
    h0 = 11
    H0 = np.sqrt(V_h_interp(h0) / 3)
    dh = - V_h_grad(h0) / (3 * H0)

    # --- Run CL evolution ---
    CL_evolution = True

    if CL_evolution:
        V_h_interp = UnivariateSpline(h_values_fit, V_h_poly, k=4,s=0)
        V_h_grad = V_h_interp.derivative()
        h0 = CLh0
        dh = -0.7009118925482988

    # ----
        
    # Define an event function to stop when h crosses zero
    def h_crosses_zero_event(Ne, x):
        return x[0]  # This triggers when h crosses zero

    # Event properties
    h_crosses_zero_event.terminal = True  # Stop the solver when the event is triggered
    h_crosses_zero_event.direction = -1   # Only trigger when h is decreasing (crossing zero from positive to negative)

    # Solve the system using solve_ivp with the event
    sol = solve_ivp(lambda Ne, x: h_derivatives(x, Ne, V_h_interp, V_h_grad),
        (NeArray[0], NeArray[-1]),
        [h0, dh],
        t_eval=NeArray,
        method='DOP853',  # or 'DOP853' for better performance
        atol=1e-12,     # Absolute tolerance
        rtol=1e-12,      # Relative tolerance,
        events=h_crosses_zero_event)

    # Access the solution up to the stopping point
    phi_Ne = sol.y[0]  # First variable (h_sol)
    dphi_dNe = sol.y[1]  # Second variable (dhdNe_sol)
    NeArray = sol.t

    ## Background quantities
    phi_Ne_interp = UnivariateSpline(NeArray, phi_Ne, k=1,s=0)
    dphi_dNe_interp = UnivariateSpline(NeArray, dphi_dNe, k=1,s=0)
    epsilon_Ne = dphi_dNe ** 2 / 2
    epsilon_Ne_interp = UnivariateSpline(NeArray, epsilon_Ne, k=1,s=0)
    eta_Ne = epsilon_Ne - 0.5 * epsilon_Ne_interp.derivative()(NeArray) / epsilon_Ne
    eta_Ne_interp = UnivariateSpline(NeArray, eta_Ne, k=1,s=0)
    Hubble_Ne = np.where(V_h_interp(phi_Ne)/(3-epsilon_Ne) > 0, np.sqrt(V_h_interp(phi_Ne) / (3 - epsilon_Ne)),0)
    Hubble_Ne_interp = UnivariateSpline(NeArray, Hubble_Ne, k=1,s=0)
    

    # Save background plots and data

    
    if CL_evolution:
        NStart = 0
    else:
        NStart = 74

    Nend = NeArray[-1]

    path = 'pbh_CL' if CL_evolution else 'pbh'
    plot_background(path, NeArray, phi_Ne, dphi_dNe,
                    epsilon_Ne, eta_Ne, Hubble_Ne)
    save_background(path, 0, Nend,
                    phi_Ne_interp, dphi_dNe_interp,
                    epsilon_Ne_interp, eta_Ne_interp,
                    Hubble_Ne_interp) 

    # --- Inputs for CL ---
    
    print(f"initial_amplitudes = {phi_Ne_interp(NStart) * Mp}")
    print(f"dphi_dNe_interp(NStart) = {dphi_dNe_interp(NStart)}")
    print(f"initial_momenta = {dphi_dNe_interp(NStart) * Hubble_Ne_interp(NStart) * Mp** 2}")
    kh0 = Hubble_Ne_interp(NStart)
    print(f"k0 = {kh0}")
    print(f"kIR = {kh0 * np.exp(-2)}")
    V_h_fit_interp = UnivariateSpline(h_values_fit, V_h_values_fit, k=4,s=0)
    print(f"phi0 = {phi_Ne_interp(NStart)}")
    print(f"V0 = {V_h_fit_interp(phi_Ne_interp(NStart))}")
    print(f"Hubble from V0 = {np.sqrt(V_h_interp(phi_Ne_interp(NStart)) / 3)}")
    print(f"Hubble from background evolution = {Hubble_Ne_interp(NStart)}")
    print(f"epsilon0 = {epsilon_Ne_interp(NStart)}")
    print("Background evolution complete.")
    
    Nend = NeArray[-1]
    Mp = 2.435e18 

    def k_Ne(Ne):
        return Hubble_Ne_interp(Ne) * np.exp(Ne-NStart) * Mp
    
    x = np.arange(0,Nend,0.0001)
    k_interp = UnivariateSpline(x, k_Ne(x), k=1,s=0)
    
    plt.figure()
    plt.plot(x,k_interp(x)/Mp)
    plt.xlabel('Ne')
    plt.ylabel('k(Ne)')
    plt.yscale('log')
    plt.savefig(path+'/plots/k_Ne.pdf')
    plt.close()

    def Ne_k(ks):
        solution = [optimize.brentq(
            lambda Ne: k_interp(Ne) - k, 
            0, Nend
        ) for k in ks]
        return np.array(solution)

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
        

    if not CL_evolution:
        # ------------------------------------
        # Single mode evolution plots
        # ------------------------------------
        
        M = 100
        k_low = 1e-11 * Mp
        k_high = k_Ne(Nend)
        

        # Find the first e-fold where eta exceeds 3
        idx_eta3 = np.argmax(eta_Ne > 3)

        if not np.any(eta_Ne > 3):
            raise ValueError("eta never exceeds 3 in the computed background.")

        Ne_eta3 = NeArray[idx_eta3]
        Ne_kval = Ne_eta3 - 10

        if Ne_kval < NeArray[0]:
            raise ValueError("10 e-folds before eta>3 lies before the start of the background evolution.")

        k_val = k_Ne(Ne_kval)
        print(f"eta first exceeds 3 at N_e = {Ne_eta3:.6f}")
        print(f"Using k_val from N_e = {Ne_kval:.6f}")
        print(f"k_val / Mp = {k_val / Mp:.3e}")
            
        Nin = Ne_k([k_val/20])[0]
        res = solve_ms_for_k_timeseries(k_val, Nin, Nend,
                                    k_Ne, epsilon_Ne_interp, dphi_dNe_interp, Foo, NStart,
                                    t_steps=8000, method="Radau", max_step=1e-4)

        res_mod = solve_ms_for_k_timeseries(k_val, Nin, Nend,
                                            k_Ne, epsilon_Ne_interp, dphi_dNe_interp, Foo_mod, NStart,
                                            t_steps=8000, method="Radau", max_step=1e-4)

        Ne = res['Ne']
        R = res['R']; I = res['I']
        R_mod = res_mod['R']; I_mod = res_mod['I']
        
        # Debugging plots
        plt.figure(figsize=(8,4))
        plt.plot(Ne, R, label='Re(f)')
        plt.plot(Ne, I, label='Im(f)')
        plt.plot(Ne, R_mod, label='Re(f) mod')
        plt.plot(Ne, I_mod, label='Im(f) mod')

        plt.xlabel(r'$N_e$')
        plt.ylabel('mode function (f)')
        plt.legend()
        plt.title(f'k = {k_val:.3e}')
        plt.savefig('pbh/plots/mode.pdf', bbox_inches='tight')
        plt.close()

        plt.figure()
        plt.plot(Ne, np.sqrt(R**2 + I**2), label='|f|')
        plt.yscale('linear')
        plt.legend()
        plt.savefig('pbh/plots/mode_u2.pdf', bbox_inches='tight')
        plt.close()
        # ---- New: plot P(Ne) for both original and modified ----
        P = res['P']
        P_mod = res_mod['P']

        # Plot P vs Ne (log y)
        plt.figure(figsize=(8,5))
        plt.plot(Ne, P, label=r'$\Delta_\mathcal{R}$ Mukhanov-Sasaki')
        plt.plot(Ne, P_mod, label=r'$\Delta_\mathcal{R}$ Modified', linestyle='--')
        plt.yscale('log')
        plt.xlabel(r'$N_e$')
        plt.ylabel(r'$\Delta_\mathcal{R}(N_e)$')
        plt.axvline(Ne_k([k_val])[0],color='black',linestyle=':',label='Mode Exit')
        plt.axvline(Ne_k([k_val])[0]+7,color='green',linestyle=':',label="`Final' Mode Evaluation")
        plt.legend()
        k_scaled = k_val / Mp
        mantissa, exp = f"{k_scaled:.1e}".split("e")
        label = rf'$\Delta_\mathcal{{R}}(k = {mantissa}\times10^{{{int(exp)}}}, N_e)$'
        plt.title(label)
        plt.savefig(f'pbh/plots/k/P_vs_Ne_{k_val/Mp:.2E}.pdf', bbox_inches='tight')
        plt.close()

        print(f'Plotted pbh/plots/k/P_vs_Ne_{k_val/Mp:.2E}.pdf')
        
        # ------------------------------------
        # Generate power spectrum
        # ------------------------------------

        ks = np.geomspace(k_low, k_high, M)
        ks, Ps, P_phis = MS_spectrum(k_low, k_high, M, k_Ne, Ne_k, epsilon_Ne_interp, dphi_dNe_interp, Foo, NStart, Nend)
        # Save standard spectrum
        np.savetxt(
            "pbh/PS_standard.txt",
            np.column_stack([ks, Ps, P_SR(Ne_k(ks))]),
            header="k  Delta_R(k)  Delta_R_slow_roll(k)"
        )
        
        ks_mod, Ps_mod, P_phis_mod = MS_spectrum(k_low, k_high, M, k_Ne, Ne_k, epsilon_Ne_interp, dphi_dNe_interp, Foo_mod, NStart, Nend)
        # Save modified spectrum
        np.savetxt(
            "pbh/PS_modified.txt",
            np.column_stack([ks_mod, Ps_mod]),
            header="k  Delta_R_mod(k)"
        )

        plt.figure(figsize=[6.4, 4.8])
        plt.title(r'Power Spectrum $\Delta_{\mathcal{R}(k)}$')
        plt.plot(ks_mod/Mp, Ps_mod, label=r'$\Delta_{\mathcal{R},\text{Mod}}(k)$')
        plt.plot(ks/Mp, Ps, label=r'$\Delta_\mathcal{R}(k)$')
        plt.plot(ks_mod/Mp, P_SR(Ne_k(ks_mod)),label=r'$\Delta_{\mathcal{R},\text{Slow Roll}}(k)$', linestyle="--", color='#d62728')
        plt.xlabel(r'$k$')
        plt.ylabel(r'$\Delta_\mathcal{R}(k)$')
        plt.xscale('log')
        plt.yscale('log')
        plt.legend()
        plt.tight_layout()
        plt.savefig('pbh/plots/PS_k.pdf')
        plt.close()

    else: 

        k_val = 8e-6 * Mp
        Nin = 0
        res = solve_ms_for_k_timeseries(k_val, Nin, Nend,
                                    k_Ne, epsilon_Ne_interp, dphi_dNe_interp, Foo, NStart,
                                    t_steps=8000, method="Radau", max_step=1e-4)

        res_mod = solve_ms_for_k_timeseries(k_val, Nin, Nend,
                                            k_Ne, epsilon_Ne_interp, dphi_dNe_interp, Foo_mod, NStart,
                                            t_steps=8000, method="Radau", max_step=1e-4)

        Ne = res['Ne']
        R = res['R']; I = res['I']
        R_mod = res_mod['R']; I_mod = res_mod['I']
        P = res['P']
        P_mod = res_mod['P']

        # Plot P vs Ne (log y)
        plt.figure(figsize=(8,5))
        plt.plot(Ne, P, label='P (orig)')
        plt.plot(Ne, P_mod, label='P (mod)', linestyle='--')
        plt.yscale('log')
        plt.xlabel(r'$N_e$')
        plt.ylabel(r'$P(N_e)$')
        plt.axvline(Ne_k([k_val])[0],color='black',linestyle=':',label='Mode Exit')
        plt.axvline(Ne_k([k_val])[0]+7,color='green',linestyle=':',label='"Final" Mode Evaluation')
        plt.legend()
        plt.title(f'Power spectrum vs $N_e$, k = {k_val/Mp:.3e}')
        
        plt.savefig(f'pbh_CL/plots/P_vs_Ne_{k_val/Mp:.2E}.pdf', bbox_inches='tight')
        plt.close()


        # ------------------------------------
        # Generate files comparison with CL
        # ------------------------------------
        
        M=75
        k_low = 1e-6 * Mp
        k_high = 4e-4 * Mp

        cmap = plt.cm.viridis
        Nf_values = [0,1,2,3,4,5]
        colors = {Nf: cmap(i / (len(Nf_values)-1 if len(Nf_values)>1 else 1)) for i, Nf in enumerate(Nf_values)}
        norm = mcolors.Normalize(vmin=min(Nf_values), vmax=max(Nf_values))

        plt.figure(figsize=[6.4, 4.8])
        for Nf in Nf_values:
            print(f'Solving {Nf}')
            ks_mod, Ps_mod, P_phis_mod = MS_spectrum_CL(k_low, k_high, 0, Nf, M, k_Ne, Ne_k, epsilon_Ne_interp, dphi_dNe_interp, Foo_mod, NStart, Nend)
            ks, Ps, P_phis = MS_spectrum_CL(k_low, k_high, 0, Nf, M, k_Ne, Ne_k, epsilon_Ne_interp, dphi_dNe_interp, Foo, NStart, Nend)
            # Save modified spectrum
            np.savetxt(
                f"pbh_CL/PS_Nf={Nf}.txt",
                np.column_stack([ks_mod, Ps_mod]),
                header="k  Delta_R_mod(k)"
            )
            plt.plot(ks_mod/Mp, Ps_mod, label=r'$\Delta_{\mathcal{R},\text{Mod}}(k,N_e)$' if Nf == 0 else None,color = cmap(norm(Nf)),linestyle="--")
            
            np.savetxt(
                f"pbh_CL/PS_modified_Nf={Nf}.txt",
                np.column_stack([ks, Ps]),
                header="k  Delta_R_mod(k)"
            )
            plt.plot(ks/Mp, Ps, label=r'$\Delta_{\mathcal{R}}(k,N_e)$' if Nf == 0 else None,color = cmap(norm(Nf)))

        sm = cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=plt.gca())
        cbar.set_label(r'$N_e$')


        plt.title(r'Power Spectrum $\Delta_{\mathcal{R},\text{Mod}}(k)$')
        plt.xlabel(r'$k$')
        plt.ylabel(r'$\Delta_\mathcal{R}(k)$')
        plt.xscale('log')
        plt.yscale('log')
        plt.ylim(2e-18, 8e-4)
        plt.legend(loc='upper right')
        plt.tight_layout()
        plt.savefig('pbh_CL/plots/PS_k.pdf')
        plt.close()

        # ------------------------------------


if __name__ == '__main__':
    main()