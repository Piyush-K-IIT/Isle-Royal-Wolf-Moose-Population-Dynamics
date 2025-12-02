import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.optimize import minimize
import matplotlib.pyplot as plt

# Fixing h - Handling Time (set based on external ecological data/literature)
H_FIXED_VALUE = 3.65 / 365.25 # approx 0.01 years/kill or 10 days/kill


# ODE Model
def hollingII_5param(t, y, theta):


    # Parameters are exponentiated for positivity
    r, a, e, gamma, K = np.exp(theta)
    h = H_FIXED_VALUE 
    M, W = y
    
    # Holling Type II Predation Term
    Predation_Term = (a * M * W) / (1 + h * a * M)
    
    # Differential Equations
    dM = r * M * (1 - M / K) - Predation_Term
    dW = e * Predation_Term - gamma * W
    return [dM, dW]


def simulate(theta, M0, W0, tspan):
    """Integrates the 5-parameter ODE system."""
    sol = solve_ivp(
        fun=lambda t, y: hollingII_5param(t, y, theta),
        t_span=(tspan[0], tspan[-1]),
        y0=[M0, W0],
        t_eval=tspan,
        method='RK45',
        rtol=1e-6, atol=1e-8 
    )
    return sol.y.T if sol.success else np.full((len(tspan), 2), np.nan)


# Loss Function
def loss_function(theta, M_obs, W_obs, t):
    """Calculates the weighted sum of squared relative log-residuals (RSS)."""
    
    pred = simulate(theta, M_obs[0], W_obs[0], t)
    M_pred, W_pred = pred[:, 0], pred[:, 1]

    # Penalty for numerical failure or negative population values
    if np.any(M_pred <= 0) or np.any(W_pred <= 0) or np.any(np.isnan(M_pred)):
        return 1e12 
    
    # Calculate residuals for log-transformed data
    residuals_M = np.log(M_obs) - np.log(M_pred)
    residuals_W = np.log(W_obs) - np.log(W_pred)
    WOLF_WEIGHT = 5.0 
    
    total_error = np.sum(residuals_M**2) + WOLF_WEIGHT * np.sum(residuals_W**2)
    return total_error


from scipy.interpolate import UnivariateSpline
from scipy.optimize import curve_fit
from scipy.signal import savgol_filter


def lv_initial_guess(M, W, t):
 
    try:
        window = min(len(t)//2*2+1, 7)
        if window < 5: window = 5
        M_s = savgol_filter(M, window, 2, mode="interp")
        W_s = savgol_filter(W, window, 2, mode="interp")
    except:
        M_s = M.copy()
        W_s = W.copy()

    M_s = np.maximum(M_s, 1e-6)
    W_s = np.maximum(W_s, 1e-6)

    dM = np.gradient(M_s, t)
    dW = np.gradient(W_s, t)
    dM_per = dM / M_s
    dW_per = dW / W_s

    # We *force* reasonable predation curvature (avoid collapse)
    h = H_FIXED_VALUE

    # crude slopes
    slope = np.cov(M_s, dW_per)[0,1] / (np.var(M_s) + 1e-8)
    gamma_init = max(0.05, -np.min(dW_per))  # must be noticeable

    # FORCE a and e NOT TO BE TINY:
    a_init = max(0.005, min(0.05, abs(slope)))  # forcing attack rate
    e_init = max(0.1, min(0.8, np.mean(W) / np.mean(M)))  # plausible EI recovery

    pred_term = (a_init * M_s * W_s) / (1 + a_init * h * M_s)
    Z = dM_per + pred_term

    Xm = np.column_stack([np.ones_like(M_s), M_s])
    try:
        beta, *_ = np.linalg.lstsq(Xm, Z, rcond=None)
        r_init = max(0.01, min(0.8, beta[0]))
        r_over_K = -beta[1]
        if r_over_K > 0:
            K_init = max(np.max(M)*1.1, r_init / r_over_K)
        else:
            K_init = np.max(M) * 1.2
    except:
        r_init = 0.2
        K_init = np.max(M) * 1.2

    a_init = min(max(a_init, 0.005), 0.2)
    e_init = min(max(e_init, 0.1), 1.0)
    gamma_init = min(max(gamma_init, 0.01), 0.5)
    r_init = min(max(r_init, 0.01), 1.0)
    K_init = max(K_init, np.max(M)*1.1)

    params = [r_init, a_init, e_init, gamma_init, K_init]

    print("\n--- Stable Holling-II Initial Guesses ---")
    for n, v in zip(["r", "a", "e", "gamma", "K"], params):
        print(f"{n:8s} = {v:.5f}")
    print("------------------------------------------\n")

    return np.log(params)



def fit_model_5param_only(M_obs, W_obs, t):
    """Calibrates 5 parameters: [r, a, e, gamma, K] (h is fixed)."""
    print(f"\n\n*** Running 5-Parameter Fit (h Fixed at {H_FIXED_VALUE:.4f}) ***")
    theta0_5 = lv_initial_guess(M_obs, W_obs, t)
    
    M_max = np.max(M_obs)
    
    # Bounds for [r, a, e, gamma, K]
    lower_p = [1e-3, 1e-7, 1e-3, 1e-3, M_max * 1.05]
    upper_p = [1.0, 1.0, 1.0, 0.5, M_max * 5.0]
    
    bounds = [(np.log(l), np.log(u)) for l, u in zip(lower_p, upper_p)]
    
    result = minimize(
        loss_function,
        theta0_5,
        args=(M_obs, W_obs, t), 
        method='L-BFGS-B', 
        bounds=bounds, 
        options={'maxiter': 10000, 'disp': True}
    )
    
    print("Optimization success:", result.success)
    print("Final loss:", result.fun)
    return result

if __name__ == "__main__":

    df = pd.read_csv("cleaned_climate_data.csv")
    
    df = df.dropna(subset=["wolves", "moose"])
    t = df["year"].values
    W_obs = df["wolves"].values
    M_obs = df["moose"].values
    

    result_5param = fit_model_5param_only(M_obs, W_obs, t)
    theta_opt_5 = result_5param.x

    params_5_est = np.exp(theta_opt_5)

    params_full = np.insert(params_5_est, 2, H_FIXED_VALUE)
    names_full = ["r", "a", "h", "e", "gamma", "K"]

    print("\n--- FINAL 5-PARAMETER FIT RESULTS ---")
    for n, v in zip(names_full, params_full):
        print(f"{n:8s} = {v:.5f}")
    print(f"Final Loss (RSS): {result_5param.fun:.4f}")


    pred = simulate(theta_opt_5, M_obs[0], W_obs[0], t)
    M_pred, W_pred = pred[:, 0], pred[:, 1]

    #plot 1
    plt.figure(figsize=(10, 6))
    ax1 = plt.gca()

    # Moose observed
    ax1.plot(t, M_obs, 'o', markersize=4, color="tab:green", label="Moose Observed")
    ax1.plot(t, M_pred, '-', color="green", linewidth=2.0, label="Moose Predicted")
    ax1.set_ylabel("Moose Population", color="green")
    ax1.tick_params(axis='y', labelcolor='green')
    ax1.grid(True, linestyle='--')

    # Wolf observed
    ax2 = ax1.twinx()
    ax2.plot(t, W_obs, 's', markersize=4, color="tab:red", label="Wolf Observed")
    ax2.plot(t, W_pred, '-', color="red", linewidth=2.0, label="Wolf Predicted")
    ax2.set_ylabel("Wolf Population", color="red")
    ax2.tick_params(axis='y', labelcolor='red')

    # Legend merge
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', ncol=2)

    plt.title(f"Plot 1: Moose & Wolf — Observed vs Model Fit (h={H_FIXED_VALUE:.4f})")
    plt.xlabel("Year")
    plt.show()

    #plot 2
    plt.figure(figsize=(10, 6))

    plt.plot(t, M_obs, 'o', markersize=4, color="tab:green", label="Moose Observed")
    plt.plot(t, M_pred, '-', linewidth=2.0, color="green", label="Moose Predicted")

    plt.ylabel("Moose Population")
    plt.xlabel("Year")
    plt.grid(True, linestyle='--')
    plt.legend()
    plt.title("Plot 2: Moose — Observed vs Model Predicted")
    plt.tight_layout()
    plt.show()

    #plot 3
    plt.figure(figsize=(10, 6))

    plt.plot(t, W_obs, 's', markersize=4, color="tab:red", label="Wolf Observed")
    plt.plot(t, W_pred, '-', linewidth=2.0, color="red", label="Wolf Predicted")

    plt.ylabel("Wolf Population")
    plt.xlabel("Year")
    plt.grid(True, linestyle='--')
    plt.legend()
    plt.title("Plot 3: Wolf — Observed vs Model Predicted")
    plt.tight_layout()
    plt.show()
