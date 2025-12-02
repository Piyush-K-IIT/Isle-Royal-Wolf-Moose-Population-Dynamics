# ==========================================================
# Reduced Holling-II predator–prey model diagnostics
# (b=0, mu=0, K free)
# ==========================================================
import numpy as np
import pandas as pd
from scipy.integrate import odeint
from scipy.optimize import minimize
from numpy.linalg import inv
import time

# -----------------------------
# 1) MODEL
# -----------------------------
def holling_type_ii_model(state, t, alpha, gamma, delta, K, a, h):
    x, y = state
    # Ensure biological realism
    x = max(x, 0.0)
    y = max(y, 0.0)
    denom = 1.0 + a * h * x
    denom = max(denom, 1e-12)
    interaction = (a * x * y) / denom
    dxdt = alpha * x * (1 - x / K) - interaction
    dydt = delta * interaction - gamma * y
    if np.isnan(dxdt) or np.isnan(dydt):
        return [0.0, 0.0]
    return [dxdt, dydt]


# -----------------------------
# 2) SIMULATION FUNCTION
# -----------------------------
def simulate_stacked(params, y0_local, time_vec):
    try:
        sol = odeint(holling_type_ii_model, y0_local, time_vec,
                     args=tuple(params), atol=1e-8, rtol=1e-8)
        if np.any(np.isnan(sol)) or np.any(np.isinf(sol)):
            return np.full((len(time_vec), 2), np.nan)
        return np.clip(sol, 0, None)
    except Exception:
        return np.full((len(time_vec), 2), np.nan)


# -----------------------------
# 3) OBJECTIVE FUNCTION
# -----------------------------
def objective(params, data_y, data_t, y0, weights):
    sim = simulate_stacked(params, y0, data_t)
    if np.isnan(sim).any():
        return 1e12
    resid = (sim - data_y) * weights
    return np.sum(resid ** 2)


# -----------------------------
# 4) DATA LOADING
# -----------------------------
df = pd.read_csv("wolf_moose_2019.csv")  # columns: year, wolves, moose
df = df.ffill().bfill()  # replace deprecated fillna(method=...)

time_vec = df["year"].to_numpy()
wolves = df["wolves"].to_numpy()
moose = df["moose"].to_numpy()

y_data = np.vstack([moose, wolves]).T
y0 = y_data[0, :]

# -----------------------------
# 5) PARAMETER SETUP
# -----------------------------
param_names = ['alpha', 'gamma', 'delta', 'K', 'a', 'h']
nominal = np.array([0.4, 0.2, 0.01, 2500.0, 0.01, 0.05], dtype=float)
bounds = [(1e-8, None)] * len(param_names)

# auto weighting to balance scales
pred_weight = np.mean(moose) / np.mean(wolves)
weights = np.array([1.0, pred_weight])
print(f"[info] automatic predator weighting based on data scales: {pred_weight:.2e}")


# -----------------------------
# 6) MULTI-START OPTIMIZATION
# -----------------------------
def multi_start_fit(n_starts=20):
    best_res = None
    best_fun = np.inf
    t0 = time.time()
    print("Running multi-start optimization (this may take a minute)...")
    for i in range(n_starts):
        x0 = nominal * np.random.uniform(0.5, 1.5, size=len(nominal))
        res = minimize(objective, x0, args=(y_data, time_vec, y0, weights),
                       bounds=bounds, method='L-BFGS-B',
                       options={'maxiter': 5000, 'ftol': 1e-12})
        print(f"Start {i}: success={res.success}, fun={res.fun:.3e}")
        if res.success and res.fun < best_fun:
            best_fun = res.fun
            best_res = res
    print(f"Multi-start done in {time.time() - t0:.1f}s")
    return best_res


best_res = multi_start_fit()
if best_res is None:
    raise RuntimeError("No successful fits found.")
best_params = best_res.x
print("\nBest objective:", best_res.fun)
print("Best parameters (name: value):")
for n, v in zip(param_names, best_params):
    print(f"  {n:<6}: {v:.6g}")


# -----------------------------
# 7) JACOBIAN & STANDARD ERRORS
# -----------------------------
def compute_jacobian(f, params, eps=1e-4):
    base = f(params)
    J = np.zeros((len(base), len(params)))
    for i in range(len(params)):
        dp = np.zeros_like(params)
        dp[i] = eps * max(1.0, abs(params[i]))
        J[:, i] = (f(params + dp) - f(params - dp)) / (2 * dp[i])
    return J


def f_flat(params):
    sim = simulate_stacked(params, y0, time_vec)
    return np.ravel((sim - y_data) * weights)


print("\nComputing Jacobian at best-fit (this may take time)...")
try:
    J = compute_jacobian(f_flat, best_params)
    JTJ = J.T @ J
    cov = inv(JTJ) * (best_res.fun / (len(J) - len(param_names)))
    se = np.sqrt(np.diag(cov))
    print("\nEstimated noise variance (sigma^2):", best_res.fun / (len(J) - len(param_names)))
    print("\nApprox parameter standard errors (SE):")
    for n, s in zip(param_names, se):
        print(f"  {n:<6}: SE = {s:.6g}")
except Exception:
    raise RuntimeError("Jacobian computation failed (integration unstable).")


corr_mat = np.corrcoef(cov / np.outer(se, se))
print("\nParameter correlation matrix (rounded):")
print(np.round(corr_mat, 3))


# -----------------------------
# 8) BOOTSTRAP UNCERTAINTY
# -----------------------------
def bootstrap_fit(n_boot=150):
    np.random.seed(0)
    results = []
    resid = y_data - simulate_stacked(best_params, y0, time_vec)
    for b in range(n_boot):
        boot_y = simulate_stacked(best_params, y0, time_vec) + resid[np.random.randint(0, len(resid), len(resid)), :]
        try:
            res = minimize(objective, best_params, args=(boot_y, time_vec, y0, weights),
                           bounds=bounds, method='L-BFGS-B', options={'maxiter': 3000})
            if res.success:
                results.append(res.x)
        except Exception:
            continue
        if (b + 1) % 10 == 0:
            print(f"Bootstrap {b + 1}/{n_boot}")
    return np.array(results)


print("\nRunning bootstrap (this may take a while) ...")
t0 = time.time()
boot = bootstrap_fit(150)
print(f"Bootstrap finished in {time.time() - t0:.1f}s with {len(boot)} successful fits")

if len(boot) > 0:
    for i, name in enumerate(param_names):
        med = np.median(boot[:, i])
        q25, q75 = np.percentile(boot[:, i], [25, 75])
        print(f"  {name:<6}: median={med:.6g}, 25%={q25:.6g}, 75%={q75:.6g}")


# -----------------------------
# 9) PROFILE LIKELIHOOD
# -----------------------------
def profile_likelihood(param_index, span_factor=0.4, n_points=8):
    base = best_params.copy()
    p_val = base[param_index]
    sweep = np.linspace(p_val * (1 - span_factor),
                        p_val * (1 + span_factor), n_points)
    prof = []
    for val in sweep:
        trial = base.copy()
        trial[param_index] = val
        res = minimize(objective, trial, args=(y_data, time_vec, y0, weights),
                       bounds=bounds, method='L-BFGS-B',
                       options={'maxiter': 2000})
        prof.append((val, res.fun))
    return np.array(prof)


print("\nRunning coarse profile likelihood for selected parameters (somewhat expensive)...")
for i in [0, 1]:  # alpha, gamma
    prof = profile_likelihood(i)
    print(f"Profile for param {param_names[i]} done.")


# -----------------------------
# 10) FINAL DIAGNOSTIC SUMMARY
# -----------------------------
print("\n--- Diagnostics Summary ---")
print("Best-fit objective:", best_res.fun)
print("Best-fit parameters:")
for n, v in zip(param_names, best_params):
    print(f"  {n:<6}: {v:.6g}")
print("Estimated sigma^2 (residual variance):", best_res.fun / (len(J) - len(param_names)))
print("Approx SEs:")
for n, s in zip(param_names, se):
    print(f"  {n:<6}: {s:.6g}")
print("Bootstrap medians & 25-75 percentiles printed above.")
