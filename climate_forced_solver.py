"""
climate_forced_solver.py

End-to-end climate-forced Holling-II solver.

Main functions:
- predict_parameters_from_models(...)
- build_parameter_interpolants(...)
- integrate_hollingII(...)
- climate_forced_simulation(...)

Requirements:
- numpy, scipy, pandas
- optional: torch, gpytorch (if using gpytorch GP models)
"""

import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d

# Optional imports for GP prediction; we catch absence gracefully
try:
    import torch
    import gpytorch
    _HAS_GP = True
except Exception:
    _HAS_GP = False


# -------------------------
# Utility: uniform interface to prediction
# -------------------------
def _is_gpytorch_model(model_and_lik):
    """Detect gpytorch ExactGP + likelihood tuple or object pair."""
    if not _HAS_GP:
        return False
    if model_and_lik is None:
        return False
    # Accept (model, likelihood) tuple or object with 'likelihood' attribute
    if isinstance(model_and_lik, (list, tuple)) and len(model_and_lik) == 2:
        return True
    if hasattr(model_and_lik, "likelihood"):
        return True
    return False


def predict_with_model(model_or_pair, X_np, return_std=False, n_samples=0, device="cpu"):
    """
    Predict mean (and optionally std or samples) for inputs X_np (2D: n_times x n_features).

    model_or_pair:
        - If gpytorch: either (model, likelihood) or object with .likelihood
        - Else: any python object with .predict(X) or callable func(X) returning array

    return_std: if True, return predictive std (if supported)
    n_samples: if >0 and gpytorch available, return 'n_samples' sampled predictions per input (n_samples x n_times)
    """
    X_np = np.asarray(X_np)
    if X_np.ndim == 1:
        X_np = X_np.reshape(1, -1)
    n = X_np.shape[0]

    # Handle gpytorch
    if _is_gpytorch_model(model_or_pair):
        if not _HAS_GP:
            raise RuntimeError("gpytorch not available in environment.")
        # normalize into (model, likelihood)
        if isinstance(model_or_pair, (list, tuple)):
            model, likelihood = model_or_pair
        else:
            model = model_or_pair
            likelihood = model_or_pair.likelihood

        model.eval(); likelihood.eval()
        # convert to torch
        X_t = torch.from_numpy(X_np).float().to(device)

        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            # predictive distribution (multivariate normal)
            preds = likelihood(model(X_t))
            mean = preds.mean.cpu().numpy()
            if return_std:
                # gpytorch gives covariance; convert to std per point (assuming diagonal)
                # We use predictive variance per point (preds.variance)
                try:
                    var = preds.variance.cpu().numpy()
                    std = np.sqrt(var)
                except Exception:
                    # fallback using full covariance matrix if needed
                    cov = preds.covariance_matrix.cpu().numpy()
                    std = np.sqrt(np.diag(cov))
                if n_samples > 0:
                    # draw independent samples per input from Normal(mean, std)
                    rng = np.random.default_rng()
                    samples = rng.normal(loc=mean, scale=std, size=(n_samples, n))
                    return mean, std, samples
                return mean, std
            else:
                if n_samples > 0:
                    # draw samples from predictive marginal normals (independent per time point)
                    var = preds.variance.cpu().numpy()
                    std = np.sqrt(var)
                    rng = np.random.default_rng()
                    samples = rng.normal(loc=mean, scale=std, size=(n_samples, n))
                    return mean, samples
                return mean

    # Non-gp model: expect it to have .predict or be callable
    # If it is a tuple like (model, scaler) or (callable, None) just treat first element as callable
    model = model_or_pair
    if isinstance(model_or_pair, (list, tuple)):
        model = model_or_pair[0]

    # If model is callable, call it. If it has .predict, use it.
    if hasattr(model, "predict"):
        mean = model.predict(X_np)
    elif callable(model):
        mean = model(X_np)
    else:
        raise RuntimeError("model_or_pair must be a gpytorch pair, sklearn-like model with .predict or a callable.")

    mean = np.asarray(mean)
    # Guarantee shape (n,)
    if mean.ndim == 2 and mean.shape[1] == 1:
        mean = mean[:, 0]
    if mean.size == 1 and n == 1:
        mean = mean.reshape(1)

    if return_std:
        # Try to get std via attribute or default to zeros (user didn't provide uncertainty)
        std = None
        if hasattr(model, "predict_std"):
            std = np.asarray(model.predict_std(X_np))
        elif hasattr(model, "predict_proba"):
            std = np.zeros_like(mean)
        else:
            std = np.zeros_like(mean)
        if n_samples > 0:
            rng = np.random.default_rng()
            samples = rng.normal(loc=mean, scale=std, size=(n_samples, n))
            return mean, std, samples
        return mean, std

    return mean


# -------------------------
# Build continuous interpolants for predicted parameters
# -------------------------
def build_parameter_interpolants(times, r_pred, a_pred, K_pred, kind="linear", fill_value="extrapolate"):
    """
    Build interpolation functions r(t), a(t), K(t) from predictions at discrete times.

    times: 1D array of timepoints (floats)
    r_pred, a_pred, K_pred: arrays of same length
    kind: interpolation kind (linear, cubic - be careful with cubic small sample sizes)
    """
    r_fun = interp1d(times, np.asarray(r_pred), kind=kind, axis=0, fill_value=fill_value, assume_sorted=True)
    a_fun = interp1d(times, np.asarray(a_pred), kind=kind, axis=0, fill_value=fill_value, assume_sorted=True)
    K_fun = interp1d(times, np.asarray(K_pred), kind=kind, axis=0, fill_value=fill_value, assume_sorted=True)
    return r_fun, a_fun, K_fun


# -------------------------
# Holling-II ODE right-hand-side
# -------------------------
def hollingII_rhs(t, y, r_fun, a_fun, K_fun, h=0.005, beta=0.1, d=0.2):
    """
    Right-hand side used by solve_ivp.
    y = [M, W]
    r_fun, a_fun, K_fun: callables of time t returning scalars
    h, beta, d: constants (handling time, assimilation efficiency, wolf mortality)
    """
    M, W = y
    # numerical safety
    M = max(M, 1e-6)
    W = max(W, 1e-6)

    r = float(np.squeeze(r_fun(t)))
    a = float(np.squeeze(a_fun(t)))
    K = float(np.squeeze(K_fun(t)))

    predation = (a * M * W) / (1.0 + h * a * M)
    dMdt = r * M * (1.0 - M / K) - predation
    dWdt = beta * predation - d * W
    return [dMdt, dWdt]


# -------------------------
# Main integration routine
# -------------------------
def integrate_hollingII(t_span, t_eval, M0, W0, r_fun, a_fun, K_fun,
                        h=0.005, beta=0.1, d=0.2, rtol=1e-6, atol=1e-8):
    """
    Integrate the Holling-II ODE over t_span = (t0, t1) and at times t_eval.
    Returns an object similar to solve_ivp result with .y (2 x len(t_eval)) and .t
    """
    sol = solve_ivp(fun=lambda t, y: hollingII_rhs(t, y, r_fun, a_fun, K_fun, h=h, beta=beta, d=d),
                    t_span=t_span, y0=[M0, W0], t_eval=t_eval, method="RK45", rtol=rtol, atol=atol)
    if not sol.success:
        raise RuntimeError("ODE integration failed: " + str(sol.message))
    return sol


# -------------------------
# Orchestrator: predict parameters -> integrate -> optionally ensemble
# -------------------------
def climate_forced_simulation(climate_df,
                              feature_columns,
                              time_column,
                              model_rx, model_a, model_K,
                              scaler=None,
                              t0_col=None,
                              M0=None, W0=None,
                              integration_dt=0.1,
                              ensemble_size=0,
                              gp_device="cpu",
                              gp_sample_seed=None,
                              constants=None,
                              interp_kind="linear"):
    """
    Full pipeline:
      1. Takes climate_df and feature_columns (list) to build X.
      2. Uses models (gp or callable) to predict r,a,K at discrete times.
      3. Optionally draws samples from predictive distributions (if models support).
      4. Builds interpolants and integrates Holling-II via RK45.
      5. Returns deterministic result or an ensemble of runs.

    Parameters
    - climate_df: pandas.DataFrame with climate/time columns
    - feature_columns: list of column names to use as covariates for GP/regressor
    - time_column: name of time column (numeric/float)
    - model_rx, model_a, model_K: each either (gpytorch_model, likelihood) or sklearn-like regressor or a callable
    - scaler: optional scaler used to preprocess X (should have .transform). If None, raw features used.
    - M0, W0: initial conditions. If None, set from first row of dataset (requires 'moose' and 'wolves' columns)
    - integration_dt: step size for output grid (years) — we'll create t_eval from min(time) to max(time)
    - ensemble_size: if >0 and models support predictive sampling, produce that many parameter-sampled integrations
    - gp_device: device for torch (cpu/gpu) if using gpytorch
    - constants: dict to override h, beta, d
    - interp_kind: interpolation kind for parameter functions
    """

    # Constants defaults
    const = {"h": 0.005, "beta": 0.1, "d": 0.2}
    if constants:
        const.update(constants)

    # Validate and prepare inputs
    if isinstance(climate_df, pd.DataFrame):
        df = climate_df.copy()
    else:
        df = pd.DataFrame(climate_df)

    times = np.asarray(df[time_column], dtype=float)
    # Sort by time if needed
    order = np.argsort(times)
    times = times[order]
    X_raw = np.asarray(df.loc[:, feature_columns].values)[order, :]

    if scaler is not None:
        X = scaler.transform(X_raw)
    else:
        X = X_raw

    # Predict mean parameter trajectories at discrete times
    # Each predict_with_model returns shape (n_times,)
    r_mean = predict_with_model(model_rx, X, return_std=False)  # shape (n,)
    a_mean = predict_with_model(model_a, X, return_std=False)
    K_mean = predict_with_model(model_K, X, return_std=False)

    # Determine M0 and W0
    if M0 is None or W0 is None:
        # try from df columns 'moose' and 'wolves'
        if "moose" in df.columns and "wolves" in df.columns:
            M0 = float(df.loc[:, "moose"].values[order][0]) if M0 is None else M0
            W0 = float(df.loc[:, "wolves"].values[order][0]) if W0 is None else W0
        else:
            raise RuntimeError("Initial states M0 and W0 not provided and data doesn't contain 'moose'/'wolves'.")

    # Build time fine grid for integration
    t_min, t_max = float(times[0]), float(times[-1])
    # t_eval: dense times for output
    t_eval = np.arange(t_min, t_max + integration_dt, integration_dt)

    # If ensemble requested and GP models available, draw samples for parameters
    ensemble_results = []
    param_ensembles = []
    if ensemble_size > 0:
        # Try to draw per-time independent samples from the predictive marginals
        # For gpytorch models this will use mean+var at each timepoint
        rng = np.random.default_rng(gp_sample_seed)
        # For each parameter, attempt predictive std and samples
        # predict_with_model(model, X, return_std=True, n_samples=ensemble_size)
        # but our predict_with_model returns (mean, std, samples) if requested
        try:
            r_mean2, r_std, r_samps = predict_with_model(model_rx, X, return_std=True, n_samples=ensemble_size, device=gp_device)
            a_mean2, a_std, a_samps = predict_with_model(model_a, X, return_std=True, n_samples=ensemble_size, device=gp_device)
            K_mean2, K_std, K_samps = predict_with_model(model_K, X, return_std=True, n_samples=ensemble_size, device=gp_device)
            # shapes: r_samps (ensemble_size, n_times)
            for s in range(ensemble_size):
                r_s = r_samps[s, :]
                a_s = a_samps[s, :]
                K_s = K_samps[s, :]
                # interpolants
                r_fun, a_fun, K_fun = build_parameter_interpolants(times, r_s, a_s, K_s, kind=interp_kind)
                sol = integrate_hollingII((t_min, t_max), t_eval, M0, W0, r_fun, a_fun, K_fun,
                                         h=const["h"], beta=const["beta"], d=const["d"])
                ensemble_results.append(sol)
                param_ensembles.append((r_s, a_s, K_s))
        except Exception as e:
            # Fallback: cannot sample predictive distributions; return deterministic only
            print("Warning: ensemble sampling failed or not supported by provided models:", e)
            ensemble_size = 0

    # Deterministic integration using means
    r_fun_mean, a_fun_mean, K_fun_mean = build_parameter_interpolants(times, r_mean, a_mean, K_mean, kind=interp_kind)
    sol_mean = integrate_hollingII((t_min, t_max), t_eval, M0, W0, r_fun_mean, a_fun_mean, K_fun_mean,
                                  h=const["h"], beta=const["beta"], d=const["d"])

    output = {
        "t_eval": t_eval,
        "solution_mean": sol_mean,               # solve_ivp result for mean-case
        "r_mean": r_mean,
        "a_mean": a_mean,
        "K_mean": K_mean,
        "times_discrete": times,
        "ensemble_solutions": ensemble_results,  # list of solve_ivp results if ensemble_size>0
        "ensemble_params": param_ensembles
    }
    return output


# -------------------------
# Example usage pattern
# -------------------------
if __name__ == "__main__":
    # This section demonstrates how to call the pipeline.
    # Replace with your actual objects (df, scaler, gp models, etc.)

    # Example: minimal synthetic demonstration with dummy regressors
    import sklearn.linear_model as lm
    # Prepare fake climate df
    years = np.arange(1980, 2000)
    n = len(years)
    df_demo = pd.DataFrame({
        "year": years,
        "temp_JanFeb": np.linspace(-5, 3, n),
        "snow_depth": np.linspace(50, 30, n),
        "moose": np.linspace(1500, 1800, n).astype(float),
        "wolves": np.linspace(40, 25, n).astype(float)
    })

    # Simple regressors that map climate to plausible parameter scales
    def make_simple_model(factor, intercept):
        return lambda X: (intercept + (X[:, 0] * factor)).ravel()

    model_rx = make_simple_model(-0.01, 0.25)   # r decreases slightly with temp in this fake example
    model_a  = make_simple_model(1e-4, 5e-4)
    model_K  = make_simple_model( -5.0, 2000.0)

    # Call simulation (deterministic)
    out = climate_forced_simulation(
        climate_df=df_demo,
        feature_columns=["temp_JanFeb", "snow_depth"],
        time_column="year",
        model_rx=model_rx,
        model_a=model_a,
        model_K=model_K,
        scaler=None,
        M0=None, W0=None,  # will take from df_demo's first row
        integration_dt=0.25,
        ensemble_size=0
    )

    # Print/plot basic results
    t = out["t_eval"]
    M = out["solution_mean"].y[0]
    W = out["solution_mean"].y[1]
    print("t (len):", len(t))
    print("M final:", M[-1], "W final:", W[-1])
