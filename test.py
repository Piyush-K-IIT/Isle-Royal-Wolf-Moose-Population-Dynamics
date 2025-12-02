import numpy as np
import pandas as pd
from scipy.integrate import odeint
from scipy.optimize import least_squares
import matplotlib.pyplot as plt

# -------------------------------------------------
# LOAD DATA
# -------------------------------------------------
df = pd.read_csv("cleaned_climate_data.csv")

t = df["year"].values
M_obs = df["moose"].values
W_obs = df["wolves"].values

# Climate variables
T_summer = df["July-Sept (temp, F)"].values
T_spring = df["Apr-May (temp, F)"].values
T_winter = df["Jan-Feb (temp, F)"].values
precip = df["May-Aug (precip, inches)"].values
snow = df["snow.depth (cm)"].values

# Normalize for stability
def z(x): return (x - x.mean()) / x.std()

C = np.vstack([
    z(T_summer),    # 0
    z(T_spring),    # 1
    z(T_winter),    # 2
    z(precip),      # 3
    z(snow)         # 4
]).T


# -------------------------------------------------
# CLIMATE DEPENDENT PARAMETER FUNCTIONS
# -------------------------------------------------
def get_params(theta, C):
    """
    theta parameter list:
    0 r0, 1 β_r1 (summer), 2 β_r2 (precip)
    3 K0, 4 β_K1 (spring), 5 β_K2 (precip)
    6 a0, 7 β_a1 (snow), 8 β_a2 (winter)
    9 h0, 10 β_h1 (snow)
    11 m0, 12 β_m1 (winter), 13 β_m2 (summer)
    14 e (efficiency)
    """

    T_summer, T_spring, T_winter, precip, snow = C.T

    r = theta[0] + theta[1] * T_summer + theta[2] * precip
    K = theta[3] + theta[4] * T_spring + theta[5] * precip

    a = theta[6] + theta[7] * snow + theta[8] * T_winter
    h = theta[9] + theta[10] * snow

    m = theta[11] + theta[12] * T_winter + theta[13] * T_summer

    e = theta[14]

    return r, K, a, h, m, e


# -------------------------------------------------
# ODE SYSTEM
# -------------------------------------------------
def system(state, t, theta, C, years):
    M, W = state
    idx = np.argmin(np.abs(years - t))

    r, K, a, h, m, e = get_params(theta, C)

    r, K, a, h, m = r[idx], K[idx], a[idx], h[idx], m[idx]

    # Holling type II
    F = (a * M * W) / (1 + a * h * M)

    dMdt = r * M * (1 - M / K) - F
    dWdt = e * F - m * W

    return [dMdt, dWdt]


# -------------------------------------------------
# SIMULATION
# -------------------------------------------------
def simulate(theta):
    M0, W0 = M_obs[0], W_obs[0]
    sol = odeint(system, [M0, W0], t, args=(theta, C, t))
    return sol[:,0], sol[:,1]


# -------------------------------------------------
# RESIDUALS FOR FITTING
# -------------------------------------------------
def residuals(theta):
    M_sim, W_sim = simulate(theta)
    return np.concatenate([(M_sim - M_obs), (W_sim - W_obs)])


# -------------------------------------------------
# INITIAL GUESS FOR 15 PARAMETERS
# -------------------------------------------------
theta0 = np.array([
    0.3, 0.1, 0.1,      # r0, β_r1, β_r2
    3000, 500, 500,     # K0, β_K1, β_K2
    0.002, 0.001, 0.001, # a0, β_a1, β_a2
    0.1, 0.01,           # h0, β_h1
    0.05, 0.02, 0.02,    # m0, β_m1, β_m2
    0.1                  # e
])

result = least_squares(residuals, theta0)
theta_hat = result.x

print("Fitted parameters:")
print(theta_hat)


# -------------------------------------------------
# PLOT FIT
# -------------------------------------------------
M_sim, W_sim = simulate(theta_hat)

fig, ax1 = plt.subplots(figsize=(12,6))

# Moose
ax1.plot(t, M_obs, "go-", label="Observed Moose")
ax1.plot(t, M_sim, "g--", label="Simulated Moose")
ax1.set_xlabel("Year")
ax1.set_ylabel("Moose", color="green")
ax1.tick_params(axis='y', labelcolor="green")

# Wolves on second axis
ax2 = ax1.twinx()
ax2.plot(t, W_obs, "ro-", label="Observed Wolves")
ax2.plot(t, W_sim, "r--", label="Simulated Wolves")
ax2.set_ylabel("Wolves", color="red")
ax2.tick_params(axis='y', labelcolor="red")

fig.tight_layout()
plt.grid(True)
plt.show()