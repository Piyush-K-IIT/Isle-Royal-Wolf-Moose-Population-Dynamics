import numpy as np
import pandas as pd
from scipy.integrate import odeint
from scipy.optimize import minimize
import matplotlib.pyplot as plt

## 1. Define the Logistic Prey Model
def logistic_prey_model(state, t, alpha, beta, gamma, delta, K):
    """
    Defines the Lotka-Volterra equations with logistic growth for the prey.
    K is the carrying capacity for the prey.
    """
    x, y = state  # Unpack prey (x) and predator (y) populations
    
    # Logistic growth equation for prey
    dxdt = alpha * x * (1 - x / K) - beta * x * y
    
    # Standard equation for predator
    dydt = delta * x * y - gamma * y
    
    return [dxdt, dydt]

## 2. Load and Prepare Your Data
try:
    df = pd.read_csv("wolf_moose_2019.csv")

    # Check if required columns exist
    if 'moose' not in df.columns or 'wolves' not in df.columns or 'year' not in df.columns:
        print("Error: CSV must contain 'year', 'moose', and 'wolves' columns.")
        raise FileNotFoundError 

    # Handle potential missing data
    df = df.fillna(method='ffill').fillna(method='bfill')

    # We assume "moose" is prey (x) and "wolves" is predator (y)
    prey_data = df['moose'].values
    predator_data = df['wolves'].values
    time_raw = df['year'].values

except FileNotFoundError:
    print("--- File 'wolf_moose_2019.csv' not found or invalid. ---")
    print("Using fallback sample data for demonstration.")
    # Fallback to sample data if file isn't found
    time_raw = np.arange(1959, 2020)
    prey_data = 1000 + 500 * np.sin((time_raw - 1959) * 0.1) + np.random.randn(len(time_raw)) * 50
    predator_data = 50 + 20 * np.sin((time_raw - 1959) * 0.1 - 1.5) + np.random.randn(len(time_raw)) * 5
    prey_data = np.maximum(0, prey_data)
    predator_data = np.maximum(0, predator_data)


# --- Data Preparation ---
# Normalize time array to start from 0 for the solver
time = time_raw - time_raw[0]

# Set the initial conditions (y0) for the model using the first data point
y0 = [prey_data[0], predator_data[0]]

# Find the maximum observed prey population to inform the guess for K
max_prey = np.max(prey_data)


## 3. Define the Error Function (Objective Function)
def calculate_error(params):
    """Calculates the sum of squared errors (SSE) for the logistic model."""
    # Unpack all 5 parameters
    alpha, beta, gamma, delta, K = params
    
    # Check for K=0 to avoid division by zero
    if K <= 0:
        return np.inf  # Return a large error if K is invalid
    
    # Run the ODE solver with the current parameter guess
    model_output = odeint(logistic_prey_model, y0, time, args=tuple(params))
    
    # Separate the model's prey and predator outputs
    prey_model = model_output[:, 0]
    predator_model = model_output[:, 1]
    
    # Calculate the SSE
    error = np.sum((prey_data - prey_model)**2) + np.sum((predator_data - predator_model)**2)
    
    return error

## 4. Run the Optimization
# We now have 5 parameters to guess: [alpha, beta, gamma, delta, K]
# We'll guess K is a bit higher than the max observed moose population
initial_guess = [0.2, 0.001, 0.1, 0.0001, max_prey + 500] 

# Add bounds for all 5 parameters
param_bounds = [(0, None), (0, None), (0, None), (0, None), (0, None)]

# Run the optimizer
result = minimize(calculate_error, initial_guess, method='L-BFGS-B', bounds=param_bounds)

# Get the best-fit parameters
best_params = result.x

## 5. Print and Visualize the Results
print("--- Fitting Results (Logistic Prey Model) ---")
print(f"Alpha (prey growth): {best_params[0]:.4f}")
print(f"Beta (predation rate): {best_params[1]:.6f}")
print(f"Gamma (predator death): {best_params[2]:.4f}")
print(f"Delta (predator efficiency): {best_params[3]:.6f}")
print(f"K (carrying capacity): {best_params[4]:.2f}")
print(f"\nFinal Sum of Squared Errors: {result.fun:.2f}")

# Simulate the model with the *best* parameters
fitted_solution = odeint(logistic_prey_model, y0, time, args=tuple(best_params))
fitted_prey = fitted_solution[:, 0]
fitted_predator = fitted_solution[:, 1]

# --- Plotting (same as before) ---
fig, ax1 = plt.subplots(figsize=(12, 7))

# --- Axis 1: Moose (Prey) ---
color_prey = 'cornflowerblue'
ax1.set_xlabel('Year', fontsize=12)
ax1.set_ylabel('Moose Population (Prey)', color=color_prey, fontsize=12)
ln1 = ax1.plot(time_raw, prey_data, 'o', label='Moose Data (Prey)', color=color_prey, alpha=0.7)
ln2 = ax1.plot(time_raw, fitted_prey, '-', label='Fitted Moose Model', color='blue', linewidth=2)
ax1.tick_params(axis='y', labelcolor=color_prey)
ax1.grid(True, linestyle='--', alpha=0.6)

# --- Axis 2: Wolves (Predator) ---
ax2 = ax1.twinx()  
color_pred = 'indianred'
ax2.set_ylabel('Wolf Population (Predator)', color=color_pred, fontsize=12)
ln3 = ax2.plot(time_raw, predator_data, 's', label='Wolf Data (Predator)', color=color_pred, alpha=0.7)
ln4 = ax2.plot(time_raw, fitted_predator, '-', label='Fitted Wolf Model', color='red', linewidth=2)
ax2.tick_params(axis='y', labelcolor=color_pred)

# --- Final Touches ---
plt.title('Logistic Prey Model Fit to Isle Royale Data', fontsize=16)
lns = ln1 + ln2 + ln3 + ln4
labs = [l.get_label() for l in lns]
ax1.legend(lns, labs, loc='upper left')
fig.tight_layout()
plt.show()