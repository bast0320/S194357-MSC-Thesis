import numpy as np
from scipy.optimize import minimize

# Assume X is your time series data
# For demonstration, we'll generate synthetic data
np.random.seed(42)
n = 1000
dt = 0.01
a, b, sigma = 0.1, -0.2, 0.3
X = np.zeros(n)
X[0] = 1

for t in range(1, n):
    X[t] = X[t-1] + (a * X[t-1] + b * X[t-2]) * dt + sigma * np.sqrt(dt) * np.random.normal()


import pandas as pd

actuals = pd.read_pickle("loaded_variables/actuals_hourly_DK1_fixed.pkl")
ensembles_DK1_onshorewindpower = pd.read_pickle("loaded_variables/ensembles_wind_onshore_DK1_fixed.pkl")
ensembles_DK1_offshorewindpower = pd.read_pickle("loaded_variables/ensembles_wind_offshore_DK1_fixed.pkl")
ensembles_DK1_solarpower = pd.read_pickle("loaded_variables/ensembles_solar_DK1_fixed_v2.pkl")

Y_DK1_OnshoreWindPower = actuals["OnshoreWindPower"]
Y_DK1_OffshoreWindPower = actuals["OffshoreWindPower"]
Y_DK1_SolarPower = actuals["SolarPower"]

X = Y_DK1_OnshoreWindPower.values[:1000]

# Define the likelihood function
def sde_log_likelihood(params, X, dt):
    a, b, sigma = params
    n = len(X)
    log_likelihood = 0
    for t in range(1, n):
        mu = X[t-1] + (a * X[t-1] + b * X[t-2]) * dt
        variance = sigma**2 * dt
        log_likelihood += -0.5 * np.log(2 * np.pi * variance) - 0.5 * ((X[t] - mu)**2 / variance)
    return -log_likelihood

# Initial guess for the parameters
initial_guess = [0.1, -0.1, 0.1]

# Perform the optimization
result = minimize(sde_log_likelihood, initial_guess, args=(X, dt), method='L-BFGS-B',
                  bounds=[(-10, 10), (-10, 10), (0.01, 1000)])

# Extract the estimated parameters
a_est, b_est, sigma_est = result.x

print(f"Estimated parameters: a = {a_est}, b = {b_est}, sigma = {sigma_est}")

# You can now use a_est, b_est, and sigma_est for further analysis


# Function to simulate paths from the SDE
def simulate_sde(a, b, sigma, X0, n, dt, num_paths):
    paths = np.zeros((num_paths, n))
    for i in range(num_paths):
        paths[i, 0] = X0
        for t in range(1, n):
            # paths[i, t] = paths[i, t-1] + (a * paths[i, t-1] + b * paths[i, t-2]) * dt + sigma * np.sqrt(dt) * np.random.normal()
            paths[i, t] = X[t-1] + (a * X[t-1] + b * X[t-2]) * dt + sigma * np.sqrt(dt) * np.random.normal()
    return paths

# Simulate 3 paths
num_paths = 3
simulated_paths = simulate_sde(a_est, b_est, sigma_est, X[0], n, dt, num_paths)

import matplotlib.pyplot as plt

# Use a built-in style as a base
plt.style.use('seaborn-v0_8-paper')

# Customize the style for publication
custom_params = {
    'figure.figsize': (10, 6),
    'axes.titlesize': 18,
    'axes.labelsize': 16,
    'lines.linewidth': 2,
    'lines.markersize': 6,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    'legend.fontsize': 14,
    'font.family': 'serif',
    'font.serif': 'Times New Roman',
    'savefig.dpi': 300, # For high-quality output
}

# Update matplotlib settings
plt.rcParams.update(custom_params)


# Plot original data and simulated paths
plt.figure(figsize=(12, 6))
plt.plot(X, label='Original Data', color='black')
for i in range(num_paths):
    plt.plot(simulated_paths[i], label=f'Simulated Path {i+1}', alpha=0.5,linewidth = 0.5, color='blue')
plt.xlabel('Time')
plt.ylabel('X')
plt.legend()
plt.title('Original Data and Simulated Paths from SDE')
plt.show()