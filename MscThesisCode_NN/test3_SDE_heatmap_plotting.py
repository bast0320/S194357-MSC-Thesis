import numpy as np
import pandas as pd
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde


# Load your data
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
result = minimize(sde_log_likelihood, initial_guess, args=(X, 0.01), method='L-BFGS-B', bounds=[(-10, 10), (-10, 10), (0.01, 1000)])
a_est, b_est, sigma_est = result.x

print(f"Estimated parameters: a = {a_est}, b = {b_est}, sigma = {sigma_est}")

# Function to simulate paths from the SDE
def simulate_sde(a, b, sigma, X0, n, dt, num_paths):
    paths = np.zeros((num_paths, n))
    for i in range(num_paths):
        paths[i, 0] = X0
        for t in range(1, n):
            # paths[i, t] = paths[i, t-1] + (a * paths[i, t-1] + b * paths[i, t-2]) * dt + sigma * np.sqrt(dt) * np.random.normal()
            paths[i, t] = X[t-1] + (a * X[t-1] + b * X[t-2]) * dt + sigma * np.sqrt(dt) * np.random.normal()

    return paths

# Simulate multiple paths
num_paths = 2000
simulated_paths = simulate_sde(a_est, b_est, sigma_est, X[0], len(X), 0.01, num_paths)

# # Plotting
# plt.figure(figsize=(12, 6))

# # Plot the 2D histogram
# plt.hexbin(np.tile(np.arange(len(X)), num_paths), simulated_paths.flatten(), gridsize=200, cmap='hot', bins='log')

# plt.plot(np.arange(len(X)), X, label='Original Data', color='black', linewidth=2)
# plt.colorbar(label='Log Density')
# plt.xlabel('Time')
# plt.ylabel('X')
# plt.legend()
# plt.title('Original Data and Simulated Paths Density from SDE')
# plt.show()


# Calculate the KDE for each time step
kde_values = np.zeros((len(X), 1000))
x_grid = np.linspace(np.min(simulated_paths), np.max(simulated_paths), 1000)

for t in range(len(X)):
    kde = gaussian_kde(simulated_paths[:, t])
    kde_values[t, :] = kde(x_grid)

# fix kde_values between 0 and 1
kde_values = (kde_values - np.min(kde_values)) / (np.max(kde_values) - np.min(kde_values))

# Plotting
plt.figure(figsize=(10, 6))
plt.imshow(kde_values.T, aspect='auto', origin='lower', extent=[0, len(X), np.min(simulated_paths), np.max(simulated_paths)], cmap='hot_r')
plt.plot(np.arange(len(X)), X, label='Original Data', color='blue', linewidth=2)
plt.colorbar(label='Probability')
plt.xlabel('Time')
plt.ylabel('X')
plt.legend()
plt.title('Original Data and Simulated Paths Density from SDE')
plt.savefig("figures/heatmap_sde_plotting.pdf")
plt.show()