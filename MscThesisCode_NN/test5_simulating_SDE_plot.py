import numpy as np
import matplotlib.pyplot as plt

# Parameters for the SDE
theta = 2.0
sigma = 0.8
num_hours = 48
delta_t = 1.0  # time step

# Initial wind power production
initial_production = 0.1
production = np.zeros(num_hours + 1)
production[0] = initial_production

# Point forecast (used as the mean reverting term in the SDE)
point_forecast = np.sin(np.linspace(0, 4 * np.pi, num_hours + 1)) ** 2

# Simulate the SDE
for t in range(num_hours):
    drift = theta * (point_forecast[t] - production[t]) * delta_t
    diffusion = sigma * production[t] * (1 - production[t]) * np.sqrt(delta_t) * np.random.normal()
    production[t + 1] = production[t] + drift + diffusion
    # Ensure the production is bounded between 0 and 1
    production[t + 1] = max(0, min(1, production[t + 1]))

# Calculate quantiles for the shaded areas
quantiles = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
forecast_quantiles = np.zeros((len(quantiles), num_hours + 1))

for i, q in enumerate(quantiles):
    forecast_quantiles[i, :] = np.percentile(production, q * 100, axis=0)

# Plotting
plt.figure(figsize=(10, 5))

# Plot the shaded areas
for i, q in enumerate(reversed(quantiles)):
    plt.fill_between(range(num_hours + 1), np.maximum(0,point_forecast - forecast_quantiles[i, :]), point_forecast + forecast_quantiles[i, :], 
                     color='red', alpha=0.1 + 0.1 * (1 - q))

# Plot the mean line
plt.plot(range(num_hours + 1), point_forecast, 'k-', linewidth=2, marker='o', markerfacecolor='white')

# Adding labels and title
plt.xlabel('Forecast horizon [hours]')
plt.ylabel('Normalized wind power production')
plt.title('SDE-based probabilistic short-term forecast')

# Adding legend for quantiles
from matplotlib.lines import Line2D
legend_elements = [Line2D([0], [0], color='red', lw=4, alpha=0.1 + 0.1 * (1 - q), label=f'{int(q*100)}%') for q in quantiles]
plt.legend(handles=legend_elements, title='Quantiles')

plt.grid(True)
plt.show()