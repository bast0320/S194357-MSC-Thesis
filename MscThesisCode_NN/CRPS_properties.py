import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta

def beta_cdf(x, a, b):
    return beta.cdf(x, a, b)

def crps_integrand(x, y, a, b):
    F = beta_cdf(x, a, b)
    return np.where(x <= y, F**2, (F - 1)**2)

# Generate data
x = np.linspace(0, 1, 1000)
y = 0.264  # The median of Beta(2,5)

# Calculate CRPS integrand for true and wrong models
true_integrand = crps_integrand(x, y, 2, 5)
wrong_integrand = crps_integrand(x, y, 1, 5)

# Create the plot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

# True model plot
ax1.plot(x, true_integrand, 'k', linewidth=2)
ax1.fill_between(x, true_integrand, color='blue', alpha=0.5)
ax1.plot(x, beta_cdf(x, 2, 5)**2, 'gray', linestyle='--', linewidth=2)
ax1.plot(x, (beta_cdf(x, 2, 5)-1)**2, 'gray', linestyle='--', linewidth=2)
ax1.axvline(x=y, color='red', linestyle='--')
ax1.plot(y, 0, 'go', markersize=10)
ax1.set_xlim(0, 1)
ax1.set_ylim(0, 1)
ax1.set_title('True model', fontsize=16)
ax1.set_xlabel('y', fontsize=14)
ax1.set_ylabel('CRPS Integrand', fontsize=14)
ax1.grid(True, linestyle=':', alpha=0.7)

# Add arrow to true model plot
ax1.annotate('', xy=(0.3, 0.1), xytext=(0.6, 0.3),
             arrowprops=dict(facecolor='black', shrink=0.05, width=2, headwidth=8))
ax1.text(0.6, 0.35, f'CRPS: {np.sum(true_integrand):.2f}', ha='center', va='bottom', fontsize=12)

# Wrong model plot
ax2.plot(x, wrong_integrand, 'k', linewidth=2)
ax2.fill_between(x, wrong_integrand, color='red', alpha=0.5)
ax2.plot(x, beta_cdf(x, 1, 5)**2, 'gray', linestyle='--', linewidth=2)
ax2.plot(x, (beta_cdf(x, 1, 5)-1)**2, 'gray', linestyle='--', linewidth=2)
ax2.axvline(x=y, color='red', linestyle='--')
ax2.plot(y, 0, 'go', markersize=10)
ax2.set_xlim(0, 1)
ax2.set_ylim(0, 1)
ax2.set_title('Wrong model', fontsize=16)
ax2.set_xlabel('y', fontsize=14)
ax2.grid(True, linestyle=':', alpha=0.7)

# Add arrow to wrong model plot
ax2.annotate('', xy=(0.2, 0.2), xytext=(0.5, 0.4),
             arrowprops=dict(facecolor='black', shrink=0.05, width=2, headwidth=8))
ax2.text(0.5, 0.45, f'CRPS: {np.sum(wrong_integrand):.2f}', ha='center', va='bottom', fontsize=12)

plt.tight_layout()
plt.savefig('crps_beta_comparison_plot.pdf', dpi=300)
plt.close()