# %%
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# load the actuals_5min_DK2.pkl file from loaded_variables
with open('loaded_variables/actuals_5min_DK2.pkl', 'rb') as f:
    actuals_5min_DK2 = pickle.load(f)

with open('loaded_variables/actuals_hourly_DK2.pkl', 'rb') as f:
    actuals_hourly_DK2 = pickle.load(f)

with open('loaded_variables/ensembles_wind_offshore_DK2.pkl', 'rb') as f:
    ensembles_wind_offshore_DK2 = pickle.load(f)


with open('loaded_variables/ensembles_solar_DK2.pkl', 'rb') as f:
    ensembles_solar_DK2 = pickle.load(f)

with open('loaded_variables/forecasts_day_ahead_wind_onshore_DK2.pkl', 'rb') as f:
    forecasts_day_ahead_wind_onshore_DK2 = pickle.load(f)

# %%
# provide summary statistics and plots of the data
# actuals_5min_DK2
cols_to_focus_on = ["OffshoreWindPower", "SolarPower", "OnshoreWindPower"]
table = actuals_hourly_DK2[cols_to_focus_on].describe().applymap(lambda x: f"{x:0.2f}")
latex_table = table.to_latex()
print(latex_table)
print(actuals_hourly_DK2[cols_to_focus_on].shape)

# %%
fig, axs = plt.subplots(len(cols_to_focus_on), 1, figsize=(8, 6), sharex=True)
fig.suptitle("Actuals Hourly DK2")

for i, col in enumerate(cols_to_focus_on):
    axs[i].plot(actuals_hourly_DK2[col], color="black", linewidth=0.5, label=f"{col}")
    axs[i].legend(loc='upper left')
    axs[i].set_ylabel("Power (MW)")
    # increase the y-axis lim to be 1.2 times the range of data
    y_min = actuals_hourly_DK2[col].min()
    y_max = actuals_hourly_DK2[col].max()
    
    axs[i].set_ylim(y_min, 1.2*y_max)

plt.xlabel("Time")
plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust the spacing between subplots
plt.savefig("figures/actuals_hourly_DK2.pdf")
plt.show()
# %%
