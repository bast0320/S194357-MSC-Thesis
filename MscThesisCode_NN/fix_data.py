# import the offshore wind data for DK2 and DK2

import pickle
import pandas as pd

# Load the data
with open('loaded_variables/actuals_hourly_DK2.pkl', 'rb') as f:
    data_DK2 = pickle.load(f)

with open('loaded_variables/actuals_hourly_DK2.pkl', 'rb') as f:
    data_DK2 = pickle.load(f)

# maybe we should load in the ensembles, but let's see...

import matplotlib.pyplot as plt

# Create the plot
# plt.figure(figsize=(10, 6))

# Plot the entire data in black
# plt.plot(data_DK2.index, data_DK2["OffshoreWindPower"], label='DK2 Offshore Wind Power', color='black')

# Calculate the max of the first 80% of the data
h_line_value = data_DK2["OffshoreWindPower"].iloc[0:int(0.8*len(data_DK2))].max()

# Plot the horizontal line
# plt.axhline(h_line_value, color='r', linestyle='--', label='Max of first 80%')

# Highlight the data above the horizontal line in blue
above_h_line = data_DK2["OffshoreWindPower"] > h_line_value
# plt.plot(data_DK2.index[above_h_line], data_DK2["OffshoreWindPower"][above_h_line], '-', color='blue', label='Above Max Value', linewidth=2)

# Add legend and labels
# plt.legend()
# plt.xlabel('Time')
# plt.ylabel('Power (MW)')
# extend y lim 15%
# plt.ylim(0, 1.15*data_DK2["OffshoreWindPower"].max())
# plt.title('Offshore Wind Power for DK2')
# plt.savefig('figures/offshore_wind_power_DK2.pdf')
# plt.show()
print(h_line_value)

#data_DK2["OffshoreWindPower"][above_h_line] *= h_line_value/max(data_DK2["OffshoreWindPower"][above_h_line])



# set all values below 0 to 0
data_DK2["OffshoreWindPower"][data_DK2["OffshoreWindPower"] < 0] = 0


plt.figure(figsize=(10, 6))
plt.plot(data_DK2.index, data_DK2["OffshoreWindPower"], label='DK2  Offshore Wind Power clipped', color='black')
plt.show()



# Now we focus on solar, we want to set all <1 to 0
data_DK2["SolarPower"][data_DK2["SolarPower"] < 1] = 0

# save as pickle
with open('loaded_variables/actuals_hourly_DK2_fixed.pkl', 'wb') as f:
    pickle.dump(data_DK2, f)



# %%
# import ensembles
import pandas as pd
wind_onshore_DK1 = pd.read_pickle("loaded_variables/ensembles_wind_onshore_DK1.pkl")
wind_onshore_DK2 = pd.read_pickle("loaded_variables/ensembles_wind_onshore_DK2.pkl")
wind_offshore_DK1 = pd.read_pickle("loaded_variables/ensembles_wind_offshore_DK1.pkl")
wind_offshore_DK2 = pd.read_pickle("loaded_variables/ensembles_wind_offshore_DK2.pkl")
solar_DK1 = pd.read_pickle("loaded_variables/ensembles_solar_DK1.pkl")
solar_DK2 = pd.read_pickle("loaded_variables/ensembles_solar_DK2.pkl")

# make sure no wind observations is below 0
wind_onshore_DK1[wind_onshore_DK1 < 0] = 0
wind_onshore_DK2[wind_onshore_DK2 < 0] = 0
wind_offshore_DK1[wind_offshore_DK1 < 0] = 0
wind_offshore_DK2[wind_offshore_DK2 < 0] = 0
solar_DK1[solar_DK1 < 0] = 0
solar_DK2[solar_DK2 < 0] = 0

# fix the wind:off:short_dk1 with the hline value as above
h_line_value = wind_offshore_DK1.iloc[0:int(0.8*len(wind_offshore_DK1))].values.flatten().max()
# do this trick, but to wind offshore DK1 # data_DK2["OffshoreWindPower"][above_h_line] *= h_line_value/max(data_DK2["OffshoreWindPower"][above_h_line])
wind_offshore_DK1[wind_offshore_DK1 > h_line_value] *= h_line_value/wind_offshore_DK1.iloc[int(0.8*len(wind_offshore_DK1)):].values.flatten().max()


# save as pickle
wind_onshore_DK1.to_pickle("loaded_variables/ensembles_wind_onshore_DK1_fixed.pkl")
wind_onshore_DK2.to_pickle("loaded_variables/ensembles_wind_onshore_DK2_fixed.pkl")
wind_offshore_DK1.to_pickle("loaded_variables/ensembles_wind_offshore_DK1_fixed.pkl")
wind_offshore_DK2.to_pickle("loaded_variables/ensembles_wind_offshore_DK2_fixed.pkl")
solar_DK1.to_pickle("loaded_variables/ensembles_solar_DK1_fixed.pkl")
solar_DK2.to_pickle("loaded_variables/ensembles_solar_DK2_fixed.pkl")


# %%
import pandas as pd
solar_DK1 = pd.read_pickle("loaded_variables/ensembles_solar_DK1.pkl")
solar_DK2 = pd.read_pickle("loaded_variables/ensembles_solar_DK2.pkl")

# Function to remove rows with any value less than 5
def remove_rows_with_values_below_threshold(dataframe, threshold=5):
    # Identify rows to keep
    rows_to_keep = ~dataframe.lt(threshold).any(axis=1)
    # Keep the rows
    filtered_dataframe = dataframe[rows_to_keep]
    return filtered_dataframe

# Apply the function to the sample dataframe
solar_DK1_v2 = remove_rows_with_values_below_threshold(solar_DK1, threshold=5)
solar_DK2_v2 = remove_rows_with_values_below_threshold(solar_DK2, threshold=5)

# save as pickle
solar_DK1_v2.to_pickle("loaded_variables/ensembles_solar_DK1_fixed_v2.pkl")
solar_DK2_v2.to_pickle("loaded_variables/ensembles_solar_DK2_fixed_v2.pkl")
# %%
