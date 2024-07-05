# Create a new DataFrame from the provided corrected data
import pandas as pd
new_data = {
    "Model": [
        "DK1 Onshore Wind Power FFNN", "DK1 Onshore Wind Power LSTM",
        "DK1 Offshore Wind Power LSTM", "DK1 Solar Power FFNN",
        "DK1 Solar Power LSTM", "DK2 Offshore Wind Power FFNN",
        "DK2 Offshore Wind Power LSTM", "DK2 Solar Power FFNN",
        "DK2 Solar Power LSTM"
    ],
    "Corrected": [
        17557864.72, 17378332.85, 9619370.97, 1419642.35, 2866043.70, 
        6538132.54, 6571776.72, 958146.61, 961117.11
    ],
    "Original": [
        17558311.99, 17558311.99, 9722929.86, 2728229.26, 2728229.26,
        6510622.61, 6510622.61, 958183.94, 958183.94
    ],
    "SDE Corrected": [
        20755813.80, 19741976.75, 9627499.75, 1411748.63, 1429837.77,
        5438854.69, 5044915.96, 586543.16, 589949.08
    ]
}

new_df = pd.DataFrame(new_data)

# Calculate the percentage scores
new_df_percentage = new_df.copy()
new_df_percentage["Corrected"] = (new_df["Corrected"] / new_df["Original"]) * 100
new_df_percentage["SDE Corrected"] = (new_df["SDE Corrected"] / new_df["Original"]) * 100
new_df_percentage["Original"] = 100

# Converting the DataFrame to LaTeX format with two decimal points
latex_table_new = new_df_percentage.to_latex( float_format="%.2f", caption="Sum of Mean Profits for Different Models and Types", label="table:mean_profits")
latex_table_new

# Save the LaTeX table to a file
with open("table_mean_profits.tex", "w") as f:
    f.write(latex_table_new)