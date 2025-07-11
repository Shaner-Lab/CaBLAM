#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import os
import matplotlib.pyplot as plt
import math
from matplotlib.colors import hsv_to_rgb
import numpy as np
import re


# ## 1. BACKGROUND SUBTRACTION

# In[ ]:


# Set input/output files
file_path = "/Users/path/to/raw_data_results.csv"
data = pd.read_csv(file_path)

output_base_dir = "/Users/output/path"

basename = os.path.basename(file_path).replace("_results.csv", "")
output_base = basename
output_dir = os.path.join(output_base_dir, basename)
os.makedirs(output_dir, exist_ok=True)


# In[ ]:


# Rename the first column to "Frames"
data.rename(columns={data.columns[0]: "Frames"}, inplace=True)

#Add a new column "Time (min)" that converts frames to minutes
data.insert(1, "Time (min)", data["Frames"] / 10 / 60)     ##Sampling Frequency (Hz)/60

# Delete the "Label" column (if it exists)
if "Label" in data.columns:
    data.drop(columns=["Label"], inplace=True)

# Subtract the value of the last item in each row (Background ROI) from all values in that row (excluding "Frames" and "Time (s)")
data_minus_bgd = data.copy()
for index, row in data.iterrows():
    last_value = row.iloc[-1]  # Get the last value in the current row
    data_minus_bgd.iloc[index, 2:] = row.iloc[2:] - last_value  # Subtract from all columns except "Frames" and "Time (s)"

output_name = f"{output_base}_results_minus_bgd.csv"
output_path = os.path.join(output_dir, output_name)
print(f"Modified dataset saved to: {output_path}")


# ## 2. INDIVIDUAL TRACES VIEWER

# In[ ]:


def get_color_map(n, cycles=3):
    """
    Generate `n` visually distinct HSV-based colors, cycling through the hue range `cycles` times.
    """
    hues = [(i * cycles / n) % 1.0 for i in range(n)]
    return [hsv_to_rgb((h, 1, 1)) for h in hues]

data = pd.read_csv(output_name) # Auto uses background subtracted data from 1
time = data["Time (min)"]  # x-axis
trace_columns = [col for col in data.columns if col.startswith("Mean(") and "Background" not in col]
lum_columns = data[trace_columns]

# Parameters
rolling_mean = lum_columns.rolling(window=5).mean()
n_cells = rolling_mean.shape[1]
n_cols = 3
n_rows = math.ceil(n_cells / n_cols)

colors = get_color_map(n_cells)

fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 2.5 * n_rows), sharex=True, facecolor='white')
axes = axes.flatten()

for idx, (col, ax) in enumerate(zip(rolling_mean.columns, axes)):
    ax.plot(time, rolling_mean[col], color=colors[idx], linewidth=1.5)

    # Title per subplot
    ax.set_title(f"Cell {idx + 1}", loc='left', fontsize=11, fontweight='bold')
    ax.set_facecolor('white')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

output_name = f"{output_base}_individual_traces"
output_path = os.path.join(output_dir, output_name)
print(f"Individual traces saved to {output_path}")

# Export as PNG and PDF
plt.savefig(f"{output_name}.png", dpi=300, bbox_inches="tight")
plt.savefig(f"{output_name}.pdf", bbox_inches="tight")


# In[ ]:




