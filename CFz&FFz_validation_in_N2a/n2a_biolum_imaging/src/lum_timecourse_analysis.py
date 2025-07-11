#!/usr/bin/env python
# coding: utf-8

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.colors import to_rgb, to_hex, hsv_to_rgb
from scipy.signal import bessel, filtfilt
from scipy.stats import median_abs_deviation


### 1. Configuration

def add_dotted_lines(ax, injection_list):
    lines = []
    for inj in injection_list:
        line = ax.axvline(x=inj["time"], color=inj["color"], linestyle="--", linewidth=2, label=inj["label"])
        lines.append(line)
    ax.legend(handles=lines, loc='upper right', fontsize=12, frameon=True)

def apply_standard_plot_formatting(ax):
    ax.tick_params(axis='x', colors='black', labelsize=14)
    ax.tick_params(axis='y', colors='black', labelsize=14)
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)
    for spine in ['left', 'bottom']:
        ax.spines[spine].set_color("black")
    ax.set_facecolor("white")
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    ax.set_xlabel("Time (min)", fontsize=14, color="black", labelpad=5)

def export_figure(plt, output_base, output_dir, suffix):
    base = output_base.rstrip("_") + f"_{suffix}"
    png_path = os.path.join(output_dir, f"{base}.png")
    svg_path = os.path.join(output_dir, f"{base}.svg")
    plt.savefig(png_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.savefig(svg_path, format="svg", bbox_inches="tight", transparent=True)
    print(f" Saved figure to:\n- {png_path}\n- {svg_path}")

plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial"], "pdf.fonttype": 42,"ps.fonttype": 42})

# Expanded Wong + Tol-inspired colorblind-friendly palette
WONG_EXTENDED_COLORS = [
    "#D55E00",  # Vermillion
    "#E69F00",  # Orange
    "#009E73",  # Bluish green
    "#56B4E9",  # Sky blue
    "#0072B2",  # Blue
    "#F0E442",  # Yellow
    "#CC79A7",  # Reddish purple
    "#CC6677",  # Soft red
    "#88CCEE",  # Light blue
    "#44AA99",  # Teal green
    "#117733",  # Deep green
    "#882255"   # Plum
]

def get_color_map(n):
    return [WONG_EXTENDED_COLORS[i % len(WONG_EXTENDED_COLORS)] for i in range(n)]


# Input file path (from background_subtraction_&_traces_viewer)
file_path = "/Users/input/file_results_minus_bgd.csv"
lum_data = pd.read_csv(file_path)
time = lum_data["Time (min)"]

# Select only columns that start with 'Mean(' and exclude background
lum_columns = lum_data[[col for col in lum_data.columns 
                 if col.startswith("Mean(") and "Background" not in col]]

# Output root directory where new folder will be created
output_base_dir = "/Users/output/dir"

# Define injection annotations
injections = [
    {"label": "+2.3 µM Fz", "time": 1.0, "color": "black"},
    {"label": "+2 µM ionomycin", "time": 9.8, "color": "grey"},
]

#Set up output directory
basename = os.path.basename(file_path).replace("_results_minus_bgd.csv", "")
output_base = basename
output_dir = os.path.join(output_base_dir, basename)
os.makedirs(output_dir, exist_ok=True)

print(f" Outputs will be saved to: {output_dir}")


###2. Timecourse plotting (pre-filter)
colors = get_color_map(len(lum_columns.columns))

plt.figure(figsize=(6, 5), facecolor='white')
plt.ylabel("Luminescence", fontsize=14, color="black")
ax = plt.gca()

for idx, column in enumerate(lum_columns.columns):
    rolling_avg = lum_columns[column].rolling(window=5).mean()
    plt.plot(time, rolling_avg, label=column, linewidth=1.5, color=colors[idx])

apply_standard_plot_formatting(ax)
add_dotted_lines(ax, injections)

plt.tight_layout(pad=3.0)
export_figure(plt, output_base, output_dir, "luminescence")
plt.show()


### 3. Bessel filter

# Design Bessel filter
order = 4
sampling_interval = np.median(np.diff(time))  # minutes
sampling_rate_hz = 1 / (sampling_interval * 60)  # Hz
cutoff_hz = 0.1
b, a = bessel(order, Wn=cutoff_hz / (0.5 * sampling_rate_hz), btype='low', analog=False)

# Apply Bessel filter to lum_columns
filtered_data = lum_columns.copy()
for col in filtered_data.columns:
    filtered_data[col] = filtfilt(b, a, filtered_data[col])

# Preserve color order
lum_columns = lum_columns.columns
filtered_data = filtered_data[lum_columns] 
colors = get_color_map(len(lum_columns))    

# Plotting
fig, ax = plt.subplots(figsize=(6, 5), facecolor='white')
plt.ylabel("Luminescence", fontsize=14, color="black")
for idx, column in enumerate(lum_columns):
    ax.plot(time, filtered_data[column], label=column, linewidth=2.0, color=colors[idx])
apply_standard_plot_formatting(ax)
add_dotted_lines(ax, injections)

# Export filtered data
filtered_output = filtered_data.copy()
filtered_output.insert(0, "Time (min)", time)
filtered_name = f"{output_base}_luminescence_bessel_filtered.csv"
filtered_path = os.path.join(output_dir, filtered_name)
filtered_output.to_csv(filtered_path, index=False)
print(f"Filtered data saved to: {filtered_path}")

export_figure(plt, output_base, output_dir, "bessel_filtered")
plt.show()


### 4. Lum timecourse mean + SEM

# Compute mean and SEM across Bessel-filtered traces
mean_lum = filtered_data.mean(axis=1)
sem_lum = filtered_data.std(axis=1, ddof=1) / np.sqrt(filtered_data.shape[1])

# Plot mean ± SEM
fig, ax = plt.subplots(figsize=(6, 5), facecolor='white')

ax.plot(time, mean_lum, color="#009E73", linewidth=2.5, label='Mean Luminescence')
ax.fill_between(time, mean_lum - sem_lum, mean_lum + sem_lum, color="#009E73", alpha=0.3)

apply_standard_plot_formatting(ax)
add_dotted_lines(ax, injections)
plt.ylabel("Luminescence", fontsize=14, color="black")

# Export
export_figure(plt, output_base, output_dir, "mean+SEM")
plt.show()


### 5. ΔL/L

# Define L0 Time Window
l0_start_time_min = 9.5
l0_end_time_min = 9.6
valid_rows = time.between(l0_start_time_min, l0_end_time_min, inclusive="both")
l0_times = time[valid_rows]

# Use filtered data for ΔL/L
l0_values = filtered_data[valid_rows].mean()
lum_columns = filtered_data[l0_values.index]  # Ensure column alignment

# ΔL/L Calculation
delta_l_over_l = (lum_columns - l0_values) / l0_values
delta_l_over_l = delta_l_over_l.reset_index(drop=True)

# Plot ΔL/L
fig, ax = plt.subplots(figsize=(7, 6), facecolor='white')
colors = get_color_map(len(delta_l_over_l.columns))
for idx, column in enumerate(delta_l_over_l.columns):
    ax.plot(time, delta_l_over_l[column], label=column, linewidth=2.0, color=colors[idx])
ax.set_ylabel("ΔL/L", fontsize=14, color="black")
apply_standard_plot_formatting(ax)
add_dotted_lines(ax, injections)
ax.autoscale(enable=True, axis='y', tight=False)

# Export plot
export_figure(plt, output_base, output_dir, "deltaLoverL")
plt.show()

# Export ΔL/L values
deltaLoverL_output = delta_l_over_l.copy()
deltaLoverL_output.insert(0, "Time (min)", time)
deltaLoverL_csv_filename = os.path.join(output_dir, f"{output_base}_deltaLoverL.csv")
deltaLoverL_output.to_csv(deltaLoverL_csv_filename, index=False)
print(f"ΔL/L values saved to: {deltaLoverL_csv_filename}")

