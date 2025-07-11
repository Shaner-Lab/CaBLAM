#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Imports
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import matplotlib.colors as mcolors
from scipy.stats import sem

# --- Plotting Defaults ---
plt.rcParams.update({
    "font.family": "Arial",
    "font.size": 11,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

COLOR_MAP = {
    "CFz": {
        "0.01 μM": "#a3cde2",
        "100 μM": "#3182bd",
        "1000 μM": "#005b94",
    },
    "FFz": {
        "0.01 μM": "#94d8be",
        "100 μM": "#2ba977",
        "1000 μM": "#007f5f",
    },
    "Fz": {
        "2.3 μM": '#E6550D',
        "0.766 μM": '#FD8D3C',
        "0.23 μM": '#FDBE85'
    }
}

# File locations
cfz_dirs = {
    "0.01 μM": "/Users/path/to/dir",
    "100 μM":  "/Users/path/to/dir",
    "1000 μM": "/Users/path/to/dir"
}
ffz_dirs = {
    "0.01 μM": "/Users/path/to/dir",
    "100 μM":  "/Users/path/to/dir",
    "1000 μM": "/Users/path/to/dir"
}
fz_dirs = {
    "0.23 μM": "/Users/path/to/dir",
    "0.766 μM": "/Users/path/to/dir", 
    "2.3 μM": "/Users/path/to/dir"
}


# --- Concentration Lists ---
cfz_concentrations = ["0.01 μM", "100 μM", "1000 μM"]
ffz_concentrations = ["0.01 μM", "100 μM", "1000 μM"]
fz_concentrations = ["2.3 μM", "0.766 μM", "0.23 μM"]

# --- Helper Functions ---
def find_file(folder, keyword):
    for file in os.listdir(folder):
        if keyword in file and file.endswith(".csv"):
            return os.path.join(folder, file)
    return None

def load_lum_data(filepath):
    df = pd.read_csv(filepath)
    time_col = "Time (min)" if "Time (min)" in df.columns else df.columns[0]
    x = df[time_col]
    y_data = df.drop(columns=[time_col])
    return x, y_data

def plot_mean_sem_trace(ax, x, y_data, color):
    mean_trace = y_data.mean(axis=1).values
    sem_trace = y_data.std(axis=1).values / np.sqrt(y_data.shape[1])
    base_rgb = np.array(mcolors.to_rgb(color))
    light_rgb = base_rgb + (1 - base_rgb) * 0.6
    light_color = tuple(light_rgb)

    step = 3
    x_vals = x.values[::step]
    mean_vals = mean_trace[::step]
    sem_vals = sem_trace[::step]

    ax.fill_between(x_vals, mean_vals - sem_vals, mean_vals + sem_vals,
                    color=light_color, alpha=0.5, linewidth=0)
    ax.plot(x, mean_trace, color=color, linewidth=2.0)

def plot_luminescence_traces(substrate, conc_list, dirs_dict, ax=None):
    for i, conc in enumerate(conc_list):
        if ax is None:
            fig, single_ax = plt.subplots(figsize=(3.25, 2.5))
        else:
            single_ax = ax[i] if isinstance(ax, list) else ax

        path = find_file(dirs_dict[conc], "luminescence_bessel_filtered")
        if path:
            x, y_data = load_lum_data(path)
            print(f"{substrate} {conc}: {y_data.shape[1]} traces")
            plot_mean_sem_trace(single_ax, x, y_data, COLOR_MAP[substrate][conc])
            single_ax.text(0.98, 0.92, f"{conc} {substrate}", transform=single_ax.transAxes,
                           ha='right', va='top', fontsize=9)

        single_ax.set_xlim(0, 45)
        single_ax.set_ylim(0, 2000)
        single_ax.set_yticks(np.arange(0, 2001, 500))
        single_ax.set_xticks(range(0, 46, 10))
        single_ax.set_xlabel("Time (min)")
        single_ax.set_ylabel("Luminescence (counts)")
        single_ax.spines['top'].set_visible(False)
        single_ax.spines['right'].set_visible(False)
        single_ax.spines['left'].set_visible(True)
        single_ax.spines['bottom'].set_visible(True)
        single_ax.tick_params(direction='out', length=4, width=1)

        if ax is None:
            single_ax.figure.tight_layout()
            single_ax.figure.savefig(f"Luminescence_{substrate}_{conc.replace(' ', '').replace('μ', 'u')}.svg", transparent=True)
            plt.show()
            plt.close()

# --- Run All Substrate Plots Individually ---
#plot_luminescence_traces("CFz", cfz_concentrations, cfz_dirs)
#plot_luminescence_traces("FFz", ffz_concentrations, ffz_dirs)
#plot_luminescence_traces("Fz", fz_concentrations, fz_dirs)

# --- Combined 9-Panel Figure ---
fig, axes = plt.subplots(3, 3, figsize=(10, 7.5), sharex=True, sharey=True)
plot_luminescence_traces("CFz", cfz_concentrations, cfz_dirs, ax=axes[:, 0].tolist())
plot_luminescence_traces("FFz", ffz_concentrations, ffz_dirs, ax=axes[:, 1].tolist())
plot_luminescence_traces("Fz", fz_concentrations, fz_dirs, ax=axes[:, 2].tolist())
fig.tight_layout()
fig.subplots_adjust(wspace=0.25, hspace=0.3)
fig.savefig("lum_timecourses_all.svg", transparent=True)
fig.savefig("lum)timecourses_all.png", dpi=600)
plt.show()

