#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


# In[ ]:


plt.rcParams.update({
    "font.family": "Arial",
    "font.size": 12,
    "pdf.fonttype": 42,
    "ps.fonttype": 42
})

# Load AUC from timecourse CSV
def load_plate_data(filepath, conc_row=1):
    raw = pd.read_csv(filepath, header=None)
    concs = raw.iloc[conc_row, 1:].astype(float).values
    times = pd.to_timedelta(raw.iloc[3:, 0]).dt.total_seconds().values / 60
    lum = raw.iloc[3:, 1:].apply(pd.to_numeric, errors='coerce').reset_index(drop=True)
    valid_rows = ~lum.isna().any(axis=1)
    lum_clean = lum[valid_rows]
    times_clean = times[valid_rows]
    auc = np.trapz(lum_clean.values, x=times_clean, axis=0)
    return concs, auc

# Load Data
cfz_concs, cfz_auc = load_plate_data("CFz.csv")
ffz_concs, ffz_auc = load_plate_data("FFz.csv")
fz_concs, fz_auc = load_plate_data("Fz.csv")

cfz_df = pd.DataFrame({"substrate": "CFz", "conc": cfz_concs, "auc": cfz_auc})
ffz_df = pd.DataFrame({"substrate": "FFz", "conc": ffz_concs, "auc": ffz_auc})
fz_df = pd.DataFrame({"substrate": "Fz", "conc": fz_concs, "auc": fz_auc})

#Helpers
def get_mean_trace(df):
    grouped = df.groupby("conc")["auc"]
    mean_aucs = grouped.mean()
    sem_aucs = grouped.sem()
    x = mean_aucs.index.values
    y = mean_aucs.values
    yerr = sem_aucs.values
    mask = x > 0
    return x[mask], y[mask], yerr[mask]

def get_log_range(x):
    return np.logspace(np.log10(min(x) * 0.5), np.log10(max(x) * 2), 500)

def r_squared(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - (ss_res / ss_tot)

def print_params(label, popt, param_names, x=None, y=None, model=None):
    print(f"{label} Fit Parameters:")
    for name, val in zip(param_names, popt):
        unit = " µM" if "EC50" in name else ""
        fmt = ".2e" if name in ["Top", "Offset (c)"] else ".3f"
        print(f"  {name:<10}: {format(val, fmt)}{unit}")
    if x is not None and y is not None and model is not None:
        y_pred = model(x, *popt)
        r2 = r_squared(y, y_pred)
        print(f"  R²         : {r2:.4f}")
      
      
# Model
def combined_hill_linear(x, top, ec50, hill_slope, m, c):
    hill_component = top / (1 + (ec50 / x)**hill_slope)
    linear_component = m * x + c
    return hill_component + linear_component

def hill_only(x, top, ec50, hill_slope):
    return top / (1 + (ec50 / x)**hill_slope)

# Fit Function
def fit_and_plot(ax, df, label, color, initial_guess, hill_only_flag=False):
    x, y, yerr = get_mean_trace(df)

    if len(x) >= 2:
        decay_slope = (y[-1] - y[-2]) / (x[-1] - x[-2])
        print(f"{label} Approximate decay slope (last two points): {decay_slope:.2f}")

    if hill_only_flag:
        model = hill_only
        guess = initial_guess[:3]
        bounds = ([0, 0, 0], [np.inf, np.inf, np.inf])
        param_names = ["Top", "EC50", "Hill Slope"]
    else:
        model = combined_hill_linear
        guess = initial_guess[:5]
        bounds = ([0, 0, 0, -1e7, -np.inf], [np.inf, np.inf, np.inf, 0, np.inf])
        param_names = ["Top", "EC50", "Hill Slope", "Slope (m)", "Offset (c)"]

    popt, _ = curve_fit(model, x, y, p0=guess, bounds=bounds)
    x_fit = get_log_range(x)
    y_fit = model(x_fit, *popt)
    # Trim at y=0 to avoid visual drop below zero
    valid_mask = y_fit >= 0
    x_fit = x_fit[valid_mask]
    y_fit = y_fit[valid_mask]

    ax.errorbar(x, y, yerr=yerr, fmt='o', capsize=3, label=label, color=color)
    ax.plot(x_fit, y_fit, color=color)
    print_params(label, popt, param_names, x, y, model)

# Plot
fig, ax = plt.subplots(figsize=(6, 7))
fit_and_plot(ax, cfz_df, "CFz", "#0072B2", initial_guess=[3e7, 100, 1.0, -3000, 1e6])
fit_and_plot(ax, ffz_df, "FFz", "#009E73", initial_guess=[2.5e7, 100, 1.0, -3000, 1e6])
fit_and_plot(ax, fz_df,  "Fz",  "#D55E00", initial_guess=[3e7, 100, 1.0], hill_only_flag=True)

ax.set_xscale("log")
ax.set_xlabel("Substrate [µM]")
ax.set_ylabel("AUC (RLU)")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.legend(frameon=False, loc='upper right')
plt.tight_layout()

# Export
fig.savefig("hybrid_fit_plot.svg", bbox_inches="tight")
fig.savefig("hybrid_fit_plot.png", dpi=600, bbox_inches="tight")
plt.show()

