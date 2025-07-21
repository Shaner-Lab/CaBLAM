#! /usr/bin/env python3
"""
titration_fit_global.py  (v4.0, 2025-07-21)

Key features
============
• Global Hill fits for any number of clones (one CSV per clone).
• Clone colours ordered by EC50, colour-map tweakable (sat/val).
• Per‑clone **marker shapes / fill styles** via `--markers` and `--fills`.
• Options for nM or pCa x-axis, font override, legend & point sizing.
• Per-clone workbooks + global summary workbook.
• Confidence-band uses full covariance propagation (ymin, EC50, nH).
• Direct contrast calculation from low-Ca data points for realistic uncertainties.
• Fit-text block: adjustable font size, line spacing, optional name,
  Hill coefficient shown as *n*ₕ.
• PDFs embed editable text thanks to matplotlib fonttype=42.
"""

import argparse
import colorsys
import importlib
import itertools
import pathlib
import sys
import warnings
from typing import List, Union, Optional

import matplotlib.pyplot as plt
import matplotlib as mpl

# ── Ensure text remains editable in vector outputs ───────────────────
mpl.rcParams['pdf.fonttype'] = 42      # TrueType in PDFs (editable)
mpl.rcParams['ps.fonttype'] = 42       # TrueType in PS/EPS
mpl.rcParams['svg.fonttype'] = 'none'  # keep SVG text as text

import numpy as np
import pandas as pd
from matplotlib.ticker import LogLocator, NullFormatter
from scipy.optimize import curve_fit
from scipy import stats

warnings.filterwarnings("ignore", category=RuntimeWarning)

# ───────────────────────── helper functions ──────────────────────────
def load_clone(csv: pathlib.Path) -> pd.DataFrame:
    """Read a CSV where columns are Ca, Signal pairs (any number)."""
    df = pd.read_csv(csv)
    if len(df.columns) % 2:
        sys.exit(f"❌ {csv.name}: need paired Ca/Signal columns.")
    parts = []
    for i in range(0, len(df.columns), 2):
        sub = df.iloc[:, i : i + 2].dropna()
        sub.columns = ["Ca_nM", "signal"]
        sub["run"] = f"run_{i//2+1}"
        sub["signal_norm"] = sub["signal"] / sub["signal"].max()
        parts.append(sub)
    return pd.concat(parts, ignore_index=True)


def pick_engine(requested: Optional[str]):
    """Return an available Excel writer engine or None."""
    if requested == "none":
        return None
    if requested:
        return requested
    for eng in ("openpyxl", "xlsxwriter"):
        if importlib.util.find_spec(eng):
            return eng
    return None


def tweak_rgb(rgb, sat_fac, val_fac):
    """Scale saturation & value of an RGB triple (0-1 range)."""
    h, s, v = colorsys.rgb_to_hsv(*rgb)
    s = max(0, min(1, s * sat_fac))
    v = max(0, min(1, v * val_fac))
    return colorsys.hsv_to_rgb(h, s, v)


def hill3(x, ymin, EC50, nH):
    """Three-parameter Hill (sigmoid) with ymax constrained to 1."""
    return ymin + (1 - ymin) / (1 + (EC50 / x) ** nH)


def fit_hill(x, y):
    p0 = [0.0, np.median(x), 1.0]
    bounds = ([-0.5, 1e-3, 0], [0.5, np.max(x) * 1e3, 8])
    return curve_fit(hill3, x, y, p0=p0, bounds=bounds, maxfev=100_000)


def calculate_direct_contrast(data, ca_threshold=10.0):
    """
    Calculate contrast directly from data points below a calcium threshold.
    
    Parameters:
    -----------
    data : pd.DataFrame
        DataFrame with 'Ca_nM' and 'signal_norm' columns
    ca_threshold : float
        Maximum calcium concentration to consider for baseline (nM)
        
    Returns:
    --------
    dict with contrast, delta, and their uncertainties
    """
    # Filter data points below threshold
    low_ca_data = data[data['Ca_nM'] <= ca_threshold]
    
    if len(low_ca_data) == 0:
        return None
    
    # Calculate mean and standard error of the mean for low-Ca points
    low_ca_signals = low_ca_data['signal_norm'].values
    mean_signal = np.mean(low_ca_signals)
    sem_signal = stats.sem(low_ca_signals)
    
    # Calculate contrast (1/mean) and its uncertainty using error propagation
    contrast = 1 / mean_signal
    contrast_se = sem_signal / (mean_signal ** 2)
    
    # Calculate delta ((1-mean)/mean) and its uncertainty
    delta = (1 - mean_signal) / mean_signal
    delta_se = sem_signal / (mean_signal ** 2)
    
    return {
        'contrast': contrast,
        'contrast_se': contrast_se,
        'delta': delta,
        'delta_se': delta_se,
        'mean_signal': mean_signal,
        'sem_signal': sem_signal,
        'n_low_ca_points': len(low_ca_signals),
        'ca_threshold': ca_threshold
    }


def ci_band(xe: np.ndarray, popt, pcov):
    """95 % confidence envelope for the fitted Hill curve."""
    ymin, EC50, nH = popt
    g = (EC50 / xe) ** nH
    denom = 1 + g

    d_ymin = g / denom
    d_EC50 = -(1 - ymin) * nH * g / (EC50 * denom**2)
    d_nH = -(1 - ymin) * g * np.log(EC50 / xe) / denom**2

    J = np.vstack((d_ymin, d_EC50, d_nH)).T  # N×3 Jacobian
    var = np.einsum("ij,jk,ik->i", J, pcov, J)
    se = np.sqrt(var)

    y_hat = hill3(xe, *popt)
    z = 1.96
    return y_hat - z * se, y_hat + z * se


def generate_filename_tags(a):
    """Generate compact descriptive tags for output filename based on settings."""
    tags = []
    
    # Auto-optimization flags (most important)
    if a.auto_fonts:
        tags.append("af")
    if a.auto_markers:
        tags.append("am")
    if a.auto_lines:
        tags.append("al")
    
    # Hill contrast flag (only if explicitly requested)
    if a.use_hill_contrast:
        tags.append("hill")
    
    # Figure size (most important)
    if a.fig_w_mm is not None or a.fig_h_mm is not None:
        w_mm = a.fig_w_mm if a.fig_w_mm is not None else a.fig_w * 25.4
        h_mm = a.fig_h_mm if a.fig_h_mm is not None else a.fig_h * 25.4
        tags.append(f"{w_mm:.0f}x{h_mm:.0f}")
    elif a.fig_w != 6.0 or a.fig_h != 4.0:
        tags.append(f"{a.fig_w:.1f}x{a.fig_h:.1f}")
    
    # Palette (abbreviated)
    if a.palette != "tab10":
        palette_abbrev = {
            "plasma": "pl", "viridis": "vi", "magma": "ma", "inferno": "in",
            "cividis": "ci", "cool": "co", "coolwarm": "cw", "rainbow": "rb",
            "tab20": "t20", "Set1": "s1", "Set2": "s2", "Set3": "s3"
        }
        tags.append(palette_abbrev.get(a.palette, a.palette[:2]))
    
    # Font (abbreviated)
    if a.font:
        font_abbrev = {"Arial": "ar", "Times": "tm", "Helvetica": "hl"}
        tags.append(font_abbrev.get(a.font, a.font[:2]))
    
    # Marker set (abbreviated)
    if a.marker_set != "auto":
        tags.append(f"m{a.marker_set[:2]}")
    elif a.markers != "auto":
        tags.append("mc")
    
    # Smart alpha (abbreviated)
    if a.smart_alpha:
        tags.append("sa")
    
    # Fit text
    if a.fit_text:
        tags.append("ft")
    
    # No excel
    if a.no_excel:
        tags.append("ne")
    
    # Units (only if not nM)
    if a.x_units != "nM":
        tags.append(a.x_units)
    
    # Scaling factors (only if not 1.0)
    if a.font_scale != 1.0:
        tags.append(f"fs{a.font_scale}")
    if a.marker_scale != 1.0:
        tags.append(f"ms{a.marker_scale}")
    if a.line_scale != 1.0:
        tags.append(f"ls{a.line_scale}")
    
    # Manual overrides (only if significantly different from defaults)
    if a.sat != 1.0:
        tags.append(f"s{a.sat}")
    if a.val != 1.0:
        tags.append(f"v{a.val}")
    if a.fit_lw != 2.0:
        tags.append(f"fl{a.fit_lw}")
    if a.ci_lw != 0.0:
        tags.append(f"cl{a.ci_lw}")
    
    return "_".join(tags) if tags else ""


def calculate_smart_alpha_cross_dataset(all_marker_data, target_dataset_idx, alpha_base=0.8, alpha_min=0.3, pt_size=34):
    """
    Calculate alpha values based on cross-dataset marker overlaps.
    Only considers overlaps with markers from OTHER datasets, not within the same dataset.
    """
    target_data = all_marker_data[target_dataset_idx]
    x_target = target_data['x']
    y_target = target_data['y']
    
    if len(x_target) == 0:
        return np.array([])
    
    # Calculate radius based on marker size (smaller than marker size)
    # Convert point size to data coordinates (approximate)
    marker_diameter = np.sqrt(pt_size) / 72.0  # Convert points to inches, then to data units
    
    # Use a radius that's about 80% of the marker diameter for precise overlap detection
    radius = marker_diameter * 0.8
    
    alphas = np.full(len(x_target), alpha_base)
    
    for i in range(len(x_target)):
        # Count nearby points from OTHER datasets only
        nearby_count = 0
        
        for dataset_idx, dataset_data in enumerate(all_marker_data):
            if dataset_idx != target_dataset_idx:  # Only check other datasets
                x_other = dataset_data['x']
                y_other = dataset_data['y']
                
                # Calculate distances to all points in this other dataset
                distances = np.sqrt((x_other - x_target[i])**2 + (y_other - y_target[i])**2)
                nearby_count += np.sum(distances < radius)
        
        # Only reduce alpha if there's overlap with markers from other datasets
        if nearby_count >= 1:
            # Conservative density factor calculation
            density_factor = min(nearby_count / 3.0, 1.0)  # Start reducing at 1+ point
            alphas[i] = alpha_base - (alpha_base - alpha_min) * density_factor
    
    return alphas


def calculate_optimal_font_sizes(fig_w, fig_h, n_clones, font_scale=1.0):
    """
    Calculate optimal font sizes based on figure dimensions and number of clones.
    """
    # Base font sizes that work well for standard 6x4 inch figure
    base_axis_fs = 10
    base_tick_fs = 8
    base_legend_fs = 8
    base_ec50_fs = 6
    
    # Calculate scaling factor based on figure area and number of clones
    fig_area = fig_w * fig_h
    base_area = 6 * 4  # Standard 6x4 inch figure
    
    # Scale based on figure area (larger figures get larger fonts)
    area_scale = np.sqrt(fig_area / base_area)
    
    # Scale based on number of clones (fewer clones can have larger fonts)
    clone_scale = max(0.8, min(1.2, 5 / n_clones))  # Scale between 0.8-1.2 based on clone count
    
    # Combine scaling factors
    overall_scale = area_scale * clone_scale * font_scale
    
    # Calculate optimal font sizes
    optimal_axis_fs = max(8, min(16, base_axis_fs * overall_scale))
    optimal_tick_fs = max(8, min(14, base_tick_fs * overall_scale * 1.2))  # Boost tick size by 20%
    optimal_legend_fs = max(8, min(18, base_legend_fs * overall_scale * 1.2))  # Boost legend size by 20%
    optimal_ec50_fs = max(4, min(10, base_ec50_fs * overall_scale))
    
    return optimal_axis_fs, optimal_tick_fs, optimal_legend_fs, optimal_ec50_fs


def calculate_optimal_marker_size(fig_w, fig_h, n_clones, marker_scale=1.0):
    """
    Calculate optimal marker size based on figure dimensions and number of clones.
    """
    # Base marker size that works well for standard 6x4 inch figure
    base_marker_size = 34
    
    # Calculate scaling factor based on figure area and number of clones
    fig_area = fig_w * fig_h
    base_area = 6 * 4  # Standard 6x4 inch figure
    
    # Scale based on figure area (larger figures get larger markers)
    # Use linear scaling instead of sqrt for more dramatic effect
    area_scale = fig_area / base_area
    
    # Scale based on number of clones (fewer clones can have larger markers)
    clone_scale = max(0.7, min(1.5, 8 / n_clones))  # More aggressive scaling
    
    # Combine scaling factors
    overall_scale = area_scale * clone_scale * marker_scale
    
    # Calculate optimal marker size with wider range
    optimal_marker_size = max(15, min(120, base_marker_size * overall_scale))
    
    return optimal_marker_size


def calculate_optimal_line_weights(fig_w, fig_h, line_scale=1.0):
    """
    Calculate optimal line weights based on figure dimensions.
    """
    # Base line weights that work well for standard 6x4 inch figure
    base_fit_lw = 2.0
    base_ci_lw = 0.0  # Confidence intervals are typically filled, not lined
    
    # Calculate scaling factor based on figure area
    fig_area = fig_w * fig_h
    base_area = 6 * 4  # Standard 6x4 inch figure
    
    # Scale based on figure area (larger figures get thicker lines)
    area_scale = np.sqrt(fig_area / base_area)
    
    # Combine scaling factors
    overall_scale = area_scale * line_scale
    
    # Calculate optimal line weights
    optimal_fit_lw = max(1.0, min(4.0, base_fit_lw * overall_scale))
    optimal_ci_lw = max(0.0, min(1.0, base_ci_lw * overall_scale))  # Keep CI lines thin or none
    
    return optimal_fit_lw, optimal_ci_lw


# ───────────────────────────── CLI ───────────────────────────────
def cli():
    ap = argparse.ArgumentParser(
        description="Global Hill fits for multiple clones (CSV inputs) with direct contrast calculation."
    )
    ap.add_argument("csv_files", nargs="+", type=pathlib.Path)
    ap.add_argument("--out-prefix", default="titration_panel")
    ap.add_argument("--palette", default="tab10")
    ap.add_argument("--sat", type=float, default=1.0, help="Saturation multiplier")
    ap.add_argument("--val", type=float, default=1.0, help="Value/brightness multiplier")
    ap.add_argument("--pt-size", type=float, default=34, help="Scatter point size (pt²)")
    ap.add_argument("--vector", choices=("pdf", "svg"), default="pdf")
    ap.add_argument("--show-n", action="store_true", help="Append (n=…) to legend labels")
    ap.add_argument(
        "--fit-text", dest="fit_text", action="store_true", default=False,
        help="Display fit parameters on plot"
    )
    ap.add_argument("--no-fit-text", dest="fit_text", action="store_false")
    ap.add_argument("--fit-hide-name", action="store_true",
                    help="Omit clone name inside fit-text")
    ap.add_argument(
        "--excel-engine", choices=("openpyxl", "xlsxwriter", "none"),
        help="Force Excel writer engine (default: auto)"
    )
    ap.add_argument("--legend-fs", type=float, default=8, help="Legend font size (pt)")
    ap.add_argument("--fig-w", type=float, default=6.0, help="Figure width (in)")
    ap.add_argument("--fig-h", type=float, default=4.0, help="Figure height (in)")
    ap.add_argument("--fig-w-mm", type=float, help="Figure width (mm, overrides --fig-w)")
    ap.add_argument("--fig-h-mm", type=float, help="Figure height (mm, overrides --fig-h)")
    ap.add_argument("--font", help="Global font family override")
    ap.add_argument("--x-units", choices=("nM", "pCa"), default="nM")
    ap.add_argument("--ec50-fs", type=float, default=6,
                    help="Font size for fit-text numbers (pt)")
    ap.add_argument("--fit-gap", type=float, default=0.055,
                    help="Vertical spacing between fit-text lines")
    ap.add_argument("--axis-fs", type=float, default=10, help="Axis label font size (pt)")
    ap.add_argument("--tick-fs", type=float, default=8, help="Tick label font size (pt)")
    ap.add_argument("--no-excel", action="store_true", help="Disable Excel workbook output")
    ap.add_argument("--smart-alpha", action="store_true", 
                    help="Use smart alpha adjustment for overlapping markers (reduces alpha in dense regions)")
    ap.add_argument("--alpha-base", type=float, default=0.8, 
                    help="Base alpha value for markers (0-1, default: 0.8)")
    ap.add_argument("--alpha-min", type=float, default=0.3, 
                    help="Minimum alpha value for overlapping markers (0-1, default: 0.3)")
    ap.add_argument("--alpha-radius", type=float, default=0.05, 
                    help="Radius factor for smart alpha detection (0.01-0.2, default: 0.05)")
    ap.add_argument("--auto-fonts", action="store_true", 
                    help="Automatically optimize font sizes to fit available space")
    ap.add_argument("--font-scale", type=float, default=1.0, 
                    help="Scaling factor for auto-optimized fonts (0.5-2.0, default: 1.0)")
    ap.add_argument("--auto-markers", action="store_true", 
                    help="Automatically optimize marker sizes based on figure dimensions")
    ap.add_argument("--marker-scale", type=float, default=1.0, 
                    help="Scaling factor for auto-optimized markers (0.5-2.0, default: 1.0)")
    ap.add_argument("--auto-lines", action="store_true", 
                    help="Automatically optimize line weights based on figure dimensions")
    ap.add_argument("--line-scale", type=float, default=1.0, 
                    help="Scaling factor for auto-optimized lines (0.5-2.0, default: 1.0)")
    ap.add_argument("--fit-lw", type=float, help="Manual line weight for fit curves (overrides auto-lines)")
    ap.add_argument("--ci-lw", type=float, help="Manual line weight for confidence intervals (overrides auto-lines)")
    
    # ── NEW marker/fill CLI options ───────────────────────────────
    ap.add_argument("--markers", default="auto", 
                    help="Comma-separated list of marker symbols, 'auto' to cycle common shapes, or predefined sets: 'basic', 'geometric', 'arrows', 'stars', 'hollow'")
    ap.add_argument("--marker-set", choices=["auto", "basic", "geometric", "arrows", "stars", "hollow"], default="auto",
                    help="Predefined marker sets: basic (o,s,D), geometric (o,s,D,^,v,<,>), arrows (^,v,<,>,P), stars (*,h,H,8), hollow (o,s,D with hollow variants)")
    ap.add_argument("--fills", default="full", 
                    help="Comma-separated fill styles (full, left, right, bottom, top, none) to cycle (note: fillstyle not supported in scatter plots)")
    
    # ── Direct contrast options ───────────────────────────────────
    ap.add_argument("--ca-threshold", type=float, default=10.0,
                    help="Maximum Ca2+ concentration for baseline calculation (nM, default: 10.0)")
    ap.add_argument("--use-hill-contrast", action="store_true", default=False,
                    help="Use Hill fit parameters for contrast calculation (less accurate uncertainties)")
    
    return ap.parse_args()


# ───────────────────────────── main ───────────────────────────────
def main():
    a = cli()

    if a.font:
        plt.rcParams["font.family"] = a.font

    # Convert mm to inches if specified
    if a.fig_w_mm is not None:
        a.fig_w = a.fig_w_mm / 25.4  # Convert mm to inches
    if a.fig_h_mm is not None:
        a.fig_h = a.fig_h_mm / 25.4  # Convert mm to inches

    # Auto-optimize font sizes if requested
    if a.auto_fonts:
        n_clones = len(a.csv_files)
        opt_axis_fs, opt_tick_fs, opt_legend_fs, opt_ec50_fs = calculate_optimal_font_sizes(
            a.fig_w, a.fig_h, n_clones, a.font_scale
        )
        a.axis_fs = opt_axis_fs
        a.tick_fs = opt_tick_fs
        a.legend_fs = opt_legend_fs
        a.ec50_fs = opt_ec50_fs

    # Auto-optimize marker sizes if requested
    if a.auto_markers:
        n_clones = len(a.csv_files)
        opt_marker_size = calculate_optimal_marker_size(
            a.fig_w, a.fig_h, n_clones, a.marker_scale
        )
        a.pt_size = opt_marker_size

    # Auto-optimize line weights if requested
    if a.auto_lines:
        opt_fit_lw, opt_ci_lw = calculate_optimal_line_weights(
            a.fig_w, a.fig_h, a.line_scale
        )
        # Only set if manual values weren't provided
        if a.fit_lw is None:
            a.fit_lw = opt_fit_lw
        if a.ci_lw is None:
            a.ci_lw = opt_ci_lw
    else:
        # Set defaults if auto-lines not used and manual values not provided
        if a.fit_lw is None:
            a.fit_lw = 2.0
        if a.ci_lw is None:
            a.ci_lw = 0.0

    # Generate filename tags
    filename_tags = generate_filename_tags(a)
    if filename_tags:
        a.out_prefix = f"{a.out_prefix}_{filename_tags}"
    
    # Set Excel engine (None if --no-excel is used)
    engine = None if a.no_excel else pick_engine(a.excel_engine)

    clones: List[dict] = []
    for csv in a.csv_files:
        data = load_clone(csv)
        x = data["Ca_nM"].to_numpy(float)
        y = data["signal_norm"].to_numpy(float)
        
        # Standard Hill fit
        popt, pcov = fit_hill(x, y)
        ymin, EC50, nH = popt
        ymin_se, EC50_se, nH_se = np.sqrt(np.diag(pcov))
        
        if a.use_hill_contrast:
            # Use Hill fit parameters for contrast calculation (less accurate)
            contrast = 1 / ymin
            contrast_se = ymin_se / ymin**2
            delta = (1 - ymin) / ymin
            delta_se = ymin_se / ymin**2
            direct_result = None
        else:
            # Calculate contrast directly from low-Ca data points (default, more accurate)
            direct_result = calculate_direct_contrast(data, a.ca_threshold)
            if direct_result is None:
                print(f"⚠  No data points below {a.ca_threshold} nM for {csv.name}, using Hill fit method")
                contrast = 1 / ymin
                contrast_se = ymin_se / ymin**2
                delta = (1 - ymin) / ymin
                delta_se = ymin_se / ymin**2
                direct_result = None
            else:
                contrast = direct_result['contrast']
                contrast_se = direct_result['contrast_se']
                delta = direct_result['delta']
                delta_se = direct_result['delta_se']
        
        clones.append(
            dict(
                csv=csv,
                data=data,
                x=x,
                y=y,
                n_trials=data["run"].nunique(),
                popt=popt,
                pcov=pcov,
                ymin=ymin,
                ymin_se=ymin_se,
                EC50=EC50,
                EC50_se=EC50_se,
                nH=nH,
                nH_se=nH_se,
                contrast=contrast,
                contrast_se=contrast_se,
                delta=delta,
                delta_se=delta_se,
                direct_result=direct_result
            )
        )
    clones.sort(key=lambda d: d["EC50"])  # sort by EC50

    cmap = plt.colormaps[a.palette]
    positions = np.linspace(0, 1, len(clones)) if len(clones) > 1 else [0.5]
    colours = [
        tweak_rgb(cmap(p)[:3], a.sat, a.val) for p in positions
    ]

    # ── Marker shapes & fill styles ───────────────────────────────
    n_clones = len(clones)
    
    # Predefined marker sets
    marker_sets = {
        "auto": ["o", "s", "D", "^", "v", "<", ">", "P", "X", "*", "h", "H", "8", "p"],
        "basic": ["o", "s", "D"],
        "geometric": ["o", "s", "D", "^", "v", "<", ">"],
        "arrows": ["^", "v", "<", ">", "P"],
        "stars": ["*", "h", "H", "8"],
        "hollow": ["o", "s", "D", "^", "v", "<", ">"]
    }
    
    if a.marker_set != "auto":
        # Use predefined marker set
        if a.marker_set in marker_sets:
            markers = list(itertools.islice(itertools.cycle(marker_sets[a.marker_set]), n_clones))
        else:
            sys.exit(f"❌ Unknown marker set: {a.marker_set}")
    elif a.markers.lower() == "auto":
        # Use default auto markers
        markers = list(itertools.islice(itertools.cycle(marker_sets["auto"]), n_clones))
    else:
        # Use user-provided markers
        user_markers = [m.strip() for m in a.markers.split(",") if m.strip()]
        if not user_markers:
            sys.exit("❌ --markers provided but no valid symbols parsed.")
        markers = list(itertools.islice(itertools.cycle(user_markers), n_clones))

    fill_styles_input = [f.strip() for f in a.fills.split(",") if f.strip()]
    valid_fills = {"full", "left", "right", "bottom", "top", "none"}
    if not set(fill_styles_input).issubset(valid_fills):
        sys.exit("❌ --fills contains invalid fillstyle; allowed: " + ", ".join(sorted(valid_fills)))
    fill_styles = list(itertools.islice(itertools.cycle(fill_styles_input), n_clones))

    # ── Prepare all marker data for smart alpha calculation ───────────────────────────────
    all_marker_data = []
    if a.smart_alpha:
        for idx, info in enumerate(clones):
            x_nM = info["x"]
            y_norm = info["y"]
            x_plot = to_pca(x_nM) if a.x_units == "pCa" else x_nM
            all_marker_data.append({
                'x': x_plot,
                'y': y_norm,
                'dataset_idx': idx,
                'clone_name': info["csv"].stem
            })

    fig, ax = plt.subplots(figsize=(a.fig_w, a.fig_h), dpi=150)
    summary_records = []

    to_pca = lambda arr: -np.log10(arr * 1e-9)
    x_label = (
        r"[Ca$^{2+}$] (nM)"
        if a.x_units == "nM"
        else r"pCa  ($-\log_{10}[M]$)"
    )

    for idx, info in enumerate(clones):
        clone_name = info["csv"].stem
        n_trials = info["n_trials"]
        label = (
            f"{clone_name} "
            + rf"$\mathit{{(n={n_trials})}}$"
            if a.show_n
            else clone_name
        )
        colour = colours[idx]
        marker = markers[idx]
        fillstyle = fill_styles[idx]

        x_nM = info["x"]
        y_norm = info["y"]
        popt = info["popt"]
        pcov = info["pcov"]
        ymin = info["ymin"]
        ymin_se = info["ymin_se"]
        EC50 = info["EC50"]
        EC50_se = info["EC50_se"]
        nH = info["nH"]
        nH_se = info["nH_se"]
        contrast = info["contrast"]
        contrast_se = info["contrast_se"]
        delta = info["delta"]
        delta_se = info["delta_se"]

        # scatter points with custom markers
        x_plot = to_pca(x_nM) if a.x_units == "pCa" else x_nM
        
        # Calculate alpha values
        if a.smart_alpha:
            alphas = calculate_smart_alpha_cross_dataset(all_marker_data, idx, a.alpha_base, a.alpha_min, a.pt_size)
            # Plot points individually with different alpha values (no labels)
            for i in range(len(x_plot)):
                ax.scatter(
                    x_plot[i:i+1],
                    y_norm[i:i+1],
                    s=a.pt_size,
                    color=colour,
                    marker=marker,
                    edgecolor="k",
                    lw=0.25,
                    alpha=alphas[i],
                    zorder=3,
                )
            # Add a separate legend entry with full opacity
            ax.scatter([], [], s=a.pt_size, color=colour, marker=marker, 
                      edgecolor="k", lw=0.25, alpha=a.alpha_base, label=label)
        else:
            # Use scatter for the data points (fillstyle not supported in scatter)
            ax.scatter(
                x_plot,
                y_norm,
                s=a.pt_size,
                color=colour,
                marker=marker,
                edgecolor="k",
                lw=0.25,
                alpha=a.alpha_base,
                label=label,
                zorder=3,
            )

        # fitted curve & CI
        xs = np.logspace(np.log10(x_nM.min() / 3), np.log10(x_nM.max() * 3), 400)
        ys = hill3(xs, *popt)
        xs_plot = to_pca(xs) if a.x_units == "pCa" else xs
        ax.plot(xs_plot, ys, color=colour, lw=a.fit_lw)
        lo, up = ci_band(xs, popt, pcov)
        ax.fill_between(xs_plot, lo, up, color=colour, alpha=0.20, linewidth=a.ci_lw)

        # per-clone workbook
        if engine and not a.no_excel:
            wb_path = pathlib.Path(f"{a.out_prefix}_{clone_name}.xlsx")
            with pd.ExcelWriter(wb_path, engine=engine) as xl:
                info["data"].to_excel(xl, "raw", index=False)
                
                # Create fit parameters DataFrame
                fit_params_data = {
                    "parameter": [
                        "ymin",
                        "EC50_nM",
                        "Hill_n",
                        "Contrast_Fmax/Fmin",
                        "DeltaF_over_F0",
                    ],
                    "value": [ymin, EC50, nH, contrast, delta],
                    "stderr": [ymin_se, EC50_se, nH_se, contrast_se, delta_se],
                }
                
                # Add direct contrast info if used
                if not a.use_hill_contrast and info["direct_result"] is not None:
                    fit_params_data["parameter"].extend([
                        "mean_low_ca_signal",
                        "sem_low_ca_signal",
                        "n_low_ca_points",
                        "ca_threshold_nM"
                    ])
                    fit_params_data["value"].extend([
                        info["direct_result"]["mean_signal"],
                        info["direct_result"]["sem_signal"],
                        info["direct_result"]["n_low_ca_points"],
                        info["direct_result"]["ca_threshold"]
                    ])
                    fit_params_data["stderr"].extend([np.nan, np.nan, np.nan, np.nan])
                
                pd.DataFrame(fit_params_data).to_excel(xl, "fit_params", index=False)
            print(f"Workbook → {wb_path}")
        elif a.no_excel:
            print("⚠  Excel output disabled with --no-excel flag.")
        else:
            print("⚠  Excel writer not available; workbook skipped.")

        summary_records.append(
            dict(
                clone=clone_name,
                n_trials=n_trials,
                ymin=ymin,
                ymin_se=ymin_se,
                EC50_nM=EC50,
                EC50_se=EC50_se,
                Hill_n=nH,
                Hill_se=nH_se,
                Contrast=contrast,
                Contrast_se=contrast_se,
                DeltaF_over_F0=delta,
                DeltaF_over_F0_se=delta_se,
            )
        )

        # on-plot fit-text
        if a.fit_text:
            prefix = "" if a.fit_hide_name else f"{clone_name}: "
            ax.text(
                0.02,
                0.98 - idx * a.fit_gap,
                (
                    prefix
                    + f"EC$_{{50}}$={EC50:.1f}±{EC50_se:.1f} nM; "
                    + r"$n_H$="
                    + f"{nH:.2f}±{nH_se:.2f}; "
                    + r"$\Delta F/F_{0}$="
                    + f"{delta:.2f}±{delta_se:.2f}"
                ),
                transform=ax.transAxes,
                color=colour,
                fontsize=a.ec50_fs,
                va="top",
                ha="left",
                bbox=dict(facecolor="white", alpha=0.6, pad=0.2, edgecolor="none"),
            )

    # ───────────── axis formatting ─────────────
    if a.x_units == "nM":
        ax.set_xscale("log")
        ax.set_xlim(
            min(c["x"].min() for c in clones) / 3,
            max(c["x"].max() for c in clones) * 3,
        )
        ax.xaxis.set_major_locator(LogLocator(base=10, numticks=15))
        ax.xaxis.set_minor_locator(LogLocator(base=10, subs=range(2, 10)))
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.tick_params(which="minor", length=3)
    else:  # pCa
        all_pca = np.concatenate([to_pca(c["x"]) for c in clones])
        ax.set_xlim(all_pca.max() + 0.2, all_pca.min() - 0.2)

    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel(x_label, fontsize=a.axis_fs)
    ax.set_ylabel("BL emission (normalized to maximum)", fontsize=a.axis_fs)
    ax.tick_params(axis='both', which='major', labelsize=a.tick_fs)
    ax.legend(loc="lower right", frameon=False, fontsize=a.legend_fs, 
              handlelength=0.5, handletextpad=0.2, columnspacing=0.5)
    fig.tight_layout()

    fig.savefig(f"{a.out_prefix}.png", dpi=300)
    fig.savefig(f"{a.out_prefix}.{a.vector}")
    plt.close(fig)

    if not a.no_excel:
        pd.DataFrame(summary_records).to_excel(
            f"{a.out_prefix}_summary.xlsx", index=False
        )
        print("✔  All files written.")
    else:
        print("✔  Plot files written (Excel output disabled).")


if __name__ == "__main__":
    main() 