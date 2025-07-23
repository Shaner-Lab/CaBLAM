# Titration Fit Global

A Python script for performing global Hill fits on multiple calcium sensor titration datasets with direct contrast calculation from low-calcium data points for realistic uncertainty estimates.

## Installation

### Prerequisites
```bash
pip install matplotlib numpy pandas scipy
```

### Optional Dependencies
For Excel output:
```bash
pip install openpyxl  # or xlsxwriter
```

## Basic Usage

### Simple Analysis
```bash
python3 titration_fit_global.py data/CaBLAM.csv data/CaMBI.csv --out-prefix demo
```

### Using All Example Datasets
```bash
python3 titration_fit_global.py data/*.csv --out-prefix all_sensors
```

## Advanced Usage Examples

### Custom Calcium Threshold
```bash
python3 titration_fit_global.py data/*.csv \
    --ca-threshold 5.0 \
    --fig-w-mm 180 --fig-h-mm 120 \
    --auto-fonts --auto-markers --auto-lines \
    --palette plasma --marker-set geometric \
    --smart-alpha --fit-text \
    --out-prefix publication
```

### Higher Threshold Example (CaBLAM_294W)
```bash
python3 titration_fit_global.py data/CaBLAM_294W.csv --ca-threshold 50 --out-prefix CaBLAM_294W_50nM
```
**Results:**
- **Contrast**: 625.0 ± 39.2
- **Baseline points**: 9 data points below 50 nM
- **Mean baseline signal**: 0.0016 ± 0.0001

### Conservative Threshold
```bash
python3 titration_fit_global.py data/*.csv \
    --ca-threshold 20.0 \
    --out-prefix conservative
```

### Aggressive Threshold
```bash
python3 titration_fit_global.py data/*.csv \
    --ca-threshold 2.0 \
    --out-prefix aggressive
```

## Command Line Options

### Contrast Calculation Options
- `--ca-threshold`: Maximum Ca2+ concentration for baseline (nM, default: 10.0)
- `--use-hill-contrast`: Use Hill fit parameters for contrast calculation (less accurate)

### Visualization Options
- `--auto-fonts`, `--auto-markers`, `--auto-lines`
- `--palette`, `--marker-set`, `--smart-alpha`
- `--fit-text`, `--x-units`, `--no-excel`
- And many other customization options

## Output Files

The script generates the same output files as the original, with enhanced uncertainty estimates:

1. **Plot files**: PNG and PDF with publication-ready formatting
2. **Excel workbooks**: 
   - Summary file with direct contrast uncertainties
   - Individual clone files with direct calculation statistics
   - Additional columns: `mean_low_ca_signal`, `sem_low_ca_signal`, `n_low_ca_points`, `ca_threshold_nM`

## Example Output

### Command Executed
```bash
python3 titration_fit_global.py data/*.csv --out-prefix example --fit-text --auto-fonts --auto-markers --auto-lines
```

### Generated Files
- `example_af_am_al_ft.png` - High-resolution plot (318 KB)
- `example_af_am_al_ft.pdf` - Vector graphics (58 KB)
- `example_af_am_al_ft_summary.xlsx` - Global summary with contrast uncertainties
- Individual clone workbooks:
  - `example_af_am_al_ft_CaBLAM.xlsx` (7.7 KB)
  - `example_af_am_al_ft_CaBLAM_294W.xlsx` (6.9 KB)
  - `example_af_am_al_ft_CaBLAM_332W.xlsx` (6.9 KB)
  - `example_af_am_al_ft_CaMBI.xlsx` (8.0 KB)
  - `example_af_am_al_ft_GeNL(Ca2+)_480.xlsx` (8.7 KB)

*Note: Filename tags indicate auto-fonts (af), auto-markers (am), auto-lines (al), and fit-text (ft)*

### Output File Naming Convention

The script generates descriptive filenames with tags indicating the settings used:
- **af**: Auto-fonts enabled
- **am**: Auto-markers enabled  
- **al**: Auto-lines enabled
- **ft**: Fit-text enabled
- **pl**: Plasma palette
- **vi**: Viridis palette
- **sa**: Smart alpha enabled
- **ne**: No Excel output
- **pCa**: pCa units instead of nM
- And many other configuration tags

This makes it easy to identify which settings were used for each analysis.

### Example Results

| Clone | EC50 (nM) | Contrast | Contrast SE | ΔF/F0 | ΔF/F0 SE |
|-------|-----------|----------|-------------|-------|----------|
| CaMBI | 59.6 ± 5.9 | 4.73 | 0.29 | 3.73 | 0.29 |
| CaBLAM_332W | 281.1 ± 9.3 | 68.0 | 5.3 | 67.0 | 5.3 |
| CaBLAM | 439.3 ± 13.9 | 83.0 | 9.4 | 82.0 | 9.4 |
| GeNL(Ca2+) | 457.1 ± 20.3 | 3.56 | 0.33 | 2.56 | 0.33 |
| CaBLAM_294W | 3066.5 ± 94.6 | 748.1 | 27.2 | 747.1 | 27.2 |

*Realistic uncertainty estimates that reflect actual experimental precision*

### Contrast Statistics Examples

**CaBLAM analysis:**
- **mean_low_ca_signal**: 0.0120 ± 0.0014
- **n_low_ca_points**: 6
- **ca_threshold**: 10.0 nM

**CaMBI analysis:**
- **mean_low_ca_signal**: 0.2114 ± 0.0129
- **n_low_ca_points**: 6
- **ca_threshold**: 10.0 nM

**CaBLAM_294W analysis (10 nM threshold):**
- **mean_low_ca_signal**: 0.0013 ± 0.0004
- **n_low_ca_points**: 6
- **ca_threshold**: 10.0 nM

**CaBLAM_294W analysis (50 nM threshold):**
- **mean_low_ca_signal**: 0.0016 ± 0.0001
- **n_low_ca_points**: 9
- **ca_threshold**: 50.0 nM

*Higher thresholds provide more baseline points but may include some calcium-dependent signal*

## Contrast Statistics

The Excel output includes additional columns:
- **mean_low_ca_signal**: Mean signal at low calcium concentrations
- **sem_low_ca_signal**: Standard error of the mean
- **n_low_ca_points**: Number of data points used for baseline
- **ca_threshold_nM**: Calcium threshold used

A higher number of low-Ca points provides more robust baseline estimation.

## When to Use Different Methods

**Default method (direct contrast) is best when:**
- You have sufficient data points at low calcium concentrations
- You want realistic uncertainty estimates based on experimental scatter
- Publishing results where uncertainty quantification is critical

**Hill fit method (`--use-hill-contrast`) is better when:**
- Limited low-Ca data points are available
- Quick analysis is needed
- The Hill fit provides good baseline estimation

## Choosing the Calcium Threshold

**Lower threshold (2-5 nM):**
- More conservative baseline estimation
- May have fewer data points
- Better for sensors with very low baseline

**Default threshold (10 nM):**
- Good balance between data availability and baseline accuracy
- Suitable for most calcium sensors

**Higher threshold (15-50 nM):**
- More data points for baseline
- May include some calcium-dependent signal
- Better for sensors with lower calcium affinities
- Example: CaBLAM_294W with 50 nM threshold gives 9 baseline points vs 6 with 10 nM

## Performance Considerations

- **Direct contrast adds negligible time** (simple statistical calculation)
- **Memory usage** unchanged
- **Robustness** depends on number of low-Ca data points
- **Automatic fallback** to Hill method if insufficient data

## Method Comparison

**Default method (direct contrast) advantages:**
- Realistic uncertainty estimates based on actual data scatter
- No reliance on model extrapolation
- Simple and intuitive calculation
- Reflects true experimental precision

**Hill fit method advantages:**
- Works with limited low-Ca data
- Provides smooth curve fitting
- Accounts for model uncertainty

## Troubleshooting

- **No low-Ca data**: Increase `--ca-threshold` or use original method
- **Large uncertainties**: Check data quality at low calcium concentrations
- **Inconsistent results**: Verify calcium threshold is appropriate for your sensor

## Example Datasets

This directory includes five example calcium sensor datasets:
1. **CaBLAM.csv** - Original CaBLAM sensor
2. **CaBLAM_294W.csv** - CaBLAM with W294 mutation  
3. **CaBLAM_332W.csv** - CaBLAM with W332 mutation
4. **CaMBI.csv** - CaMBI sensor
5. **GeNL(Ca2+)_480.csv** - GeNL calcium sensor

## Version

This script is version 4.0 (2025-07-21). 
