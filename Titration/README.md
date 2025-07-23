# Calcium Sensor Titration Analysis

A Python toolkit for analyzing calcium sensor titration data with realistic uncertainty estimation.

## Quick Start

```bash
# Install dependencies
pip install matplotlib numpy pandas scipy openpyxl

# Run analysis on example data
python3 titration_fit_global.py data/*.csv --out-prefix my_analysis
```

## Repository Structure

```
Titration/
├── titration_fit_global.py    # Main analysis script
├── data/                      # Input datasets
│   ├── CaBLAM.csv
│   ├── CaBLAM_294W.csv
│   ├── CaBLAM_332W.csv
│   ├── CaMBI.csv
│   └── GeNL(Ca2+)_480.csv
├── examples/                  # Example outputs
│   ├── example_af_am_al_ft_*  # Complete 5-sensor analysis
│   └── CaBLAM_294W_50nM_*     # Single sensor with custom threshold
└── docs/                      # Documentation
    └── README.md              # Detailed usage guide
```

## Key Features

- **Direct Contrast Calculation**: Realistic uncertainty estimates from low-Ca data points
- **Global Hill Fits**: Simultaneous analysis of multiple sensors
- **Publication-Ready Plots**: Auto-optimized styling and formatting
- **Comprehensive Output**: Excel workbooks with detailed statistics

## Documentation

For detailed usage instructions, examples, and advanced features, see:
- **[docs/README.md](docs/README.md)** - Complete documentation

## Example Results

| Sensor | EC50 (nM) | Contrast | Uncertainty |
|--------|-----------|----------|-------------|
| CaMBI | 59.6 ± 5.9 | 4.73 | ±0.29 |
| CaBLAM | 439.3 ± 13.9 | 83.0 | ±9.4 |
| CaBLAM_294W | 3066.5 ± 94.6 | 748.1 | ±27.2 |

*Realistic uncertainties based on experimental scatter, not model extrapolation*

## Version

v4.0 (2025-07-21) 