# CaBLAM Repository

A comprehensive repository containing analysis tools, validation data, and research materials for **CaBLAM!** (Calcium Bioluminescent Activity Monitor), a high-contrast bioluminescent Ca²⁺ indicator derived from an engineered *Oplophorus gracilirostris* luciferase.

## 📖 Publication

This repository accompanies the research paper:

**Lambert et al., *CaBLAM! A high-contrast bioluminescent Ca²⁺ indicator derived from an engineered Oplophorus gracilirostris luciferase***

- **Preprint**: [https://www.biorxiv.org/content/10.1101/2023.06.25.546478v3](https://www.biorxiv.org/content/10.1101/2023.06.25.546478v3)

## 🗂️ Repository Structure

### 📊 `cablam_in_vivo_analysis/`
**MATLAB scripts for in vivo mouse data analysis**

Contains MATLAB scripts to reproduce figures and statistics from the in vivo mouse experiments. Includes analysis for:
- Infusion experiments
- Long-duration recordings  
- Running wheel experiments
- Trial-based analysis

**Quick Start:**
1. Download demo data from: https://doi.org/10.26300/7sg5-w257
2. Place `demo_data` folder inside `cablam_in_vivo_analysis/`
3. Open `run_all_in_vivo.m` in MATLAB
4. Set path variable and run

**Requirements:** MATLAB R2024b with Image Processing and Statistics toolboxes

### 🔬 `CFz&FFz_validation_in_N2a/`
**Python analysis for N2a cell validation experiments**

Corresponds to Supplementary Figures 7 and 8, containing:
- **`n2a_biolum_imaging/`**: Bioluminescence imaging analysis
  - Background subtraction and trace viewing
  - Luminescence timecourse analysis with Bessel filtering
  - Dot plot generation for statistical comparisons
  - Timecourse figure generation
- **`substrate_dose_response/`**: Dose-response curve analysis

**Quick Start:**
```bash
cd CFz&FFz_validation_in_N2a
pip install -r requirements.txt
# Run Jupyter notebooks in notebooks/ directory
```

### 🧪 `Titration/`
**Python toolkit for calcium sensor titration analysis**

Advanced analysis tool for calcium sensor characterization with realistic uncertainty estimation.

**Key Features:**
- Direct contrast calculation from low-Ca data points
- Global Hill fits for multiple sensors
- Publication-ready plots with auto-optimized styling
- Comprehensive Excel output with detailed statistics

**Quick Start:**
```bash
cd Titration
pip install matplotlib numpy pandas scipy openpyxl
python3 titration_fit_global.py data/*.csv --out-prefix my_analysis
```

**Example Results:**
| Sensor | EC50 (nM) | Contrast | Uncertainty |
|--------|-----------|----------|-------------|
| CaMBI | 59.6 ± 5.9 | 4.73 | ±0.29 |
| CaBLAM | 439.3 ± 13.9 | 83.0 | ±9.4 |
| CaBLAM_294W | 3066.5 ± 94.6 | 748.1 | ±27.2 |

### 🐟 `zebrafish data & code/`
**Zebrafish experimental data and analysis**

Contains experimental data from zebrafish studies with corresponding analysis scripts.

## 🚀 Getting Started

### Prerequisites

**For MATLAB analysis:**
- MATLAB R2024b or later
- Image Processing Toolbox
- Statistics and Machine Learning Toolbox

**For Python analysis:**
- Python 3.7+
- See individual `requirements.txt` files in each directory

### Installation

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd CaBLAM_repo
   ```

2. **Install Python dependencies:**
   ```bash
   # For CFz&FFz validation
   cd CFz&FFz_validation_in_N2a
   pip install -r requirements.txt
   
   # For titration analysis
   cd ../Titration
   pip install matplotlib numpy pandas scipy openpyxl
   ```

3. **Download demo data:**
   - In vivo data: https://doi.org/10.26300/7sg5-w257
   - Place in appropriate directories as described in individual README files

## 📋 Usage Examples

### In Vivo Analysis (MATLAB)
```matlab
% Set path to cablam_in_vivo_analysis folder
pth = 'C:\Users\user\Desktop\cablam_in_vivo_analysis\';
run('run_all_in_vivo.m');
```

### Titration Analysis (Python)
```bash
# Analyze all sensors
python3 titration_fit_global.py data/*.csv --out-prefix all_sensors

# Custom calcium threshold
python3 titration_fit_global.py data/CaBLAM_294W.csv --ca-threshold 50 --out-prefix CaBLAM_294W_50nM
```

### N2a Validation (Python/Jupyter)
```bash
# Start Jupyter notebook
jupyter notebook notebooks/
# Open background_subtraction_&_traces_viewer.ipynb
```

## 🤝 Contributing

This repository is released under the [CC0 1.0 Universal](LICENSE) license, allowing unrestricted use, modification, and distribution.

## 📞 Contact

For questions about the CaBLAM sensor or analysis tools, please refer to the original publication or contact the corresponding authors.

## 📚 Related Resources

- **Demo Data**: https://doi.org/10.26300/7sg5-w257
- **Preprint**: https://www.biorxiv.org/content/10.1101/2023.06.25.546478v3
- **Individual README files** in each directory for detailed usage instructions

---

*This repository contains all analysis tools and data necessary to reproduce the results presented in the CaBLAM publication.* 