## **/CFz&FFz_validation_in_N2a**  
&nbsp;&emsp;&emsp;&emsp; Corresponds to supplementary figures 7and 8, in both `.py` (`src/`) and Jupyter (`notebooks/`) format  
&nbsp;&emsp;&emsp;&emsp; Demo_files are found in each sub directory 

### &nbsp;│⎯   **/n2a_biolum_imaging**    
### &nbsp;&emsp;  │⎯  **background_subtraction_&_traces_viewer.ipynb**  
&nbsp;&nbsp;&emsp;&emsp;First step in analysis, uses `.csv` files from ImageJ ROI measurements (see Methods).    
&nbsp;&nbsp;&emsp;&emsp;Used to generate background-subtracted ROIs for subsequent analysis  
&nbsp;&nbsp;&emsp;&emsp;Also shows individual traces — good for checking if cells died/disappeared to exclude from analysis   
&nbsp;&nbsp;&emsp;&emsp;&emsp;- Example input file, in demo_files see *"FFz1_100uM_results.csv"    
&nbsp;&nbsp;&emsp;&emsp;&emsp;- Example output files, in demo_files see *"FFz1_100uM_results_minus_bgd.csv", "FFz1_100uM_individual_traces.pdf"     
###  &nbsp;&emsp;  │⎯  **lum_timecourse_analysis.ipynb**  
&nbsp;&emsp;&emsp;Next step — must be run to get output files for plotting code (`bessel_filtered`, `deltaL/L`)   
&nbsp;&emsp;&emsp;Note that output directories are created, that organization will be needed for `dotplots.ipynb`    
&nbsp;&emsp;&emsp;- Example input file, in demo_files see *"FFz1_100uM_results_minus_bgd.csv"* (output from `background_subtraction`)   
&nbsp;&emsp;&emsp;- Example output files, in demo_files see:     
&nbsp;&emsp;&emsp;&nbsp;&emsp;&emsp; *FFz1_100uM_results_luminescence.png* - plot before bessel-filter  
&nbsp;&emsp;&emsp;&nbsp;&emsp;&emsp; *FFz1_100uM_bessel_filtered.png* - plot after bessel-filter   
&nbsp;&emsp;&emsp;&nbsp;&emsp;&emsp; *FFz1_100uM_bessel_filtered.csv* - output .csv of bessel-filtered data, used in `dotplots.ipynb`    
&nbsp;&emsp;&emsp;&nbsp;&emsp;&emsp; *FFz1_100uM_deltaLoverL_bessel_filtered.csv* - output of deltaLoverL calculation, used in `dotplots.ipynb`  
###  &nbsp;&emsp;  │⎯   **dotplots.ipynb**  
&nbsp;&emsp;&emsp;Plotting (panels B, D, F, & G)  
&nbsp;&emsp;&emsp;To plot all data, I would parse the output files created from each "input file" for that luc/concentration  
&nbsp;&emsp;&emsp;*(from`lum_timecourse_analysis.ipynb`)*    
&nbsp;&emsp;&emsp; - Just add the mNeonGreen fluorescence file to the given directory, see *"FFz1_100uM_mNeonGreen_results.csv"* in demo_files  
&nbsp;&emsp;&emsp; - Example raw plot output (use a .svg and edited in illustrator for final version), see "*Combined_CFz_FFz_Dotplots_MeanCI.png*"    
###  &nbsp;&emsp;  │⎯  **lum_timecourses_figure.ipynb**  
&nbsp;&emsp;&emsp;Plots bessel-filtered.csv files in one output figure (panels A, C, and E)  
&nbsp;&emsp;&emsp; - Example raw plot output (use a .svg and edited in illustrator for final version), see *"Luminescence_Timecourses_100uM.png*"
 ### &nbsp;│⎯   **/substrate_dose_response**    
###  &nbsp;&emsp;  │⎯  **dose_response_curve.ipynb**  
&nbsp;&emsp;&emsp;All-in-one dose response analysis    
&nbsp;&emsp;&emsp;Calculates AUC from input `.csv` files (manually combined from microplate reader outputs)   
&nbsp;&emsp;&emsp;Sets model, plots curve, and prints fitted parameters   
&nbsp;&emsp;&emsp; - Example input file, in demo_files see *"CFz_forplotting.csv"* Concentrations and replicate numbers are added manually post-hoc   
 ### &nbsp;│⎯  **requirements.txt**  
&nbsp;&emsp;&emsp;Basic with Anaconda package in ~2024, see for version numbers. Used MacOS 14.6.1   


