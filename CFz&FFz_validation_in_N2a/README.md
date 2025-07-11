### **/CFz&FFz_validation_in_N2a**  
&nbsp; │  &emsp;&emsp;&emsp; *Corresponds to supplementary figures XX and XX, in both `.py` (`src/`) and Jupyter (`notebooks/`) format*    
 ### &nbsp;│⎯   **/n2a_biolum_imaging**    
&nbsp; │&emsp;&emsp;&emsp;  │⎯  **background_subtraction_&_traces_viewer.ipynb**  
&nbsp; │&emsp;&emsp;&emsp;  │&emsp;&emsp;&emsp;&emsp;*First step in analysis, uses `.csv` files from ImageJ ROI measurements (see Methods)*  
&nbsp; │&emsp;&emsp;&emsp;  │&emsp;&emsp;&emsp;&emsp;*Also shows individual traces — good for checking if cells died/disappeared to exclude from analysis*    
&nbsp; │&emsp;&emsp;&emsp;  │⎯  **lum_timecourse_analysis.ipynb**  
&nbsp; │&emsp;&emsp;&emsp;  │&emsp;&emsp;&emsp;&emsp;*Next step — must be run to get output files for plotting code (`bessel_filtered`, `deltaL/L`)*  
&nbsp; │&emsp;&emsp;&emsp;  │⎯  **dotplots.ipynb**  
&nbsp; │&emsp;&emsp;&emsp;  │&emsp;&emsp;&emsp;&emsp;*Plotting (panels B, D, F, & G)*    
&nbsp; │&emsp;&emsp;&emsp;  │&emsp;&emsp;&emsp;&emsp;*Also shows calculations for Max ΔL/L and Max L/F*    
&nbsp; │&emsp;&emsp;&emsp;└─  **lum_timecourses_figure.ipynb**  
&nbsp; │&emsp;&emsp;&emsp;   &emsp;&emsp;&emsp;&emsp;*No new calculations here — just plots bessel-filtered timecourses in one output figure*   

 ### &nbsp;│⎯   **/substrate_dose_response**    
&nbsp; │&emsp;&emsp;&emsp;└─  **dose_response_curve.ipynb**  
&nbsp; │&emsp;&emsp;&emsp;   &emsp;&emsp;&emsp;&emsp;*All-in-one dose response analysis*    
&nbsp; │&emsp;&emsp;&emsp;   &emsp;&emsp;&emsp;&emsp;*Calculates AUC from input `.csv` files (manually combined from microplate reader outputs)*   
&nbsp; │&emsp;&emsp;&emsp;   &emsp;&emsp;&emsp;&emsp;*Sets model, plots curve, and prints fitted parameters*      
&nbsp; │⎯  **requirements.txt**  
&nbsp; │&emsp;&emsp;&emsp; Basic with Anaconda package in ~2024, see for version numbers    
   └─ **README.md**


