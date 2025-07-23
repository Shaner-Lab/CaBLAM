% ---- CaBLAM in vivo ----

% running this script will generate example processed data, 
% as well as the figures and stats from the in vivo mouse data in:
% CaBLAM! A high-contrast bioluminescent Ca2+ indicator derived from an engineered Oplophorus gracilirostris luciferase
% Labert et al.
% https://www.biorxiv.org/content/10.1101/2023.06.25.546478v3

% Instructions:
% 1. Download 'demo_data' here: https://doi.org/10.26300/7sg5-w257
% 2. Place demo_data folder inside the cablam_in_vivo_analysis folder
% 3. Set 'pth' to where cablam_in_vivo_analysis in located on your computer
pth = 'C:\Users\Jerem\Desktop\'; % path to '\cablam_in_vivo_analysis'

% Development Environment:
%     MATLAB R2024b
%     Computer Type: PCWIN64
%     Operating System: Microsoft Windows 11 Home
%     Number of Processors: 20
%     CPU Information: 12th Gen Intel(R) Core(TM) i9-12900H 
%     GPU Information: NVIDIA GeForce RTX 3060 Laptop GPU
%     Physical Memory (RAM): 65201 MB (6.84e+10 bytes)

% Required MATLAB Toolboxes:
%     image_toolbox
%     statistics_toolbox

clc

%% add subfolders to path
addpath(genpath([pth 'cablam_in_vivo_analysis\scripts']));

%% ---- 1. infusion ffz data ----
% infusion data processing, example data
inf_process_all(pth)

% infusion data, generate figures and stats
inf_figures_and_stats(pth)

%% ---- 2. retro-orbital cfz data ----

% retro-orbital data processing, example data
ro_process_all(pth)

% retro-orbital data, generate figures and stats
ro_figures_and_stats(pth)

%% ---- 3. long duration ffz imaging session ----
long_duration_all(pth)


%% clean-up

% remove subfolders from path
rmpath(genpath([pth 'cablam_in_vivo_analysis\scripts']));