This repository includes two toolboxes packaged directly with the code:

- `wfdb_toolbox/` — provided as standard MATLAB m-code (no installation required)
- `tensor_toolbox-v3.2/` — Tensor Toolbox 

Please add these toolboxes to the MATLAB path before running the scripts:

```matlab
% Get repository root (directory where this script is located)
repo_root = fileparts(mfilename('fullpath'));

% Paths to included toolboxes
wfdb_path   = fullfile(repo_root, 'mcode');          % WFDB m-code
tensor_path = fullfile(repo_root, 'tensor_toolbox-v3.2');   % Tensor Toolbox

% Add toolboxes (and their subfolders) to MATLAB path
addpath(genpath(wfdb_path));
addpath(genpath(tensor_path));


Additionally, following add-ons have to be done ( toolboxes by Mathworks):

Wavelet Toolbox  (by MathWorks)
Signal Processing Toolbox  (by MathWorks)
Statistics and Machine Learning Toolbox (by MathWorks)
Deep Learning Toolbox (by MathWorks)
Parallel Computing Toolbox (by MathWorks)

