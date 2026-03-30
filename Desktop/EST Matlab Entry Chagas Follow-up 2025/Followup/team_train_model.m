function team_train_model(input_directory, output_directory, verbose)
% team_train_model - Parallel version (feature extraction parallelized)
% Logic, structure, and saving behavior are IDENTICAL to serial version.
% Only change: uses parfor for feature extraction.
% 1. TENSOR TOOLBOX Auto-Setup (Path Only, No Unzip)
% ==================================================
try
    exist_tt = exist('tensor', 'file') || exist('@tensor', 'dir');
catch
    exist_tt = false;
end

if ~exist_tt
    fprintf('Tensor Toolbox not found on path. Attempting to add local folder...\n');

    % Determine current script folder (works in deployed mode also)
    this_file   = mfilename('fullpath'); 
    this_folder = fileparts(this_file);

    % Folder where user has placed the extracted tensor toolbox
    tensor_folder = fullfile(this_folder, 'tensor_toolbox-v3.2');

    if exist(tensor_folder, 'dir')
        addpath(genpath(tensor_folder));
        fprintf('Tensor Toolbox added to MATLAB path from: %s\n', tensor_folder);
    else
        fprintf(['WARNING: Tensor Toolbox folder NOT found.\n' ...
                 'Expected at: %s\n' ...
                 'Continuing assuming platform already provides Tensor Toolbox.\n'], tensor_folder);
    end

else
    fprintf('Tensor Toolbox already found on MATLAB path.\n');
end



if nargin < 4
    CPU_usage = 0.65; % default: use 100% CPU
end

if verbose >= 1
    disp('Finding Challenge data...')
end

% Find all header files recursively
records = dir(fullfile(input_directory, '**', '*.hea'));
num_records =length(records);
if num_records < 1
    error('No records were provided')
end

if verbose >= 1
    disp('Extracting raw tensor and metadata features (parallel mode)...')
end

if ~isdir(output_directory)
    mkdir(output_directory)
end

% --- 1. INITIALIZE RAW FEATURE MATRICES ---
X_full = [];
X_metadata_raw = [];
labels = [];

original_path = pwd;
if ~isdeployed, addpath(genpath(original_path)); end

% --- Parallel setup ---
if isempty(gcp('nocreate'))
    % Determine the number of workers to request
    total_cores = feature('numcores');
    
    % Find the maximum allowed workers for the 'local' cluster
    cluster = parcluster('local');
    max_workers_allowed = cluster.NumWorkers;
    
    % Calculate desired workers based on CPU usage
    desired_workers = max(1, floor(total_cores * CPU_usage));
    
    % Use the minimum of desired workers and the allowed maximum
    num_workers = min(desired_workers, max_workers_allowed);
    
    parpool('local', num_workers);
else
    pool = gcp('nocreate');
    % If a pool exists, adjust it to respect limits
    cluster = parcluster('local');
    max_workers_allowed = cluster.NumWorkers;
    total_cores = feature('numcores');
    desired_workers = max(1, floor(total_cores * CPU_usage));
    num_workers = min(desired_workers, max_workers_allowed);

    if pool.NumWorkers ~= num_workers
        delete(pool);
        parpool('local', num_workers);
    else
        % Pool is already running with optimal workers
        num_workers = pool.NumWorkers;
    end
end

fprintf('Using %d workers (limited by cluster setting of %d) for feature extraction.\n', num_workers, cluster.NumWorkers);

% --- Preallocate cell arrays for parfor aggregation ---
feat_cell= cell(num_records,1);
proj_cell = cell(num_records, 1);
meta_cell = cell(num_records, 1);
label_cell = cell(num_records, 1);
skip_count = 0;

progress = 0;
pct_step = max(1, floor(num_records / 20)); % 5% progress steps

parfor j = 1:num_records

    rec_name = records(j).name;
    rec_folder = records(j).folder;
    try
        header = fileread(fullfile(rec_folder, rec_name));
        cd(rec_folder);
        [projected_features, metadata_features] = get_features(rec_name, header);
        if isempty(projected_features) || isempty(metadata_features)
            fprintf('Warning: [SKIPPING RECORD] No features returned for %s.\n', rec_name);
            proj_cell{j} = [];
            meta_cell{j} = [];
            label_cell{j} = [];
            feat_cell{j}=[];
            continue;
        end
        feat_indices= load('feat_indices.mat');
        feat1= [projected_features, metadata_features];
        feat2=feat1(:,feat_indices.top_indices_1000);
        feat_cell{j} = sign(feat2) .* log1p(abs(feat2));
        % proj_cell{j} = sign(projected_features) .* log1p(abs(projected_features));
        % meta_cell{j} = sign(metadata_features) .* log1p(abs(metadata_features));

        label_cell{j} = get_labels(header);
    catch ME
        fprintf('Warning: [FATAL EXTRACTION ERROR] for %s: %s\n', rec_name, ME.message);
        proj_cell{j} = [];
        meta_cell{j} = [];
        label_cell{j} = [];
        feat_cell{j}=[];
    end

    if mod(j, pct_step) == 0
        fprintf('Progress: %d/%d (%.0f%%)\n', j, num_records, 100*j/num_records);
    end
end

cd(original_path)
rmpath(genpath(original_path))



valid_idx = ~cellfun(@isempty, feat_cell);
X_full = cell2mat(feat_cell(valid_idx));
labels = cell2mat(label_cell(valid_idx));

% --- Add check for no successful records ---
if isempty(X_full)
    error('Feature extraction failed for ALL records. No data to train on.');
end

% --- Sanity check ---
if size(X_full, 1) ~= length(labels)
    error('Internal Data Mismatch: features (%d) ≠ labels (%d).', ...
        size(X_full, 1), length(labels));
end

if verbose >= 1
    fprintf('Raw features collected from %d records.\n', size(X_full, 1));
    fprintf('  - X_full: [%d x %d]\n', size(X_full));
    % fprintf('  - X_metadata_raw:  [%d x %d]\n', size(X_metadata_raw));
    fprintf('  - Labels:          [%d x %d]\n', size(labels));
    disp('--------------------------------------------------')
end

% % --- 2. PREPROCESSING & FEATURE SELECTION ---
% A. Mean imputation for metadata
feat_mean = zeros(1, size(X_full, 2));
for k = 1:size(X_full, 2)
    col =X_full(:, k);
    non_nan_col = col(~isnan(col));
    if isempty(non_nan_col)
        feat_mean(k) = 0;
    else
        feat_mean(k) = mean(non_nan_col);
    end
end
for k = 1:size(X_full, 2)
    nan_indices = isnan(X_full(:, k));
    X_full(nan_indices, k) = feat_mean(k);
end

if verbose >= 1
    fprintf('Metadata imputation complete. Metadata shape: [%d x %d]\n', size(X_full));
end

% %C. Compute Fisher Scores and Select Top 1000
% if exist('fscmrmr', 'file')
%     if verbose >= 1
%         disp('Calculating Fisher Scores using fscmrmr...')
%     end
%     [idx, ~] = fscmrmr(X_full1 , labels);
%     top_indices_1000 = idx(1:1000);
% else
%     if verbose >= 1
%         warning('fscmrmr not found. Using first 1000 indices.');
%     end
%     top_indices_1000 = 1:min(1000, size(X_full1 , 2));
% end
%top_indices_1000 = 1:min(1000, size(X_full1 , 2));

%X_full = X_full1(:,top_indices_1000);





% clear X_selected;    % --- Memory saving: Clear selected features
% clear X_metadata_raw; % --- Memory saving: Clear raw metadata

% % E. Final scaling
 Meann = mean(X_full, 1);
 Stdd = std(X_full, 0, 1) + 1e-8;
  features = (X_full - Meann )./ Stdd;
  features(isnan(features))=0;
 %clear X_full; % --- Memory saving: Clear unscaled full data
 classes=sort(unique(labels));

if verbose>=1
    fprintf('Training the model on the data... \n')
end

t = ClassificationTree.template('MaxNumSplits',10); 
 model = fitensemble(features,labels,'RusBoost',300,t,'type','classification');
 save_models(output_directory,model,classes,Meann,Stdd)

if verbose>=1
    fprintf('Done. \n')
end
end


function save_models(output_directory, classification_model, classes,Meann,Stdd)

filename = fullfile(output_directory,'classification_model.mat');
save(filename,'classification_model','classes','-v7.3','Meann','-v7.3','Stdd','-v7.3');

end

function label = get_labels(header)
    header_lines = strsplit(header, '\n');
    dx = header_lines(startsWith(header_lines, '# Chagas label:'));
    if ~isempty(dx)
        dx = strsplit(dx{1}, ':');
        dx = strtrim(dx{2});
        if startsWith(dx, 'Fa') || startsWith(dx, 'False')
            label = 0;
        else
            label = 1;
        end
    else
        error('# Labels missing!')
    end
end

function [X_synth, Y_synth] = smote_data(X_minority, Y_minority, N, K)
    num_minority = size(X_minority, 1);
    K = min(K, num_minority - 1);
    if K <= 0
        X_synth = [];
        Y_synth = [];
        return;
    end
    samples_per_point = floor(N / num_minority);
    remainder_samples = N - samples_per_point * num_minority;
    [~, IDX] = knnsearch(X_minority, X_minority, 'k', K + 1);
    IDX = IDX(:, 2:end);
    synthetic_features = zeros(N, size(X_minority, 2));
    synthetic_labels = repmat(Y_minority(1), N, 1);
    synth_count = 0;
    for i = 1:num_minority
        num_to_generate = samples_per_point + (i <= remainder_samples);
        current_point = X_minority(i, :);
        for g = 1:num_to_generate
            random_col = randi(K);
            raw_idx = IDX(i, random_col);
            if isfinite(raw_idx) && raw_idx >= 1 && raw_idx <= num_minority
                neighbor_idx = round(raw_idx);
            else
                neighbor_idx = randi(num_minority);
            end
            neighbor_point = X_minority(neighbor_idx, :);
            gamma = rand();
            synthetic_sample = current_point + (neighbor_point - current_point) * gamma;
            synth_count = synth_count + 1;
            synthetic_features(synth_count, :) = synthetic_sample;
        end
    end
    X_synth = synthetic_features(1:synth_count, :);
    Y_synth = synthetic_labels(1:synth_count);
end

% function save_models(output_directory, classification_model, classes, ...
%     top_indices_700, final_mean, final_std)
%     % This helper function saves the model and all required preprocessing assets,
%     % ensuring variable names match the team_run_model loading structure.
% 
%     filename = fullfile(output_directory, 'classification_model.mat');
% 
%     % Save all necessary assets using the required names:
%     % classification_model, classes, proj_mean, proj_std, 
%     % top_indices_700, final_mean, final_std
%     save(filename, 'classification_model', 'classes', ...
%         'top_indices_700', ...
%         'final_mean', 'final_std', '-v7.3');
% end
