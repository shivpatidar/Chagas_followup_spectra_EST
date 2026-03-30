function [binary_output,probability_output]=team_run_model(data_record, model_struct,verbose)
%[binary_output,probability_output]=team_run_model(data_record, model_struct, verbose)
% team_run_model - Runs the classification model on a single record.
% The input 'model_struct' is the loaded model structure from load_model.m

if verbose>=1
    disp(['Processing record: ' data_record])
end

% Default outputs (in case of failure)
binary_output='False';
probability_output=0.0;
% run
try
    % === UNPACK MODEL STRUCTURE ===
    model = model_struct.classification_model;
    % fisher_indices = model_struct.top_indices_1000;   % Adjust if named differently
    scaler_final_mean = model_struct.Meann;
    scaler_final_std = model_struct.Stdd;

catch ME
    warning('Failed while unpacking model_struct: %s', ME.message);
    return;
end

try
    % === FEATURE EXTRACTION ===
    header = fileread(data_record);
    original_path = pwd;
    current_folder = fileparts(data_record);
    if ~isempty(current_folder) && ~strcmp(pwd, current_folder)
        cd(current_folder);
    end

    [projected_features, metadata_features] = get_features(data_record, header);

    if ~strcmp(pwd, original_path)
        cd(original_path);
    end

    % --- Debug: Print extracted feature shapes ---
    %fprintf("  [DEBUG] Projected features size: [%d x %d]\n", size(projected_features,1), size(projected_features,2));
    %fprintf("  [DEBUG] Metadata features size:  [%d x %d]\n", size(metadata_features,1), size(metadata_features,2));

    % Skip if no features returned
    if isempty(projected_features) || isempty(metadata_features)
        warning('No features returned (too few R-peaks). Defaulting to negative prediction.');
        return;
    end

    % Ensure correct shapes (1xN vector)
    projected_features=sign(projected_features) .* log1p(abs(projected_features));
    metadata_features=sign(metadata_features) .* log1p(abs(metadata_features));
    projected_features = reshape(projected_features, 1, []);
    metadata_features = reshape(metadata_features, 1, []);

    % % Check projection dimension
    % expected_proj_dim = 1000;  %%UPDATE THISSS ACCORDINGLY!
    % if size(projected_features, 2) ~= expected_proj_dim
    %     warning('Dimension mismatch: Expected %d projected features, got %d. Skipping record (update this in team_run_model).', ...
    %         expected_proj_dim, size(projected_features, 2));
    %     return;
    % end

    % === PREPROCESSING PIPELINE ===
    
    X_full = [projected_features, metadata_features];
    % C. Apply Fisher-selected indices
    feat_indices= load('feat_indices.mat');
    X_full = X_full(:, feat_indices.top_indices_1000);
 
    % E. Final scaling (Stage 2 Scaling)
    features_final = (X_full - scaler_final_mean) ./ scaler_final_std;
    %fprintf("  [DEBUG] Final feature matrix size: [%d x %d]\n", size(features_final,1), size(features_final,2));

    

    % === MODEL PREDICTION ===
   
[predicted_class,probabilities]=model.predict(features_final);

if predicted_class==0
    binary_output='False';
else
    binary_output='True';
end
probabilities=probabilities./sum(probabilities);
probability_output=probabilities(2);

catch ME
    warning('Error during main processing: %s', ME.message);
    return;
end

if verbose>=1
    fprintf('Prediction complete: %s (Prob: %.4f)\n', binary_output, probability_output);
end
end
