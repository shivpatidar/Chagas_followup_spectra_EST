function [binary_output,probability_output]=team_run_model(data_record, model_struct, verbose)
% team_run_model - Runs the classification model on a single record.
% The input 'model_struct' is the loaded model structure from load_model.m

if verbose>=1
    disp(['Processing record: ' data_record])
end

% Default outputs (in case of failure)
binary_output='False';
probability_output=0.0;

try
    % === UNPACK MODEL STRUCTURE ===
    model = model_struct.classification_model;
    scaler_proj_mean = model_struct.proj_mean;
    scaler_proj_std = model_struct.proj_std;
    fisher_indices = model_struct.top_indices_700;   % Adjust if named differently
    scaler_final_mean = model_struct.final_mean;
    scaler_final_std = model_struct.final_std;

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
    projected_features = reshape(projected_features, 1, []);
    metadata_features = reshape(metadata_features, 1, []);

    % Check projection dimension
    expected_proj_dim = length(scaler_proj_mean);
    if size(projected_features, 2) ~= expected_proj_dim
        warning('Dimension mismatch: Expected %d projected features, got %d. Skipping record.', ...
            expected_proj_dim, size(projected_features, 2));
        return;
    end

    % === PREPROCESSING PIPELINE ===
    
    % A. Replace NaNs/Infs in projected features (New addition for robustness)
    projected_features(~isfinite(projected_features)) = 0;

    % A. Replace NaNs/Infs in metadata
    metadata_features(~isfinite(metadata_features)) = 0;

    % B. Scale projected features (Stage 1 Scaling)
    X_std = (projected_features - scaler_proj_mean) ./ scaler_proj_std;
    %fprintf("  [DEBUG] After scaling projected features: [%d x %d]\n", size(X_std,1), size(X_std,2));

    % C. Apply Fisher-selected indices
    X_selected = X_std(:, fisher_indices);
    %fprintf("  [DEBUG] After Fisher selection: [%d x %d]\n", size(X_selected,1), size(X_selected,2));

    % D. Concatenate with metadata
    X_full = [X_selected, metadata_features];
    %fprintf("  [DEBUG] After concatenating metadata: [%d x %d]\n", size(X_full,1), size(X_full,2));

    % E. Final scaling (Stage 2 Scaling)
    features_final = (X_full - scaler_final_mean) ./ scaler_final_std;
    %fprintf("  [DEBUG] Final feature matrix size: [%d x %d]\n", size(features_final,1), size(features_final,2));

    % Check final input size before classify()
    expected_input_dim = model.Layers(1).InputSize(1);
    if size(features_final, 2) ~= expected_input_dim
        warning('Incorrect input size: expected %d, got %d. Defaulting to negative prediction.', ...
            expected_input_dim, size(features_final, 2));
        binary_output = 'False';
        probability_output = 0.0;
        return;
    end

    % === MODEL PREDICTION ===
    [predicted_labels, raw_probabilities] = classify(model, features_final);
    %fprintf("  [DEBUG] Model classification output size: [%d x %d]\n", size(raw_probabilities,1), size(raw_probabilities,2));

    % === THRESHOLD AND OUTPUT ===
    CUSTOM_THRESHOLD = 0.6;
    probability_output = raw_probabilities(1, 2);

    if probability_output >= CUSTOM_THRESHOLD
        binary_output='True';
    else
        binary_output='False';
    end

catch ME
    warning('Error during main processing: %s', ME.message);
    return;
end

if verbose>=1
    fprintf('Prediction complete: %s (Prob: %.4f)\n', binary_output, probability_output);
end
end
