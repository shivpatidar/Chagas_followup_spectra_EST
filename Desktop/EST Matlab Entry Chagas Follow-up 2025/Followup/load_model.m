function classification_model = load_model(model_directory, verbose)
% Load the models that were saved by team_train_model.

if verbose>=1
    disp('Loading model and preprocessing assets...')
end

filename = fullfile(model_directory,'classification_model.mat');

% Load all variables, including the classification model object, scalers, and indices
S = load(filename); 

% Return the entire structure (S) as the 'classification_model' argument.
% This structure contains S.classification_model, S.scaler_proj_mean, etc.
classification_model = S; 

if verbose>=1
    disp('Model and assets loaded successfully.')
end

end