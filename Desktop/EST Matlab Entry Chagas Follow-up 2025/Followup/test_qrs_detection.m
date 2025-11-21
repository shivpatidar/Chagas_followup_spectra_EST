function test_qrs_detection()
    % This is a standalone script to debug R-peak detection.
    
    fprintf('--- QRS Detection Debugger ---\n');
    
    % --- 1. Add Toolboxes to Path ---
    % Add paths only if they are not already on the path
    if ~contains(path, 'mcode')
        addpath('D:\Physionet Challenge 2025\Matlab_ex_phy\mcode');
    end
    if ~contains(path, 'tensor_toolbox-v3.2')
        addpath(genpath('D:\Physionet Challenge 2025\Matlab_ex_phy\tensor_toolbox-v3.2'));
    end
    
    % --- 2. Define File to Test ---
    % Using the file that failed first in your log
    data_folder = 'D:\Physionet Challenge 2025\Matlab_ex_phy\test_training_data';
    record_name = '107019'; % The file that failed
    
    fprintf('Testing with record: %s\n', record_name);
    
    % --- 3. Load Data ---
    try
        % Navigate to the data folder to load the file
        original_path = pwd;
        cd(data_folder);
        [signals, fs] = rdsamp(record_name);
        cd(original_path); % Go back to the original directory
    catch ME
        fprintf('Error loading record %s: %s\n', record_name, ME.message);
        return;
    end

    if isempty(signals)
        fprintf('Failed to load signal data for %s.\n', record_name);
        return;
    end
    
    % Select Lead II
    lead_II_raw = signals(:, 2);
    
    % --- 4. Normalize Signal ---
    mu = nanmean_custom(lead_II_raw, 1);
    sigma = nanstd_custom(lead_II_raw, 1) + 1e-8;
    lead_II_norm = (lead_II_raw - mu) ./ sigma;
    
    % --- 5. Run QRS Detection ---
    % We will run the function and get its internal 'mwa' signal for plotting
    fprintf('Running pan_tompkins_qrs_matlab...\n');
    [r_peaks, mwa_signal] = pan_tompkins_qrs_matlab(lead_II_norm, fs);
    
    % --- 6. Report Results ---
    if isempty(r_peaks)
        fprintf('!!! R-PEAK DETECTION FAILED !!!\n');
        fprintf('No R-peaks were found for this record.\n');
    else
        fprintf('SUCCESS: Found %d R-peaks.\n', length(r_peaks));
        disp('R-Peak Indices:');
        disp(r_peaks'); % Transpose for better column display
    end

    % --- 7. Plot for Visual Inspection ---
    fprintf('Generating debug plot...\n');
    
    figure; % Create a new figure window
    
    % Plot 1: The processed MWA signal and the detected peaks
    ax1 = subplot(2,1,1);
    plot(ax1, mwa_signal);
    hold(ax1, 'on');
    if ~isempty(r_peaks)
        % Plot the detected peaks on the MWA signal
        plot(ax1, r_peaks, mwa_signal(r_peaks), 'rv', 'MarkerFaceColor', 'r', 'DisplayName', 'Detected R-Peaks');
    end
    title(ax1, ['Processed MWA Signal & Detected Peaks (Record: ' record_name ')']);
    legend(ax1);
    hold(ax1, 'off');
    
    % Plot 2: The original (normalized) signal and the peak locations
    ax2 = subplot(2,1,2);
    plot(ax2, lead_II_norm);
    hold(ax2, 'on');
    if ~isempty(r_peaks)
        % Plot the same peak indices on the original signal
        plot(ax2, r_peaks, lead_II_norm(r_peaks), 'ro', 'DisplayName', 'R-Peak Locations');
    end
    title(ax2, 'Original Normalized Lead II Signal');
    xlabel(ax2, 'Sample Index');
    legend(ax2);
    hold(ax2, 'off');
    
    % Link the x-axes so zooming in one zooms the other
    linkaxes([ax1, ax2], 'x');

end

%% --- NESTED HELPER FUNCTIONS (Copied from get_features.m) --- %%

function [r_peaks, mwa] = pan_tompkins_qrs_matlab(sig, fs)
    % Modified to return the MWA signal for debugging
    
    if ~exist('fir1', 'file') || ~exist('filtfilt', 'file') || ~exist('findpeaks', 'file')
        error('Signal Processing Toolbox is required for "pan_tompkins_qrs_matlab".');
    end

    % 1. Bandpass Filter (5-15 Hz)
    nyquist = fs / 2;
    b = fir1(149, [5 15] / nyquist, 'bandpass'); % 150-tap filter
    filtered = filtfilt(b, 1, sig);

    % 2. Differentiate
    diff_sig = [0; diff(filtered)]; % Pad to maintain length

    % 3. Square
    squared = diff_sig.^2;

    % 4. Moving Window Average (150ms)
    mwa_len = round(0.15 * fs);
    mwa_b = ones(1, mwa_len) / mwa_len;
    mwa = conv(squared, mwa_b, 'same');

    % 5. Find Peaks
    if all(isnan(mwa)) || all(mwa == 0)
         r_peaks = [];
         return;
    end
    threshold = mean(mwa, 'omitnan') * 1.2;
    min_dist = round(fs / 2); % 0.5 sec distance
    
    [~, r_peaks] = findpeaks(mwa, 'MinPeakHeight', threshold, 'MinPeakDistance', min_dist);
end

function result = nanmean_custom(X, dim)
    if nargin < 2, dim = find(size(X) ~= 1, 1); end
    if isempty(dim), dim = 1; end
    X_sum = sum(X, dim, 'omitnan');
    X_count = sum(~isnan(X), dim);
    X_count(X_count == 0) = 1; 
    result = X_sum ./ X_count;
    result(X_count == 1 & isnan(X_sum)) = NaN; 
end

function result = nanstd_custom(X, flag, dim)
    if nargin < 3, dim = find(size(X) ~= 1, 1); end
    if nargin < 2, flag = 0; end
    if isempty(dim), dim = 1; end
    mu = nanmean_custom(X, dim);
    sq_diff = (X - mu).^2;
    X_sum_sq_diff = sum(sq_diff, dim, 'omitnan');
    n = sum(~isnan(X), dim);
    if flag == 0, n_denom = n - 1; else, n_denom = n; end
    n_denom(n_denom <= 0) = 1; 
    variance = X_sum_sq_diff ./ n_denom;
    result = sqrt(variance);
    result(n == 0) = NaN; 
end