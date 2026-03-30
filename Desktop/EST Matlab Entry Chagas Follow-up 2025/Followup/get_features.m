function [projected_features, metadata_features] = get_features(file, header)
% get_features - Extracts both projected tensor features and metadata features
%
% This is the full MATLAB implementation translated from the Python 'team_code.py'.
% It replaces all dummy data with real signal processing, CWT, and statistical
% feature extraction.
%
% Dependencies:
%   
%   - Tensor Toolbox ('tensor', 'ttm')
%   - Wavelet Toolbox ('cwt')
%   - Signal Processing Toolbox ('fir1', 'filtfilt', 'findpeaks')
%   - Statistics and Machine Learning Toolbox ('skewness', 'kurtosis')

    % --- Define Constants from Python Script ---
    selected_leads_idx = [2, 7, 9]; % MATLAB 1-idx for Python [2, 7, 9] (Leads III, aVL, V2)
    groups = { [1 2 3], [4 5 6], [7 8 9 10 11 12] };
    MAX_BEATS = 18;
    window_sec = 3;
    scales = 1:64;
    wavelet = 'coif4'; % coif4 wavelet
    
    projected_features = [];
    metadata_features = [];

    %Extracting sampling freq
    header_tmp1=strsplit(header,' ');
    fs=str2num(header_tmp1{3});
    %Initialise variables
  
    statsfeat=zeros(1,326);
    statsHR=zeros(1,30);
    
    %FB=zeros(1,40);
    signals=read_challenge_signals(file,header);
    signals_orig=signals;
    targetFs = 400;

        if fs ~= targetFs
            [p, q] = rat(targetFs / fs);  % Get rational approximation for resampling
            allecg_400Hz = resample(signals, p, q);  % Resample to 400 Hz
        else
            allecg_400Hz = signals;  % No resampling needed
        end
    % Compute resampling factors
    fs_orig=fs;
    fs=targetFs;
    signals=bandpassecg(allecg_400Hz',targetFs);
    % try
     %signals=scale_signals(signals,header);
    % catch
     %signals=normalize(signals);
    % end

    lead2ecg=signals_orig(:,2);
    
    try
        [qrs,sign,en_thres] = qrs_detect2(lead2ecg',0.25,0.6,fs_orig);%400Hz is sampling freq
        if length(qrs)<6
            [qrs,sign,en_thres] = qrs_detect2(signals_orig(:,8)',0.25,0.6,fs_orig);
        else
        end
        
    catch
        [qrs,sign,en_thres] = qrs_detect2(signals_orig(:,8)',0.25,0.6,fs_orig);%400Hz is sampling freq
        
    end
    qrs_new=floor(qrs.*(targetFs/fs_orig));
    
    allecg=signals;
    allecg_beats=signals;
        %% ======================================================
    %  REPLACE RAW ECG WITH AVERAGED R-CENTERED WINDOWS
    % ======================================================
    
    win_sec = 4;   % or 4 — match what you validated in testbench
    half_win = win_sec/2;
    half_win_samp = round(half_win * fs);
    
    rpeaks = qrs_new(:);
    N = size(allecg,1);
    
    windows = {};
    i = 1;
    prev_end = -inf;
    
    while i <= length(rpeaks)
    
        rp = rpeaks(i);
    
        start_idx = rp - half_win_samp;
        end_idx   = rp + half_win_samp;
    
        % skip overlapping windows
        if start_idx <= prev_end
            i = i + 1;
            continue;
        end
    
        % bounds
        if start_idx < 1 || end_idx > N
            i = i + 1;
            continue;
        end
    
        windows{end+1} = allecg(start_idx:end_idx, :);
    
        prev_end = end_idx;
    
        % next window must not overlap
        i = find(rpeaks > prev_end + half_win_samp, 1, 'first');
        if isempty(i)
            break;
        end
    end
    
    if isempty(windows)
        warning('No valid windows extracted — keeping original ECG.');
    else
        W = cat(3, windows{:});
        avg_window = mean(W, 3);
    
        % 🔥 overwrite allecg with averaged template
        allecg = avg_window;
    end

    
    % --- 4. Tensor Creation (Real CWT) ---
    
    leads_for_tensor = allecg(:, selected_leads_idx);
    T_tensor_data = create_tensor(leads_for_tensor, fs, qrs_new, window_sec, scales, wavelet, MAX_BEATS);
    
    if isempty(T_tensor_data)
    
        warning('Tensor empty. Checking for zero leads and attempting replacement: %s', file);
    
        selected_leads_idx_fixed = replace_bad_leads_basic( ...
            selected_leads_idx, allecg, signals_orig);
    
        leads_for_tensor = allecg(:, selected_leads_idx_fixed);
    
        T_tensor_data = create_tensor(leads_for_tensor, fs, qrs_new, window_sec, scales, wavelet, MAX_BEATS);
    
        if isempty(T_tensor_data)
            warning('Tensor creation failed even after lead replacement: %s', file);
            return;
        end
    end

    
    T = tensor(T_tensor_data); % Convert to Tensor Toolbox object
    
     % --- 4. Tensor Creation (Real CWT) ---
    selected_leads_idx = [2, 7, 9]; % MATLAB 1-idx for Python [2, 7, 9] (Leads III, aVL, V2)
    MAX_BEATS = 18;
    window_sec = 3;
    scales = 1:64;
    wavelet = 'mexh';
    leads_for_tensor_beats = allecg_beats(:, selected_leads_idx);
    T_tensor_data_beats = create_tensor_beats(leads_for_tensor_beats, fs, qrs_new, window_sec, scales, wavelet, MAX_BEATS);

    if isempty(T_tensor_data_beats)
    
        warning('Tensor empty. Checking for zero leads and attempting replacement: %s', file);
    
        selected_leads_idx_fixed = replace_bad_leads_basic( ...
            selected_leads_idx, allecg_beats, signals_orig);
    
        leads_for_tensor = allecg_beats(:, selected_leads_idx_fixed);
    
        T_tensor_data_beats = create_tensor(leads_for_tensor_beats, fs, qrs_new, window_sec, scales, wavelet, MAX_BEATS);
    
        if isempty(T_tensor_data)
            warning('Tensor creation failed even after lead replacement: %s', file);
            return;
        end
    end
    
    T_beats = tensor(T_tensor_data_beats); % Convert to Tensor Toolbox object

    % --- 5. Tensor Projection (Requires Tensor Toolbox) ---
    U0_data = load('factor_U1_cov.mat');
    U1_data = load('factor_U2_cov.mat');
    U2_data = load('factor_U3_cov.mat');
    
    U0 = U0_data.U0;
    U1 = U1_data.U1;
    U2 = U2_data.U2;

    % Check dimensions explicitly before ttm
    if size(T_beats, 1) ~= size(U0, 1) || size(T_beats, 2) ~= size(U1, 1) || size(T_beats, 3) ~= size(U2, 1)
        error('Tensor/Factor dimension mismatch. T_size=[%d,%d,%d], U0=%d, U1=%d, U2=%d', ...
            size(T_beats,1), size(T_beats,2), size(T_beats,3), size(U0,1), size(U1,1), size(U2,1));
    end
      
    core = ttm(T_beats, U0', 1); % Mode 1 projection
    core = ttm(core, U1', 2); % Mode 2 projection
    core = ttm(core, U2', 3); % Mode 3 projection

    
    % --- Flatten individually ---
    core_flat = core.data(:);   % [numel(core) x 1]
    T_flat    = T.data(:);      % [numel(T)    x 1]
    
    % --- Concatenate feature vectors ---
    combo_flat = [core_flat; T_flat]';  % row vector


    

    
    projected_features = combo_flat;


    % --- 6. Metadata and Statistical Features ---
    age = get_age(header);
    sex = get_sex(header);
    weight = get_weight(header);
    
    % Extract stats for ALL 12 leads
    stats_12_lead = extract_all_stats(allecg);
    
    % Compute HRV features from combined R-peaks
    [hrv_rr_stats, hrv_hr_stats] = compute_hrv_features(qrs_new, fs);
    
    % Concatenate all metadata (must match Python order)
    qrs_sign_feat   = mean(sign);
    qrs_thresh_feat = mean(en_thres);

    metadata_features = [ ...
        age, sex, weight, ...
        qrs_sign_feat, qrs_thresh_feat, ...
        stats_12_lead, ...
        hrv_hr_stats', hrv_rr_stats' ...
    ];

    
%% --- NESTED HELPER FUNCTIONS --- %%

    % --- QRS Detection (Pan-Tompkins) ---
    function r_peaks = pan_tompkins_qrs_matlab(sig, fs)
        % This is the updated R-peak detector with fixes.
        
        if ~exist('fir1', 'file') || ~exist('filtfilt', 'file') || ~exist('findpeaks', 'file')
            error('Signal Processing Toolbox is required for "pan_tompkins_qrs_matlab".');
        end
        
        % 1. Bandpass Filter (5-15 Hz)
        nyquist = fs / 2;
        % FIX 1: Python firwin(150) = 150 taps. MATLAB fir1(N) = N+1 taps.
        % So, N=149 gives 150 taps.
        b = fir1(149, [5 15] / nyquist, 'bandpass'); 
        filtered = filtfilt(b, 1, sig);
        
        % 2. Differentiate
        % FIX 2: Pad diff with a 0 to maintain signal length and alignment
        diff_sig = [0; diff(filtered)];
        
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

    function r_peaks_all_leads = get_all_r_peaks(sig3, fs)
        num_leads = size(sig3, 2);
        r_peaks_all_leads = cell(1, num_leads);
        for i = 1:num_leads
            try
                r_peaks_all_leads{i} = pan_tompkins_qrs_matlab(sig3(:, i), fs);
            catch ME
                % fprintf('QRS detection failed for lead %d: %s\n', i, ME.message);
                r_peaks_all_leads{i} = []; % Failed for this lead
            end
        end
    end

    function r_peaks_combined = combine_r_peaks_median(r_peaks_leads)
        trim = 1;
        trimmed_cells = {};
        for i = 1:length(r_peaks_leads)
            r = r_peaks_leads{i};
            if length(r) > trim
                trimmed_cells{end+1} = r(trim+1:end);
            end
        end
        
        if isempty(trimmed_cells)
            r_peaks_combined = [];
            return;
        end
        
        min_len = min(cellfun('length', trimmed_cells));
        aligned = zeros(length(trimmed_cells), min_len);
        for i = 1:length(trimmed_cells)
            aligned(i, :) = trimmed_cells{i}(1:min_len);
        end
        
        r_peaks_combined = round(median(aligned, 1));
    end

    % --- Tensor Creation (CWT) ---
    function T_tensor_data = create_tensor(leads, fs, ~, ~, scales, wavelet, ~)
    % CREATE_TENSOR
    % Uses averaged ECG window directly.
    % No beat extraction, padding, or MAX_BEATS logic.
    
        % leads: [samples x numLeads]
    
        numLeads = size(leads, 2);
        numFeat  = length(scales) * 2;
    
        % allocate single "slice"
        feats_block = zeros(numFeat, numLeads);
    
        for l = 1:numLeads
    
            seg = leads(:, l);
    
            if std(seg) < 1e-5
                warning('Lead %d nearly flat — zeroing its features.', l);
                feats_block(:, l) = zeros(numFeat,1);
                continue;
            end
    
    
            feats_block(:, l) = extract_scalogram_matlab( ...
                                    seg, fs, scales, wavelet);
        end
            if all(std(leads,0,1) < 1e-5)
            T_tensor_data = [];
            return;
        end
    
    
        % keep 3-D shape for tensor toolbox compatibility
        T_tensor_data = reshape(feats_block, ...
                                size(feats_block,1), ...
                                size(feats_block,2), ...
                                1);
    end
    
      function T_tensor_data_beats = create_tensor_beats(leads, fs, r_peaks, window_sec, scales, wavelet, MAX_BEATS)
        win_len = round(fs * window_sec);
        beat_tensors_cell = {};
        
        % Pad leads in case a beat segment goes out of bounds
        leads_padded = [leads; zeros(win_len, size(leads, 2))];
        
        for r_idx = 1:length(r_peaks)
            r = r_peaks(r_idx);
            
            % Get segment for all 3 leads
            seg_block = leads_padded(r : (r + win_len - 1), :);
            
            % Check for flatline in any lead
            if any(std(seg_block, 1) < 1e-5)
                continue; % Skip this beat
            end

            feats_for_beat = zeros(length(scales)*2, size(leads, 2)); % Pre-allocate
            
            for l = 1:size(leads, 2)
                seg = seg_block(:, l);
                feats_for_beat(:, l) = extract_scalogram_matlab(seg, fs, scales, wavelet);
            end
            
            beat_tensors_cell{end+1} = feats_for_beat;
        end
        
        if isempty(beat_tensors_cell)
            T_tensor_data = [];
            return;
        end

        % Pad/Truncate beats to MAX_BEATS
        nb = length(beat_tensors_cell);
        if nb < MAX_BEATS
            reps = floor(MAX_BEATS / nb);
            rem = mod(MAX_BEATS, nb);
            final_beats = [repmat(beat_tensors_cell, 1, reps), beat_tensors_cell(1:rem)];
        else
            final_beats = beat_tensors_cell(1:MAX_BEATS);
        end
        
        T_tensor_data_beats = cat(3, final_beats{:});
    end

    function features = extract_scalogram_matlab(sig, fs, scales, wavelet)
        if ~exist('cwt', 'file')
            error('Wavelet Toolbox is required for "extract_scalogram_matlab".');
        end
        
        try
            coef = cwt(sig, scales, wavelet);
        catch ME
            warning('cwt function failed. Trying alternative syntax. Error: %s', ME.message);
            coef = cwts(sig, 'scales', scales, 'wavelet', wavelet);
        end

        mag = abs(coef);
        mean_val = mean(mag, 2); % Mean across time (axis=1 in Python)
        mad_val = mean(abs(mag - mean_val), 2); % MAD across time
        
        features = [mean_val; mad_val]; % Return column vector
    end

    % --- Statistics Extraction ---
    function stats_vec = extract_stats_matlab(signal)
        if ~exist('skewness', 'file') || ~exist('kurtosis', 'file')
            warning('Statistics and Machine Learning Toolbox is required for "extract_stats_matlab". Returning NaNs.');
            stats_vec = NaN(30, 1);
            return;
        end

        x = signal(:); % Flatten
        n = length(x);
        if n == 0
            stats_vec = NaN(30, 1);
            return;
        end
        
        dx = diff(x);
        if isempty(dx), dx = 0; end % Handle single point input
        ddx = diff(dx);
        if isempty(ddx), ddx = 0; end % Handle two point input

        % Hjorth Parameters
        activity = var(x, 1); % flag=1 for N, not N-1
        mobility = std(dx, 1) / (std(x, 1) + 1e-8);
        mobility_dx = std(ddx, 1) / (std(dx, 1) + 1e-8);
        complexity = mobility_dx / (mobility + 1e-8);
        
        % Standard Stats
        mean_val = mean(x);
        std_val = std(x, 1);
        se_mean = std_val / sqrt(n);
        var_val = var(x, 1);
        cv = std_val / (mean_val + 1e-8);
        md = mean(abs(x - mean_val));
        
        % Percentiles
        p10 = prctile(x, 10);
        p90 = prctile(x, 90);
        per_range = p90 - p10;
        q1 = prctile(x, 25);
        median_val = prctile(x, 50);
        q3 = prctile(x, 75);
        iqr_val = q3 - q1;
        
        % Distribution
        skewness_val = skewness(x);
        vq = (q3 - q1) / (q3 + q1 + 1e-8);
        rms_val = sqrt(mean(x.^2));
        sum_val = sum(x);
        min_val = min(x);
        max_val = max(x);
        range_val = max_val - min_val;
        sumsq_val = sum(x.^2);
        kurtosis_val = kurtosis(x);
        mad_val = mad(x, 1); % Median absolute deviation from median
        q75 = q3;
        sq_dev = (x - mean_val).^2;
        mssd = mean(sq_dev);
        mode_val = mode(x);
        max_min_abs = abs(max_val - min_val);
        
        % Entropy
        p = abs(x) / (sum(abs(x)) + 1e-12);
        shannon_ent = -sum(p .* log2(p + 1e-12));
        log_energy_ent = sum(log(x.^2 + 1e-8)) * mean(abs(x));

        stats_vec = [
            mean_val; std_val; se_mean; var_val; cv; md; per_range;
            q1; median_val; iqr_val; q3; skewness_val; vq; rms_val;
            sum_val; min_val; max_val; range_val; sumsq_val; kurtosis_val;
            mad_val; q75; mssd; mode_val; max_min_abs;
            shannon_ent; log_energy_ent;
            activity; mobility; complexity
        ];
    end

    function all_stats = extract_all_stats(signals)
        num_leads = size(signals, 2);
        all_stats_cell = cell(1, num_leads);
        for i = 1:num_leads
            all_stats_cell{i} = extract_stats_matlab(signals(:, i));
        end
        % Concatenate all column vectors horizontally, then transpose to a row vector
        all_stats = cat(1, all_stats_cell{:})'; 
    end

    function [hrv_rr_stats, hrv_hr_stats] = compute_hrv_features(r_peaks, fs)
        rr = diff(r_peaks) / fs;
        if length(rr) < 2
            hrv_rr_stats = NaN(30, 1);
            hrv_hr_stats = NaN(30, 1);
            return;
        end
        
        hr = 60 ./ rr;
        
        hrv_rr_stats = extract_stats_matlab(rr);
        hrv_hr_stats = extract_stats_matlab(hr);
    end

    % --- Base MATLAB / Metadata Helpers ---
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
    
    function age=get_age(header)
        header=strsplit(header,'\n');
        age_tmp=header(startsWith(header,'# Age:'));
        if isempty(age_tmp)
            age=NaN;
        else
            age_tmp=strsplit(age_tmp{1},':');
            age=str2double(age_tmp{2});
        end
    end
    
    function sex=get_sex(header)
        header=strsplit(header,'\n');
        sex_tmp=header(startsWith(header,'# Sex:'));
        if isempty(sex_tmp)
             sex=NaN;
        else
            sex_tmp=strsplit(sex_tmp{1},':');
            sex_str=strtrim(sex_tmp{2});
            if startsWith(sex_str,'Fem')
                sex=0;
            elseif startsWith(sex_str,'Mal')
                sex=1;
            else
                sex=2; % Unknown/Other
            end
        end
    end

    function weight=get_weight(header)
        header=strsplit(header,'\n');
        weight_tmp=header(startsWith(header,'# Weight:'));
        if isempty(weight_tmp)
            weight=NaN;
        else
            weight_tmp=strsplit(weight_tmp{1},':');
            weight=str2double(weight_tmp{2});
        end
    end

end
function fixed_idx = replace_bad_leads_basic(selected_idx, allecg, signals_orig)

groups = { [1 2 3], [4 5 6], [7 8 9 10 11 12] };
fixed_idx = selected_idx;

num_leads_available = size(signals_orig, 2);

for k = 1:length(selected_idx)

    idx = selected_idx(k);
    is_bad = false;

    % Index out of bounds
    if idx > size(allecg, 2)
        is_bad = true;
    else
        sig = allecg(:, idx);

        % Basic validity check
        if isempty(sig) || all(sig == 0) || all(isnan(sig))
            is_bad = true;
        end
    end

    if ~is_bad
        continue;
    end

    % -------- Find group ----------
    found_group = [];
    for g = 1:length(groups)
        if any(groups{g} == idx)
            found_group = groups{g};
            break;
        end
    end

    if isempty(found_group)
        warning('Lead %d not found in any group. Skipping replacement.', idx);
        continue;
    end

    % -------- Find replacement ----------
    candidates = found_group(found_group <= num_leads_available);
    replacement = [];

    for c = candidates
        if c > size(allecg, 2)
            continue;
        end

        sig_c = allecg(:, c);

        if isempty(sig_c) || all(sig_c == 0) || all(isnan(sig_c))
            continue;
        end

        replacement = c;
        break;
    end

    if isempty(replacement)
        warning('No valid replacement found for bad lead %d.', idx);
    else
        warning('Lead %d is invalid (all zero). Replacing with lead %d.', idx, replacement);
        fixed_idx(k) = replacement;
    end

end
end



% % --- Plot ECG and detected peaks ---
% figure;
% t = (0:4000-1)/400;
% plot(t, lead2ecg, 'b', 'LineWidth', 1.2); hold on;
% plot(t(qrs_new), lead2ecg(qrs_new), 'ro', 'MarkerSize', 6, 'LineWidth', 1.5);
% xlabel('Time (s)');
% ylabel('Amplitude (mV)');
% title('ECG Signal with Detected QRS Peaks');
% legend('ECG', 'Detected QRS Peaks');
% grid on;