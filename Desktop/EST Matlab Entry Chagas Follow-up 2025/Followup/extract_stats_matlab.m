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
