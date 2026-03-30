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