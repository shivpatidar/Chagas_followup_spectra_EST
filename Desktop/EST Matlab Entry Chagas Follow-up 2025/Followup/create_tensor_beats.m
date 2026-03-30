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