function all_stats = extract_all_stats(signals)
        num_leads = size(signals, 2);
        all_stats_cell = cell(1, num_leads);
        for i = 1:num_leads
            all_stats_cell{i} = extract_stats_matlab(signals(:, i));
        end
        % Concatenate all column vectors horizontally, then transpose to a row vector
        all_stats = cat(1, all_stats_cell{:})'; 
    end