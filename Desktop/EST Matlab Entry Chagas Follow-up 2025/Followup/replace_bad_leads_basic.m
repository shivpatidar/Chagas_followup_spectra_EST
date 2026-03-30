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
