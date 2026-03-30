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
