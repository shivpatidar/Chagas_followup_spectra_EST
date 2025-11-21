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
