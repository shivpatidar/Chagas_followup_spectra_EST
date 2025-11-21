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