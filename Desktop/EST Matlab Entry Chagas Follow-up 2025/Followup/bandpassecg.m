function filecg = bandpassecg(allecg,Fs)
%UNTITLED7 Summary of this function goes here
%   Detailed explanation goes here

% Define cutoff frequencies for bandpass (in Hz)
f_low = 0.5;   % Lower cutoff frequency
f_high = 150; % Upper cutoff frequency

% Normalize frequencies to Nyquist frequency (Fs/2)
Wn = [f_low f_high] / (Fs/2);

% Define filter order
n = 4; % Order of the filter

% Design Butterworth bandpass filter
[b, a] = butter(n, Wn, 'bandpass');
filecg=[];
% Apply the filter to the signal
for i=1:size(allecg,1)
filecg(:,i) = filtfilt(b, a, allecg(i,:));  % Zero-phase filtering
end
% Plot original and filtered signals
% figure;
% subplot(2,1,1);
% plot(t, x);
% title('Original Signal');
% xlabel('Time (s)');
% ylabel('Amplitude');
% 
% subplot(2,1,2);
% plot(t, y);
% title('Filtered Signal (Butterworth Bandpass)');
% xlabel('Time (s)');
% ylabel('Amplitude');


end

