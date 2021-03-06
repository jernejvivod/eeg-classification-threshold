
function feature_vec = engineer_features(signal, fs)
   % Extract features from signal. The parameter fs represents the sampling frequency.
    
   % Get features.
   feature_mu = mean(signal);
   feature_sig = std(signal);
   feature_var = var(signal);
   feature_med = median(signal);
   feature_max_t = max(signal);
   feature_min_t = min(signal);
   feature_range = feature_max_t - feature_min_t;
   feature_rms = rms(signal);
   [~, feature_argmin] = min(signal);
   [~, feature_argmax] = max(signal);
   feature_energy_t = sum(abs(signal).^2);
   feature_ent = entropy(signal/max(abs(signal)));
   feature_skw = skewness(signal);
   feature_krt = kurtosis(signal);
   feature_iqr = iqr(signal);
   feature_mad = mad(signal);
   [feature_wd, feature_lo, feature_hi, feature_power] = obw(signal, fs);
   signal_f = fft(signal)/length(signal);
   feature_bndpwr = bandpower(signal_f);
   feature_energy_f = sum(abs(signal_f));
   feature_mean_f = mean(abs(signal_f));
   feature_max_f = max(abs(signal_f));
   feature_min_f = min(abs(signal_f));
   feature_normc = fftshift(abs(signal_f)); 
   
   
   % Stack into feature vector
   feature_vec = [feature_mu, ...
     feature_sig, ...
     feature_var, ...
     feature_med, ... 
     feature_max_t, ...
     feature_min_t, ...
     feature_range, ...
     feature_rms, ...
     feature_argmin, ...
     feature_argmax, ...
     feature_wd, ...
     feature_lo, ...
     feature_hi, ...
     feature_power, ...
     feature_energy_t, ...
     feature_ent, ...
     feature_skw, ...
     feature_krt, ...
     feature_iqr, ...
     feature_mad, ...
     feature_bndpwr, ...
     feature_energy_f, ...
     feature_mean_f, ...
     feature_max_f, ...
     feature_min_f
     % feature_normc(1:10), ...
     ];
end

