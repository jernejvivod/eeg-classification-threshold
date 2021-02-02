import numpy as np
from feature_extraction.feature_extraction_utils.filt import get_filter


def extract_features(signals, features_to_extract, set_type, target=None):
    """
    Extract features from specified signals to obtain feature vector. The
    features to be extracted as well as additional data are specified by the features_to_extract
    dictionary.

    Args:
        signals (numpy.ndarray): signals (multichannel, same time values) from which to extract the features.
        features_to_extract (dict): dictionary in specified form specifying which features to extract as well
                                    as additional parameters for the feature extraction process.
        set_type (str): "train" or "test".
        target (numpy.ndarray): target values (applicable for training set).

    Returns:
        (numpy.ndarray): obtained feature vector.
    """
    
    # Initialize array for obtained features. 
    features = np.empty((signals.shape[0], 0), dtype=float)

    # Go over features to be extracted.

    ### CSP ###
    if "CSP" in features_to_extract:
        from mne.decoding import CSP
        filt_coeff = get_filter(features_to_extract["CSP"]["samp_freq"], 
                features_to_extract["CSP"]["f_pass"], 
                features_to_extract["CSP"]["f_stop"], 
                features_to_extract["CSP"]["taps"])
        
        # If extracting features from training set.
        if set_type == "train"  :
            
            # Initialize and fit CSP transformer.
            csp = CSP(transform_into="csp_space", n_components=features_to_extract["CSP"]["n_components"], reg=None, norm_trace=False)
            csp.fit(signals, target)
            features_to_extract["CSP"]["csp"] = csp

            # Transform signals and compute features.
            trans = np.array([[np.convolve(xj, filt_coeff, mode="valid") for xj in xi] for xi in csp.transform(signals)])
            features = np.hstack((features, np.array([np.log(np.var(x, axis=1)) for x in trans])))

        # If extracting features from testing set.
        elif set_type == "test":
            
            # Transform signals using CSP transformer and obtain features.
            csp = features_to_extract["CSP"]["csp"]
            trans = np.array([[np.convolve(xj, filt_coeff, mode="valid") for xj in xi] for xi in csp.transform(signals)])
            features = np.hstack((features, np.array([np.log(np.var(x, axis=1)) for x in trans])))
        else:
            raise(ValueError("Invalid set_type argument value"))

    if "spec" in features_to_extract:
        import matlab.engine
        eng = matlab.engine.start_matlab()
        
        # Get feature using MATLAB script.
        res_features = np.array([[np.ravel(np.array(eng.engineer_features(matlab.double(list(signal)), 160))) 
            for signal in interval] for interval in signals])
        features = np.hstack((features, res_features.reshape(res_features.shape[0], -1)))

    
    # Return extracted features and features_to_extract dictionary.
    return features, features_to_extract

