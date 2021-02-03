import numpy as np
import yaml
from feature_extraction.feature_extraction_utils.filt import get_filter
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


def parse_feature_extraction_config(file_path):
    """
    Parse feature extraction configuration dictionary from yaml file.

    Args:
        file_path (str): path to yaml file.

    Returns:
        (dict): dictionary in specified form specifying which features to extract as well
                as additional parameters for the feature extraction process.
    """
    with open(file_path, 'r') as f:
        return yaml.safe_load(f)


def extract_features(signals, sampling_freq, feature_extraction_config, target):
    """
    Extract features from specified signals to obtain feature vector. The
    features to be extracted as well as additional data are specified by the feature_extraction_config
    dictionary.

    Args:
        signals (numpy.ndarray): signals (multichannel, same time values) from which to extract the features.
        feature_extraction_config (dict): dictionary in specified form specifying which features to extract as well
                                    as additional parameters for the feature extraction process.
    Returns:
        (numpy.ndarray): obtained feature vector.
    """
    
    # Initialize array for obtained features. 
    features = np.empty((signals.shape[0], 0), dtype=float)

    # Go over features to be extracted.
    if feature_extraction_config['CSP']['use']:
        from mne.decoding import CSP

        # Get FIR filter coefficients.
        filt_coeff = get_filter(sampling_freq, 
                feature_extraction_config["CSP"]["f_pass"], 
                feature_extraction_config["CSP"]["f_stop"], 
                feature_extraction_config["CSP"]["taps"])
        
        # Initialize and fit CSP transformer.
        csp = CSP(transform_into="csp_space", n_components=feature_extraction_config["CSP"]["n_components"], reg=None, norm_trace=False)
        csp.fit(signals, target)

        # Transform signals and compute features.
        trans = np.array([[np.convolve(xj, filt_coeff, mode="valid") for xj in xi] for xi in csp.transform(signals)])
        features = np.hstack((features, np.array([np.log(np.var(x, axis=1)) for x in trans])))

    if feature_extraction_config['spec']['use']:
        import matlab.engine
        eng = matlab.engine.start_matlab()
        
        # Get feature using MATLAB script.
        res_features = np.array([[np.ravel(np.array(eng.engineer_features(matlab.double(list(signal)), 160))) 
            for signal in interval] for interval in signals])
        features = np.hstack((features, res_features.reshape(res_features.shape[0], -1)))

    
    # Return extracted features.
    return features


def proc_features(features, target, feature_extraction_config):
    """
    Perform processing and dimensionality reduction of features using
    specified methods.

    Args:
        features (numpy.ndarray): features of instances.
        target (numpy.ndarray): target values for the instances.
        (dict): dictionary in specified form specifying the parameters of the feature preprocessing
                and dimensionality reduction process.

    Returns:
        (numpy.ndarray): processed feature vectors.

    """
    
    # If more features than specified, perform dimensionality reduction using specified method.
    if not features.shape[1] == feature_extraction_config['dim_rdc']['dim']:
        if feature_extraction_config['dim_rdc']['rdc_method'] == 'LDA':
            lda = LinearDiscriminantAnalysis(n_components=feature_extraction_config['dim_rdc']['dim'])
            return lda.fit_transform(features, target)
        else:
            raise ValueError("Invalid dimensionality reduction method '{0}' specified in the configuration file."
                    .format(feature_extraction_config['dim_rdc']['rdc_method']))
    else:
        return features


