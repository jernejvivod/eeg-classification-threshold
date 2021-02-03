import numpy as np
from scipy.signal import fir_filter_design as ffd


def get_filter(sampling_freq, f_pass, f_stop, taps):
    """Get FIR filter coefficients using the Remez exchange algorithm.

    Args:
        f_pass (float): Passband edge.
        f_stop (float): Stopband edge.
        taps (int): Number of taps or coefficients in the resulting filter.

    Returns:
        (numpy.ndarray): Computed filter coefficients.
    
    """
    return ffd.remez(taps, [0, f_pass/sampling_freq, f_stop/sampling_freq, 0.5], [0, 1]) 

