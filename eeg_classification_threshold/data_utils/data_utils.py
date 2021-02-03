import numpy as np
import pyedflib
from sklearn.preprocessing import LabelEncoder
import glob


def parse_raw_data(path):
    """
    Parse raw data from edf file and return required information in the form of
    a dictionary.

    Args:
        path (string): path to the edf file.

    Returns:
        (dict): dictionary containing the required information (signals, labels and sampling frequency).
    """
    
    # Open edf file and check if all sampling frequencies equal.
    f = pyedflib.EdfReader(path)
    if (not all(f.getSampleFrequency(0) == f.getSampleFrequencies())):
        raise ValueError("Sampling frequencies not equal for all signals")

    # Construct resulting dictionary and return it.
    res = {"signals" : np.vstack([f.readSignal(idx) for idx in range(len(f.getSignalLabels()))]), 
           "samp_freq" : f.samplefrequency(0),
           "annotations" : (lambda x : {"start" : x[0], "duration" : x[1], "label" : LabelEncoder().fit_transform(x[2])})(f.readAnnotations())}
    # Trim redundant trailing signal.
    res["signals"] = res["signals"][:, :(int(sum([res["samp_freq"]*dur for dur in res["annotations"]["duration"]])))]
    f._close()
    return res


def get_intervals(raw_data, interval_len_s=2.0, overlap=0.5):
    """
    Transform raw data into signal segments of specified length and with specified
    overlap. Label it with a label to which the majority of the constituent samples belong.

    Args:
        raw_data (dict): raw data dictionary obtained by the parse_raw_data function.
        interval_len_s (float): length of each interval (in seconds).
        overlap (float): overlap between neighbouring intervals.
    
    Returns:
        (tuple): a numpy array containing the signal intervals in rows,
                 a numpy array containing the labels (transformed to numerical values) for each interval.
    """
    
    # Get number of samples in one interval.
    n_samples_interval = int(round(interval_len_s*raw_data["samp_freq"]))

    # Get number of samples to move forward to achieve specified overlap.
    move_add = n_samples_interval - int(np.ceil(n_samples_interval*overlap))

    # Get number of intervals.
    n_intervals = (raw_data["signals"].shape[1] - (n_samples_interval - move_add))//move_add

    # Allocate array for storing signal intervals.
    intervals = np.empty((n_intervals,), dtype=object)

    # Allocate array for target labels.
    target = np.empty((n_intervals,), dtype=int)
    
    # Get labels for each sample.
    labels = np.hstack([np.repeat(lab, int(raw_data["samp_freq"]*dur)) 
        for (dur, lab) in zip(raw_data["annotations"]["duration"], raw_data["annotations"]["label"])])
    
    # Construct intervals.
    start_idx = 0
    for interval_idx in range(n_intervals):

        # intervals[interval_idx] = raw_data["signals"][:, start_idx:(min(start_idx+n_samples_interval, raw_data["signals"].shape[1]))]
        intervals[interval_idx] = raw_data["signals"][:, start_idx:start_idx + n_samples_interval]

        # Label of interval is the most common label of constituent samples.
        # u, c = np.unique(labels[start_idx:(min(start_idx+n_samples_interval, labels.shape[0]))], return_counts=True)
        u, c = np.unique(labels[start_idx:start_idx + n_samples_interval], return_counts=True)
        target[interval_idx] = u[np.argmax(c)]
        start_idx += move_add
        
    # Return intervals and their labels.
    return np.stack(intervals), target, raw_data["samp_freq"]


def get_joined_intervals(folder_path, interval_len_s=2.0, overlap=0.5):
    """
    Parse and join data from files into a single long recording.

    Args:
        folder_path (str): path to folder containing the data files.
        interval_len_s (float): length of each interval (in seconds).
        overlap (float): overlap between neighbouring intervals.
    
    Returns:
        (tuple): a numpy array containing the signal intervals in rows,
                 a numpy array containing the labels (transformed to numerical values) for each interval.
    """
    
    # Initialize variable for keeping the previous obtained sampling frequency.
    samp_freq_prev = -1.0

    # Initialize lists for intervals and target values.
    intervals = list()
    target = list()

    for f_name in glob.glob(folder_path + "*.edf" if folder_path[-1] == "/" else folder_path + "/*.edf"):
        
        # Parse data in next file and split into intervals.
        raw_data_nxt = parse_raw_data(f_name)
        intervals_nxt, target_nxt, samp_freq_nxt = get_intervals(raw_data_nxt, interval_len_s, overlap)

        # If first data file, set sampling frequency variable.
        if samp_freq_prev == -1.0:
            samp_freq_prev = samp_freq_nxt
        elif samp_freq_prev != samp_freq_nxt: 
            # If sampling frequency of current data file not equal to that of previous, raise exception.
            raise ValueError("Sampling frequencies of records in specified folder not equal.")
        else:
            # Append intervals and target values obtained from current data file to lists.
            intervals.append(intervals_nxt)
            target.append(target_nxt)
    
    # Stack intervals and target values and return along with sampling frequency.
    return np.vstack(intervals), np.hstack(target), samp_freq_prev

