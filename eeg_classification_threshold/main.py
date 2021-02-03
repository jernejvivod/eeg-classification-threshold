from data_utils import data_utils
from feature_extraction import feature_extraction
from results_utils.scatter_plot import get_scatter_plot

import argparse


def main():
    """
    Main script for the project

    Author: Jernej Vivod
    """

    # Parse arguments.
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", type=str, choices=["rf", "svm", "lda"], default="lda",
            help="classification model to use")
    parser.add_argument("--activities-only", action='store_true',
            help="classify only between the activities")
    parser.add_argument("--dataset", type=str, default="./demo_data/S001/",
            help="dataset folder to use")
    parser.add_argument('--overlap', type=float, default=0.0,
            help="overlap between extracted intervals")
    parser.add_argument('--interval-len', type=float, default=4.0,
            help="interval length (in seconds)")
    args = parser.parse_args()


    # Segment signals into intervals and compute their labels. Also obtain sampling frequency.
    intervals, target, sampling_freq = data_utils.get_joined_intervals(args.dataset, interval_len_s=args.interval_len, overlap=args.overlap)

    # If classifying only between activities (exclude rest).
    if args.activities_only:
        intervals = intervals[target != 0, :, :]
        target = target[target != 0]

    # Parse features to be extracted from xml file.
    feature_extraction_config = feature_extraction.parse_feature_extraction_config('./feature_extraction_config.yml')

    # Extract features.
    features = feature_extraction.extract_features(signals=intervals, sampling_freq=sampling_freq, feature_extraction_config=feature_extraction_config, target=target) 
    
    # Process features - perform dimensionality reduction.
    features_proc = feature_extraction.proc_features(features=features, target=target, feature_extraction_config=feature_extraction_config)

    # Save scatter plot.
    get_scatter_plot(features_proc, target, plot_thresh=feature_extraction_config['thresh']['plot_thresh'], 
            thresh_method=feature_extraction_config['thresh']['thresh_method'])


# Run evaluations.
if __name__ == "__main__":
    main()

