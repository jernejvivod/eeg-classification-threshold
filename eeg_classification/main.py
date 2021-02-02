from data_utils import data_utils
from feature_extraction import feature_extraction
from results_utils.confusion_matrix import get_confusion_matrix
from results_utils.classification_report import get_classification_report

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm

import argparse

# Evaluation using a train-test split.
def eval_tts(intervals, target, clf, features_to_extract, test_size=0.2, shuffle=False):
    
    # Split signal intervals into training and test intervals.
    intervals_train, intervals_test, y_train, y_test = train_test_split(intervals, target, test_size=test_size, shuffle=shuffle)

    # Perform feature extraction for training and test intervals.
    x_train, features_to_extract = feature_extraction.extract_features(intervals_train, features_to_extract, set_type="train", target=y_train)
    x_test, _ = feature_extraction.extract_features(intervals_test, features_to_extract, set_type="test")

    # Fit classifier on training and test sets.
    clf.fit(x_train, y_train)

    # Get resulting classification report for trained classifier on test set.
    get_classification_report(clf, x_test, y_test, clf.name, "./results/classification_reports.txt")

    # Get resulting confusion matrix for trained classifier on test set.
    get_confusion_matrix(clf, x_test, y_test, clf.name,  "".join(["./results/confusion_matrices/conf_", clf.name, ".png"]))


def main():

    # Parse arguments.
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", type=str, choices=["rf", "svm", "lda"], default="lda",
            help="classification model to use")
    parser.add_argument("--dataset", type=str, default="./demo_data/S001R13.edf",
            help="dataset to use")
    parser.add_argument("--use-spec", type=bool, default=False,
            help="use spectral (and some additional) features")
    parser.add_argument('--overlap', type=float, default=0.0,
            help="overlap between extracted intervals")
    parser.add_argument('--interval-len', type=float, default=4.0,
            help="interval length (in seconds)")
    args = parser.parse_args()

    
    # Segment signals into intervals and compute their labels. Also obtain sampling frequency.
    intervals, target, sampling_freq = data_utils.get_intervals(data_utils.parse_raw_data(args.dataset), interval_len_s=args.interval_len, overlap=args.overlap)

    # Construct specified classification pipeline.
    if args.method == "lda":
        clf_pipeline = Pipeline([("scaling", RobustScaler()), ("clf", LinearDiscriminantAnalysis())])
        clf_pipeline.name = "LDA"
    elif args.method == "rf":
        clf_pipeline = Pipeline([("scaling", RobustScaler()), ("clf", RandomForestClassifier())])
        clf_pipeline.name = "RF"
    elif args.method == "svm":
        clf_pipeline = Pipeline([("scaling", RobustScaler()), ("clf", svm.SVC())])
        clf_pipeline.name = "SVC"

    # Initialize feature extraction parameters (see extract_features function documentation)
    features_to_extract = {"CSP" : {"samp_freq" : sampling_freq, 
                                    "f_pass" : 8.0, 
                                    "f_stop" : 30.0, 
                                    "taps" : 30, 
                                    "n_components" : 5},
                           "spec" : {}}
    if not (args.use_spec):
        del features_to_extract["spec"]
    
    # Evaluate classification performance using a train-test split.
    eval_tts(intervals, target, clf_pipeline, features_to_extract, test_size=0.2, shuffle=False)


# Run evaluations.
if __name__ == "__main__":
    main()

