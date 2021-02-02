import numpy as np
import os
from sklearn.metrics import classification_report


def get_classification_report(clf, data_test, target_test, clf_name, cr_save_path):
    """Compute and save classification report to file.

    Args:
        clf (object): classifier for which to plot the confusion matrix.
        data_test (numpy.ndarray): Test data samples.
        target_test (numpy.ndarray): test data labels (target variable values).
        clf_name (str): name of the classifier.
        cr_save_path (str): path to file for storing the classification report.
    """
    
    # Compute classification report and write to file.
    cr = classification_report(target_test, clf.predict(data_test), target_names=np.unique(target_test).astype(str))
    with open(cr_save_path, "a") as f:
        if os.stat(cr_save_path).st_size != 0:
            f.write("\n\n\n")
        f.write(f"# {clf.name}\n")
        f.write(cr)
    
