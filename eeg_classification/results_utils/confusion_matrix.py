import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import plot_confusion_matrix


def get_confusion_matrix(clf, data_test, target_test, clf_name, cm_save_path):
    """Plot and save confuction matrix for specified classifier.

    Args:
        clf (object): classifier for which to plot the confusion matrix.
        data_test (numpy.ndarray): Test data samples.
        target_test (numpy.ndarray): test data labels (target variable values).
        clf_name (str): name of the classifier (used for plot labelling).
        cm_save_path (str): path for saving the confusion matrix plot.

    """

    # Set number of decimals.
    np.set_printoptions(precision=2)

    # Plot confusion matrix and save plot.
    disp = plot_confusion_matrix(clf, data_test, target_test,
                                 display_labels=np.unique(target_test).astype(str),
                                 cmap=plt.cm.Blues,
                                 normalize='true',
                                 xticks_rotation='vertical')

    # UNCOMMENT TO SET TITLE.
    disp.ax_.set_title("Normalized Confusion Matrix - " + clf_name)
    disp.figure_.set_size_inches(9.0, 9.0, forward=True)
    plt.tight_layout()
    plt.savefig(cm_save_path)
    plt.clf()
    plt.close()

