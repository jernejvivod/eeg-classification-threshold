import matplotlib.pyplot as plt
import numpy as np
import random


def get_scatter_plot(features_proc, target, plot_thresh=False, thresh_method=None):
    """
    Plot scatter plot of instances described by specified features. Also plot decision threshold
    if specified.

    Args:
        features_proc (numpy.ndarray):
        target (numpy.ndarray):
        plot_thresh (bool):
        thresh_method (str):

    Returns:
        Returns None on successful completion.
    """

    # Get plot bounds.
    bound_min_f1 = np.min(features_proc[:, 0]) - 0.2
    bound_max_f1 = np.max(features_proc[:, 0]) + 0.2
    bound_min_f2 = np.min(features_proc[:, 1]) - 0.2
    bound_max_f2 = np.max(features_proc[:, 1]) + 0.2
    
    # If plotting threshold.
    if plot_thresh:
        
        # Get classifier and set title depending on decision method.
        if thresh_method == 'logistic_regression':
            from sklearn.linear_model import LogisticRegression
            clf = LogisticRegression()
            plt.title("Decision Boundary - Logistic Regression")
        elif thresh_method == 'svm':
            from sklearn.svm import SVC
            clf = SVC(kernel='rbf')
            plt.title("Decision Boundary - Support Vector Machine (RBF kernel)")
        elif thresh_method == 'decision_tree':
            from sklearn.tree import DecisionTreeClassifier
            from sklearn import tree
            clf = DecisionTreeClassifier(max_depth=2)
        else:
            raise ValueError("Unknown thresh_method parameter value '{0}'".format(thresh_method))
        
        # Fit classifier.
        clf.fit(features_proc, target)

        # If classifier decision tree, plot the tree and save plot.
        if thresh_method == 'decision_tree':
            tree.plot_tree(clf)
            plt.savefig('./results/decision_tree.png')
            plt.clf()
            plt.close()
            plt.title("Decision Boundary - Decision Tree")
        
        # Plot decision boundary.
        xx, yy = np.mgrid[bound_min_f1:bound_max_f1:0.01, bound_min_f2:bound_max_f2:0.01]
        grid = np.c_[xx.ravel(), yy.ravel()]
        probs = clf.predict(grid).reshape(xx.shape)
        plt.contourf(xx, yy, probs, cmap=plt.cm.RdYlBu, alpha=0.3)
       

    # Initialize lists for plotting legend.
    legend_a = list()
    legend_b = list()
    
    # Get unique target values.
    target_vals = np.unique(target)

    # Initialize list of colors.
    colors = ['red', 'green', 'blue', 'black', 'yellow']
    if len(colors) < len(target_vals):
        raise ValueError("Not enough colors in 'colors' list")

    # Plot scatterplot of instances.
    for idx, target_nxt in enumerate(np.unique(target)):
        sct_nxt = plt.scatter(features_proc[target == target_nxt, 0], features_proc[target == target_nxt, 1], edgecolor='white', color=colors[idx])
        legend_a.append(sct_nxt)
        legend_b.append("T" + str(target_nxt))

    
    # Set plot axis limits.
    plt.xlim((bound_min_f1, bound_max_f1))
    plt.ylim((bound_min_f2, bound_max_f2))

    # Annotate and save plot.
    plt.xlabel("f1")
    plt.ylabel("f2")
    plt.legend(legend_a, legend_b)
    plt.tight_layout()
    if plot_thresh:
        plt.savefig('./results/boundary_' + thresh_method + '_' + str(len(target_vals)) + '.png')
    else:
        plt.savefig('./results/scatter_' + str(len(target_vals)) + '.png')

    plt.clf()
    plt.close()


