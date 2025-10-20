"""
EnICCS PLS-DA Classification Module

This module handles Partial Least Squares Discriminant Analysis (PLS-DA) classification for EnICCS.
It includes functions for data preprocessing, model training, optimization, validation,
and prediction for multiclass classification of hyperspectral images.
"""

import numpy as np
from scipy.optimize import curve_fit
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, f1_score
from sklearn.neighbors import LocalOutlierFactor


def reshape_image_to_table(image):
    """
    Reshapes the hyperspectral image to a 2D table for analysis
    """
    ns, nr, nc = image.shape
    hyperspectral_2d = image.reshape((ns, nr * nc)).T.copy()  # Transpose to (pixels, bands)
    return hyperspectral_2d


def get_pixellabels(mask, hyperspectral_2d):
    """
    Matches the mask to the hyperspectral image and return the labeled pixels
    """
    labels = mask.flatten()  # Flatten the mask to match the hyperspectral image shape
    labeled_pixels = hyperspectral_2d[labels >= 0, :]  # Select pixels with labels
    labels = labels[labels >= 0]  # Keep corresponding labels

    return labeled_pixels, labels


def outlier_removal(labeled_pixels, labels, contamination: float = 0.25, n_neighbors=50):
    """
    Performs basic outlier removal using Local Outlier Factor.
    Removes likely mislabeled pixels from classes 2 and 3 (cloud and cloud shadow)

    :param labeled_pixels: 2D array of labeled pixels
    :param labels: 1D array of labels
    :param n_neighbors: number of neighbors to consider for LOF
    :param contamination: approximate proportion of outliers in the data

    :return: non_outlier_labeled_pixels, non_outlier_labels

    """
    non_outlier_labeled_pixels = None
    non_outlier_labels = None

    # for classes 2 and 3 remove outliers
    for i in [2, 3]:
        class_samples = labeled_pixels[labels == i, :]
        lof = LocalOutlierFactor(n_neighbors=n_neighbors, contamination=contamination)
        outliers = lof.fit_predict(class_samples)
        class_pixels_no_outliers = class_samples[outliers != -1]
        class_labels_no_outliers = np.ones(class_pixels_no_outliers.shape[0], dtype=int) * i

        if non_outlier_labeled_pixels is None:
            non_outlier_labeled_pixels = class_pixels_no_outliers
            non_outlier_labels = class_labels_no_outliers
        else:
            non_outlier_labeled_pixels = np.vstack((non_outlier_labeled_pixels, class_pixels_no_outliers))
            non_outlier_labels = np.hstack((non_outlier_labels, class_labels_no_outliers))

    # add classes that were not assessed for outliers back to the data (class 0 and 1)
    for i in [0, 1]:
        class_samples = labeled_pixels[labels == i, :]
        class_labels = np.ones(class_samples.shape[0], dtype=int) * i
        if non_outlier_labeled_pixels is None:
            non_outlier_labeled_pixels = class_samples
            non_outlier_labels = class_labels
        else:
            non_outlier_labeled_pixels = np.vstack((non_outlier_labeled_pixels, class_samples))
            non_outlier_labels = np.hstack((non_outlier_labels, class_labels))

    return non_outlier_labeled_pixels, non_outlier_labels


def balance_classes(labeled_pixels, labels, num_samples):
    """
    Balances the classes by randomly selecting n samples from each class

    :param labeled_pixels: 2D array of labeled pixels
    :param labels: 1D array of labels
    :param num_samples: number of samples to select from each class

    :return: balanced_pixels, balanced_labels
    """

    unique, counts = np.unique(labels, return_counts=True)

    # Handle case where num_samples is None or classes don't have enough samples
    if num_samples is None:
        min_class_size = min(counts)  # Use smallest class size
    else:
        min_class_size = min(num_samples, counts.min())  # Use smaller of requested or available

    # Randomly select the same number of samples from each class
    balanced_pixels = np.zeros((min_class_size * len(unique), labeled_pixels.shape[1]))
    balanced_labels = np.zeros(min_class_size * len(unique), dtype=int)
    for i, label in enumerate(unique):
        class_samples = labeled_pixels[labels == label, :]
        random_indices = np.random.choice(class_samples.shape[0], min_class_size, replace=False)
        balanced_pixels[i * min_class_size:(i + 1) * min_class_size, :] = class_samples[random_indices, :]
        balanced_labels[i * min_class_size:(i + 1) * min_class_size] = label

    return balanced_pixels, balanced_labels


def split_data(balanced_pixels, balanced_labels, test_size=0.3, random_state=321):
    """
    Splits the data into training and testing sets. The labels are one-hot encoded for training.
    Split is mostly necessary for if PLS-DA model Auto_optimize is = True

    :param balanced_pixels: 2D array of balanced pixels
    :param balanced_labels: 1D array of balanced labels
    :param test_size: proportion of the data to be used for testing
    :param random_state: random seed for reproducibility
    """
    X_train, X_test, y_train, y_test = train_test_split(balanced_pixels,
                                                        balanced_labels,
                                                        test_size=test_size,
                                                        random_state=random_state)

    y_train = OneHotEncoder().fit_transform(y_train.reshape(-1, 1)).toarray()
    # y_test can stay in vector form for evaluation

    return X_train, X_test, y_train, y_test


def multiclass_plsda(X, y, n_components):
    """
    Fits a PLS-DA model to the data

    :return: pls_da model object
    """
    # one hot encoding
    # y = to_categorical(y)
    # PLS-DA Model
    pls_da = PLSRegression(n_components=n_components)
    pls_da.fit(X, y)
    return pls_da


def pls_da_predict(model, X):
    """
    Predicts the class labels using the PLS-DA model.
    Turns the continuous outputs into categorical predictions.

    :return: categorical predictions
    """
    continuous_outputs = model.predict(X)
    categorical_predictions = np.argmax(continuous_outputs, axis=1) # Assuming one-hot encoded targets
    return categorical_predictions


def f1_weighted_scorer(model, X, y_test):
    """
    Custom scorer for cross-validation that calculates the F1 score for the weighted average of
    the classes. sklearn framework based. handles one-hot encoding
    """
    true_y = np.argmax(y_test, axis=1)  # y_train is one-hot encoded
    predicted_y = pls_da_predict(model, X) # y_pred is not one-hot encoded due to multiclass_plsda
    return f1_score(true_y, predicted_y, average='weighted')


def find_optimal_ncomp_via_saturation_point(n_comp_list, f1_scores_list, plot=False):
    """
    Finds the optimal number of components by fitting a logistic curve to the F1 scores and
    extracting n_comp closest to the saturation point.
    Only used when auto_optimize is set to True

    :param n_comp_list: list of possible number of components
    :param f1_scores_list: list of F1 scores corresponding to the number of components
    :param plot: boolean to plot the saturation curve fitting

    :return: optimal number of components, corresponding F1 score
    """
    x_data = n_comp_list
    Y_data = f1_scores_list

    # Logistic function
    def logistic(x, L, k, x0):
        return L / (1 + np.exp(-k * (x - x0)))

    # Curve fitting
    popt, _ = curve_fit(logistic, x_data, Y_data, maxfev=10000)

    # Extracting the saturation point (L)
    L = popt [0]
    saturation_point = L

    # Extracting the closest data point to the saturation point
    absolude_diff = np.abs(Y_data - saturation_point)

    # calculate the percentage of the range of absolute difference
    value_range = (absolude_diff.max() - absolude_diff.min())

    # find the first index where the difference is less than j% of the percent_range of the saturation point
    # i.e.j% of the range of absolute difference (to equate for cases where overall change is very small)
    threshold = (value_range * 0.005) * saturation_point

    # Find indices where absolude_diff is less than the threshold
    indices_below_threshold = np.where(absolude_diff < threshold)[0]

    if indices_below_threshold.size > 0:
        # If values are below the threshold, take the first one
        closest_index_range_value_range = indices_below_threshold[0]
    else:
        # If no values are below the threshold, find the closest one
        closest_index_range_value_range = np.argmin(np.abs(absolude_diff - threshold))

    if plot:
        plt.scatter(x_data, Y_data, label='Training F1 scores')
        plt.plot(x_data, logistic(x_data, *popt), label='Fitted curve')
        plt.axhline(y=saturation_point, color='r', linestyle='--', label='Saturation level')
        plt.scatter(x_data[closest_index_range_value_range], Y_data[closest_index_range_value_range]
                    , facecolors='none', edgecolors='b', linewidths=2, s=150,
                    label='Closest Data Point')

        plt.xlabel('Number of latent variables')
        plt.ylabel('Training F1 scores')
        plt.title('PLS-DA latent variable selection via saturation curve fitting')
        plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
        plt.legend(loc='lower right')
        plt.show()

    # return closest n_comp and corresponding F1 score
    return x_data[closest_index_range_value_range], Y_data[closest_index_range_value_range]


def plsda_model_builder(X_train, y_train,
                        auto_optimize = False,
                        plot=False,
                        verbose=False,
                        n_jobs=-1):
    """
    Builds a PLS-DA model with the optimal number of components.
    Integrates auto optimization argument.

    :param X_train: training data
    :param y_train: training labels
    :param auto_optimize: boolean to optimize the number of components
    :param plot: boolean to plot the saturation curve fitting

    :return: pls_da model object
    """

    if auto_optimize:
        # cross validate to find optimal number of components
        if verbose:
            print('Optimizing number of components for PLS-DA model')

        _, _, pls_da = cv_optimize_n_components(X_train, y_train,
                                                max_components=20,
                                                cv=10,
                                                njobs=n_jobs,
                                                plot=plot,
                                                verbose=verbose)
    else:
        pls_da = multiclass_plsda(X_train, y_train, n_components=10)

    return pls_da


def cv_optimize_n_components(X_train, y_train, max_components,
                             cv=10,
                             njobs=-1,
                             verbose=False,
                             plot=False):
    """
    Applies cross-validation to find the optimal number of components for the PLS-DA model.
    Returns the F1 scores for each number of components and the optimal number of components.

    :param X_train: training data
    :param y_train: training labels
    :param max_components: maximum number of components to test
    :param cv: number of cross-validation folds
    :param njobs: number of parallel jobs (default = -1)
    :param plot: boolean to plot the F1 score vs number of components

    Note: f1_weighted_scorer is a sklaern compatible scorer defined above. It is passed into
    cross_val_score directly.
    """

    n_components_list = list(range(2, max_components))

    f1_scores = []
    for i in n_components_list:
        pls_da = multiclass_plsda(X_train, y_train, i)  # make the model object
        scores = cross_val_score(pls_da, X_train, y_train,
                                 cv=cv,
                                 scoring=f1_weighted_scorer,
                                 verbose=0,
                                 n_jobs=njobs)

        f1_scores.append(scores.mean())


    optimal_n_components, _ = find_optimal_ncomp_via_saturation_point(
        n_components_list, f1_scores, plot=plot)
    if verbose:
        print(f'Optimal number of components: {optimal_n_components}, '
              f'from saturation point analysis with CV-F1 score: {max(f1_scores):.2f}')

    # build the final model with the optimal number of components
    pls_da = multiclass_plsda(X_train, y_train, optimal_n_components)

    return f1_scores, optimal_n_components, pls_da


def get_vip(pls_da_model):
    """
    Extract Variable Importance in Projection (VIP) scores (aka feature importance) from the PLS-DA
    model. Can help to understand model results and identify important features.

    :param pls_da_model: PLS-DA model object

    :return: VIP scores for each feature (here Band)

    Note:   Implementation after Peerbhay et al., 2013
            https://doi.org/10.1016/j.isprsjprs.2013.01.013.
            and sources therein.
    """
    T = pls_da_model.x_scores_    # Scores -> new coordinates in PLS space
    W = pls_da_model.x_weights_   # Weights -> contribution of feature to LVs
    # P = pls_da_model.x_loadings_  # X loadings -> correlation of features with LVs
    Q = pls_da_model.y_loadings_  # Y loadings -> correlation of targets with LVs

    # Calculate the sum of squares of the scores, weighted by the Y loadings aka.
    SSQ = np.sum(T ** 2, axis=0) * np.sum(Q ** 2, axis=0)
    total_variance = np.sum(SSQ)

    # Calculate the contribution of each variable to the components
    weights_squared = W ** 2
    contribution = np.dot(weights_squared, (SSQ / total_variance))

    # VIP scores for each variable
    vip = np.sqrt(pls_da_model.n_features_in_ * contribution)

    # subset and order descending VIP
    vip_df = pd.DataFrame(vip, columns=['VIP'])
    vip_df['Band_ID'] = range(1, vip.shape[0] + 1)

    return vip_df


def predict_on_image(hyperspectral_image, pls_da_model):
    """
    Applies trained PLS-DA model to a hyperspectral image for prediction.
    Returns a predicted mask image.
    """
    # reshape image to table
    hyperspectral_2d = reshape_image_to_table(hyperspectral_image)

    predicted_mask = pls_da_predict(pls_da_model, hyperspectral_2d)

    # reshape predicted mask to image shape
    predicted_mask_image = predicted_mask.reshape(hyperspectral_image[0].shape)
    return predicted_mask_image


def get_validation_report(X_test, y_test, pls_da_model, format_output=False, verbose=False):
    """
    Generates a sklearn validation report/F1 Score for the final PLS-DA model.
    Just for reporting purposes.
    Formats output for collection and comparison.
    """
    # get validation report
    y_pred = pls_da_predict(pls_da_model, X_test)
    f1_score_rd = np.round(f1_score(y_test, y_pred, average='weighted'), 2)

    if verbose:
        print('PLS-DA validation F1 score: ', f1_score_rd)

    if format_output:
        report = classification_report(y_test, y_pred, output_dict=True)
        report_df = pd.DataFrame(report).transpose()
        return report_df
    return f1_score_rd



# wrapper
def train_plsda(hs_image_obj, mask_obj, num_samples = 3000, max_components=20, cv=10, njobs=-1):
    """
    A standalone wrapper function to train a PLS-DA model for multiclass classification.

    This function provides a complete training pipeline including preprocessing,
    model training, and validation. Can be used independently from the main EnICCS pipeline.

    For the full EnICCS pipeline including postprocessing, use classify_image() in main module.

    :param hs_image_obj: HsImage object
    :param mask_obj: Mask object with multiclass_mask
    :param num_samples: number of samples to balance each class to
    :param max_components: maximum number of PLS components to test
    :param cv: number of cross-validation folds
    :param njobs: number of parallel jobs

    :return: pls_da model, VIP_df DataFrame
    """

    # reshape image to table
    hyperspectral_2d = reshape_image_to_table(hs_image_obj.image)

    # get pixel labels
    labeled_pixels, labels = get_pixellabels(mask_obj.multiclass_mask, hyperspectral_2d)

    # balance classes
    balanced_pixels, balanced_labels = balance_classes(labeled_pixels, labels,
                                                       num_samples=num_samples)

    # remove outliers
    balanced_pixels, balanced_labels = outlier_removal(balanced_pixels, balanced_labels)

    # split data
    X_train, X_test, y_train, y_test = split_data(balanced_pixels, balanced_labels)

    # cross validate to find optimal number of components
    f1_scores, optimal_n_components, pls_da = cv_optimize_n_components(X_train, y_train,
                                                                       max_components,
                                                                       cv=cv,
                                                                       njobs=njobs)

    # get VIP scores
    vip_df = get_vip(pls_da)

    # # only print F1 score
    get_validation_report(X_test, y_test, pls_da, format_output=False)

    return pls_da, vip_df
