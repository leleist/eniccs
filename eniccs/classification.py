import numpy as np
from scipy.optimize import curve_fit
from scipy.ndimage import binary_closing, binary_opening, binary_dilation
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

# from tensorflow.python.keras.utils.np_utils import to_categorical
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import make_scorer, classification_report, confusion_matrix, f1_score
from sklearn.neighbors import LocalOutlierFactor


def reshape_image_to_table(image):
    """
    Reshapes the hyperspectral image to a 2D table for analysis
    """
    ns, nr, nc = image.shape
    hyperspectral_2D = image.reshape((ns, nr * nc)).T.copy()  # Transpose to (pixels, bands)
    return hyperspectral_2D


def get_pixellabels(mask, hyperspectral_2D):
    """
    Matches the mask to the hyperspectral image and return the labeled pixels
    """
    labels = mask.flatten()  # Flatten the mask to match the hyperspectral image shape
    labeled_pixels = hyperspectral_2D[labels >= 0, :]  # Select pixels with labels
    labels = labels[labels >= 0]  # Keep corresponding labels

    return labeled_pixels, labels


def outlier_removal(labeled_pixels, labels, n_neighbors=50, contamination=0.25):
    """
    Performs basic outlier removal using Local Outlier Factor to narrow down the scope for ML Training.
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

    # print(np.unique(labels, return_counts=True))

    for i in [2, 3]:
        class_samples = labeled_pixels[labels == i, :]
        lof = LocalOutlierFactor(n_neighbors=n_neighbors, contamination=contamination)
        outliers = lof.fit_predict(class_samples)
        class_pixels_no_outliers = class_samples[outliers != -1]
        class_labels_no_outliers = np.ones(class_pixels_no_outliers.shape[0], dtype=int) * i
        # print(f'Class {i} outliers removed: {class_samples.shape[0] - class_pixels_no_outliers.shape[0]} samples')
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

    # print(f"remaining samples after outlier removal: {non_outlier_labeled_pixels.shape[0]}")

    return non_outlier_labeled_pixels, non_outlier_labels


def balance_classes(labeled_pixels, labels, n=3000):
    """
    Balances the classes by randomly selecting n samples from each class

    :param labeled_pixels: 2D array of labeled pixels
    :param labels: 1D array of labels
    :param n: number of samples to select from each class

    :return: balanced_pixels, balanced_labels
    """
    unique, counts = np.unique(labels, return_counts=True)
    # min_class_size = counts.min()
    unique, counts = np.unique(labels, return_counts=True)
    # min_class_size = counts.min()
    min_class_size = n
    # print(f'New minimum class size: {n} samples')

    # Randomly select the same number of samples from each class but make sure to not mix index with value
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
    X_train, X_test, y_train, y_test = train_test_split(balanced_pixels, balanced_labels, test_size=test_size, random_state=random_state)

    # y_train = to_categorical(y_train) # replaced due to high overhead of importing tensorflow
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
    categorical_predictions = np.argmax(continuous_outputs, axis=1)  # Assuming one-hot encoded targets
    return categorical_predictions


def f1_weighted_scorer(model, X, y_test):
    """
    Custom scorer for cross-validation that calculates the F1 score for the weighted average of the classes
    sklearn framework based
    """
    # TODO: Check if this is even necessary because y_test is not one-hot encoded as to function split_data()
    true_y = np.argmax(y_test, axis=1)  # y_train is one-hot encoded
    predicted_y = pls_da_predict(model, X) # y_pred is not one-hot encoded due to multiclass_plsda return
    return f1_score(true_y, predicted_y, average='weighted')


def find_optimal_ncomp_via_saturation_point(n_comp_list, f1_scores_list, plot_bool=False):
    """
    Finds the optimal number of components by fitting a logistic curve to the F1 scores and extracting n_comp
    closest to the saturation point.
    Only used when auto_optimize is set to True

    :param n_comp_list: list of possible number of components
    :param f1_scores_list: list of F1 scores corresponding to the number of components
    :param plot_bool: boolean to plot the saturation curve fitting

    :return: optimal number of components, corresponding F1 score
    """
    x_data = n_comp_list
    Y_data = f1_scores_list

    # Logistic function
    def logistic(x, L, k, x0):
        return L / (1 + np.exp(-k * (x - x0)))

    # Curve fitting
    popt, pcov = curve_fit(logistic, x_data, Y_data, maxfev=10000)

    # Extracting the saturation point (L)
    L, k, x0 = popt
    saturation_point = L

    ## Extracting the closest data point to the saturation point
    absolude_diff = np.abs(Y_data - saturation_point)

    # find the first index where the difference is less than j% of the saturation point
    closest_index_percent = np.where(absolude_diff < 0.005 * saturation_point)[0][0]

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

    closest_index = np.argmin(np.abs(Y_data - saturation_point))
    closest_data_point = (x_data[closest_index], Y_data[closest_index])

    # Extracting the point with the max F1 score (for comparison)
    max_F1_index = np.argmax(Y_data)
    max_F1 = Y_data[max_F1_index]

    # calculating the difference between the saturation point and the max F1 score
    diff = max_F1 - saturation_point

    # TODO: add fitted line quality evaluation with a criterion on pcov


    if plot_bool: # TODO: make this plot more concise (remove at least 2 dots, move value prints into the plot )
        plt.scatter(x_data, Y_data, label='Training F1 scores')
        plt.plot(x_data, logistic(x_data, *popt), label='Fitted curve')
        # abline the saturation point on y axis
        plt.axhline(y=saturation_point, color='r', linestyle='--', label='Saturation level')
        # plot the saturation point on the curve
        #plt.scatter(closest_data_point[0], closest_data_point[1], color='r', label='Closest data Point')
        # plot the point with the max F1 score
        #plt.scatter(x_data[max_F1_index], Y_data[max_F1_index], color='g', label='Max F1-score')
        # plt.scatter(x_data[closest_index_percent], Y_data[closest_index_percent], color='y', label='Closest Data Point within 0.5% of saturation level')
        plt.scatter(x_data[closest_index_range_value_range], Y_data[closest_index_range_value_range], facecolors='none', edgecolors='b', linewidths=2, s=150, label='Closest Data Point')
        plt.xlabel('Number of latent variables')
        plt.ylabel('Training F1 scores')
        plt.title('PLS-DA latent variable selection via saturation curve fitting')
        plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
        plt.legend(loc='lower right')
        plt.show()


    # print(f'Estimated Saturation Point at F1= {saturation_point} euqlas n_components = {x_data[closest_index]}')
    # print(f' max F1: {max_F1} at n_components = {x_data[max_F1_index]}, with a difference of {diff} between the saturation point and the max F1 score.')

    return x_data[closest_index_range_value_range], Y_data[closest_index_range_value_range]

def PLSDA_model_builder(X_train, y_train, auto_optimize = False, plot_bool=False):
    """
    Builds a PLS-DA model with the optimal number of components.
    Integrates auto optimization argument.

    :param X_train: training data
    :param y_train: training labels
    :param auto_optimize: boolean to optimize the number of components
    :param plot_bool: boolean to plot the saturation curve fitting

    :return: pls_da model object
    """

    if auto_optimize:
        # cross validate to find optimal number of components
        print('Optimizing number of components for PLS-DA model')
        _, _, pls_da = CV_optimize_n_components(X_train, y_train, max_components=20, cv=10, njobs=-1, plot_bool=plot_bool)
    else:
        pls_da = multiclass_plsda(X_train, y_train, n_components=10)

    return pls_da

def CV_optimize_n_components(X_train, y_train, max_components, cv=10, njobs=-1, plot_bool=False):
    """
    Applies cross-validation to find the optimal number of components for the PLS-DA model.
    Returns the F1 scores for each number of components and the optimal number of components.

    :param X_train: training data
    :param y_train: training labels
    :param max_components: maximum number of components to test
    :param cv: number of cross-validation folds
    :param njobs: number of parallel jobs (default = -1)
    :param plot_bool: boolean to plot the F1 score vs number of components
    """

    n_components_list = list(range(2, max_components))

    # TODO: Check this, it is not used in the function
    # Create a custom scorer object using make_scorer
    PLSDA_multiclass_scorer = make_scorer(f1_weighted_scorer, greater_is_better=True)

    f1_scores = []
    expained_variance_CV = []
    for i in n_components_list:
        pls_da = multiclass_plsda(X_train, y_train, i)  # make the model object
        scores = cross_val_score(pls_da, X_train, y_train, cv=cv, scoring=f1_weighted_scorer, verbose=0, n_jobs=njobs)

        f1_scores.append(scores.mean())

    # plot the f1 scores against the number of components
    # fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    # ax.plot(n_components_list, f1_scores, marker='o')
    # ax.set_title('F1 score vs number of components')
    # ax.set_xlabel('Number of components')
    # ax.set_ylabel('F1 score')
    # plt.show()

    optimal_n_components, cooresponding_F1score = find_optimal_ncomp_via_saturation_point(n_components_list, f1_scores, plot_bool=plot_bool)
    print(f'Optimal number of components: {optimal_n_components}, from saturation point analysis with F1 score: {max(f1_scores):.2f}')

    # build the final model with the optimal number of components
    pls_da = multiclass_plsda(X_train, y_train, optimal_n_components)

    return f1_scores, optimal_n_components, pls_da


def get_VIP(pls_da_model):
    """
    Extract Variable Importance in Projection (VIP) scores (aka feature importance) from the PLS-DA model.
    Can help to understand model results and identify important features.

    :param pls_da_model: PLS-DA model object

    :return: VIP scores for each feature (here Band)
    """
    # print(type(pls_da_model), 'in vip')
    # TODO: cite reference for VIP calculation
    T = pls_da_model.x_scores_    # Scores -> new coordinates in PLS space
    W = pls_da_model.x_weights_   # Weights -> impact of features on latent variables
    P = pls_da_model.x_loadings_  # X loadings -> impact of features on scores
    Q = pls_da_model.y_loadings_  # Y loadings -> impact of targets on scores

    # Calculate the sum of squares of the scores, weighted by the Y loadings aka.
    SSQ = np.sum(T ** 2, axis=0) * np.sum(Q ** 2, axis=0)
    total_variance = np.sum(SSQ)

    # Calculate the contribution of each variable to the components
    weights_squared = W ** 2 # TODO: remove? because square is taken below
    contribution = np.dot(P ** 2, (SSQ / total_variance))

    # VIP scores for each variable
    VIP = np.sqrt(pls_da_model.n_features_in_ * contribution)

    # subset and oder descending VIP
    VIP_df = pd.DataFrame(VIP, columns=['VIP'])
    VIP_df['Band_ID'] = range(1, VIP.shape[0] + 1)

    return VIP_df

def predict_on_image(hyperspectral_image, pls_da_model):
    """
    Applies trained PLS-DA model to a hyperspectral image for prediction.
    Returns a predicted mask image.
    """
    # predict on image
    # reshape image to table
    hyperspectral_2D = reshape_image_to_table(hyperspectral_image)
    # predict on image
    predicted_mask = pls_da_predict(pls_da_model, hyperspectral_2D)
    # reshape predicted mask to image shape
    predicted_mask_image = predicted_mask.reshape(hyperspectral_image[0].shape)
    return predicted_mask_image

def get_validation_report(X_test, y_test, pls_da_model, format=False):
    """
    Generates a sklearn validation report/F1 Score for the final PLS-DA model.
    Just for reporting purposes.
    Formats output for collection and comparison.
    """
    # get validation report
    y_pred = pls_da_predict(pls_da_model, X_test)
    f1_score_rd = np.round(f1_score(y_test, y_pred, average='weighted'), 2)
    print('internal F1 score: ', f1_score_rd)

    if format:
        report = classification_report(y_test, y_pred, output_dict=True)
        report_df = pd.DataFrame(report).transpose()
        #report_df = report_df.iloc[[0,1,2,5]]
        return report_df
    else:
        return None


# wrapper
def train_PLSDA(hs_image_obj, mask_obj, max_components=20, cv=10, njobs=-1):
    """
    Wrapper function to preprocess the data and train a PLS-DA model for multiclass classification.
    """

    # TODO: Is not used! similar code in Main. Solve this! remove one of them

    # reshape image to table
    hyperspectral_2D = reshape_image_to_table(hs_image_obj.image)

    # get pixel labels
    labeled_pixels, labels = get_pixellabels(mask_obj.multiclass_mask, hyperspectral_2D)

    # balance classes
    balanced_pixels, balanced_labels = balance_classes(labeled_pixels, labels)

    # remove outliers
    balanced_pixels, balanced_labels = outlier_removal(balanced_pixels, balanced_labels)

    # split data
    X_train, X_test, y_train, y_test = split_data(balanced_pixels, balanced_labels)

    # cross validate to find optimal number of components
    f1_scores, optimal_n_components, pls_da = CV_optimize_n_components(X_train, y_train, max_components, cv=cv, njobs=njobs) # TODO: is the triple return even needed anymore?
    print(type(pls_da), 'before vip')
    # get VIP scores
    VIP_df = get_VIP(pls_da)

    # get validation report
    validation_report(X_test, y_test, pls_da)

    return pls_da, VIP_df


# TODO: whats this for? delete?
## build a vector with all function names of this file
#classification_functions = [reshape_image_to_table, get_pixellabels, outlier_removal, balance_classes, split_data,
#                            multiclass_plsda, pls_da_predict, f1_weighted_scorer, CV_optimize_n_components,
#                            get_VIP, validation_report, predict_on_image, train_PLSDA]
