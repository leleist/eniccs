"""
The main module orchestrates the EnICCS pipeline for improving EnMAP's cloud and cloud shadow masks.

it contains functions that utilize both the HsImage and Mask classes to refine existing masks,
train a PLS-DA classification model, classify the hyperspectral image, and postprocess the results
to generate final cloud and cloud shadow masks.

Functions
---------
- improve_cloud_mask_over_land: Refines the cloud mask using Cloud Index over land areas.
- improve_cloud_shadow_mask: Refines the cloud shadow mask using spectral indices.
- run_eniccs: Main wrapper function that executes the entire EnICCS pipeline.
- refine_ccs_masks: Wrapper for refining cloud and cloud shadow masks.
- classify_image: Wrapper for training the PLS-DA model and classifying the image.

For detailed information please refer to the readme and accompanying publication.

This module serves as the first access point for anyone intending to adapt or extend the EnICCS
pipeline to surface types beyond tropical western kenya.
Here changes can be made to existing band indices and thresholds, or new ones can be added.
"""

import numpy as np
from .classification import (reshape_image_to_table, get_pixellabels, balance_classes,
                             outlier_removal, split_data, PLSDA_model_builder, get_vip,
                             get_validation_report,
                             predict_on_image)
from .masks import Mask
from .hs_image import HsImage

# find undetected clouds and update cloud mask
def improve_cloud_mask_over_land(
        spectral_image_obj: 'HsImage',
        mask_obj: 'Mask'
) -> None:
    """
    Improve cloud detection over land using Cloud Index.

    This function extends the original cloud mask by applying the  Cloud
    Index (CI) from Zhai et al. 2018 (ISPRS) and the Cloud over Land Test (CLT)
    to detect previously unidentified (small) clouds over land.

    Parameters
    ----------
    spectral_image_obj : HsImage
        EnICCS hyperspectral image object
    mask_obj : Mask
        EnICCS mask object

    Returns
    -------
    None
        Function modifies the mask_obj.mask_data in-place.
    """
    spectral_image = spectral_image_obj.image
    no_data = spectral_image_obj.no_data_value
    masklist = mask_obj.mask_data

    no_data_condition = spectral_image[0, :, :] == no_data

    band_75 = spectral_image[74, :, :]  # 860 nm, NIR
    band_153 = spectral_image[152, :, :]  # 1653 nm, SWIR1
    band_6 = spectral_image[5, :, :]  # 444 nm, Blue
    band_28 = spectral_image[27, :, :]  # 549 nm, Green
    band_47 = spectral_image[46, :, :]  # 659 nm, Red

    band_75 = band_75 * 0.0001 + 0
    band_153 = band_153 * 0.0001 + 0
    band_6 = band_6 * 0.0001 + 0
    band_28 = band_28 * 0.0001 + 0
    band_47 = band_47 * 0.0001 + 0

    band_75[no_data_condition] = np.nan
    band_153[no_data_condition] = np.nan
    band_6[no_data_condition] = np.nan
    band_28[no_data_condition] = np.nan
    band_47[no_data_condition] = np.nan

    band_75[band_75 > 1] = 1
    band_153[band_153 > 1] = 1
    band_6[band_6 > 1] = 1
    band_28[band_28 > 1] = 1
    band_47[band_47 > 1] = 1

    band_75[band_75 < 0] = 0
    band_153[band_153 < 0] = 0
    band_6[band_6 < 0] = 0
    band_28[band_28 < 0] = 0
    band_47[band_47 < 0] = 0

    # calculate universal CloudIndex (CI) after Zhai et al. 2018, ISPRS
    # catching potential divide by zero error. Nas are handeled accordingly downstream
    with np.errstate(divide='ignore',
                     invalid='ignore'):

        ci = (band_75 + 2 * band_153) / (band_6 + band_28 + band_47)

    # apply threshold to CI
    ci_threshold = 1  # small value from (0.01, 0.1, 1, 10, 100) as in Zhai et al. 2018

    # Cloud_over_Land_Test
    clt_mask = np.zeros(spectral_image.shape[1:])
    clt_mask[(band_6 >= 0.15) & (band_75 >= 0.25) & (ci >= ci_threshold) & (ci <= 2)] = 1

    # update cloud mask in masklist
    masklist[3] = np.where(clt_mask == 1, 1, masklist[3])

    mask_obj.maskdata = masklist


def improve_cloud_shadow_mask(
        spectral_image_obj: 'HsImage',
        mask_obj: 'Mask'
) -> None:
    """
    Improve cloud shadow detection using spectral indices and thresholding.

    This function extends and refines the original cloud shadow mask by applying
    a difference index (DI) based on NIR and Red bands, combined with water-land
    discrimination to reduce false positives over water bodies.

    Parameters
    ----------
    spectral_image_obj : HsImage
        EnICCS hyperspectral image object.
    mask_obj : Mask
        EnICCS mask object containing the original cloud shadow mask.

    Returns
    -------
    None
        Function modifies the mask_obj.mask_data in-place.
    """

    spectral_image = spectral_image_obj.image
    no_data = spectral_image_obj.no_data_value
    masklist = mask_obj.mask_data

    no_data_condition = spectral_image[0, :, :] == no_data

    band_108 = spectral_image[107, :, :]  # 1070 nm, NIR
    band_45 = spectral_image[44, :, :]  # 641 nm, Red
    band_29 = spectral_image[28, :, :]  # 550 nm, Green

    band_108 = band_108 * 0.0001
    band_45 = band_45 * 0.0001
    band_29 = band_29 * 0.0001

    band_108[no_data_condition] = np.nan
    band_45[no_data_condition] = np.nan
    band_29[no_data_condition] = np.nan

    band_108[band_108 > 1] = 1
    band_45[band_45 > 1] = 1
    band_29[band_29 > 1] = 1

    band_108[band_108 < 0] = 0
    band_45[band_45 < 0] = 0
    band_29[band_29 < 0] = 0


    # can be simplified to:  (1,  𝑖𝑓 0≤𝐶𝐼≤0.3 𝑎𝑛𝑑 𝑅𝑒𝑑≤0.8∗𝐺𝑟𝑒𝑒𝑛, 0 else)

    # calculate difference index (2*NIR - Red) to increase the distance between cloud and shadow.
    di = 2 * band_108 - band_45

    # create a binary mask of DI where values between 0.015 and 0.03 are set to 1
    di_binary = np.zeros(spectral_image.shape[1:])
    di_binary[(di >= 0) & (di <= 0.3)] = 1

    extended_cloudshadow = np.where(di_binary == 1, 1, 0).astype(np.uint8)

    water_land_mask = (
                band_45 > 0.8 * band_29)
    # exploits logic of green hump in veg. spectrum. scaling down green allows to group up light
    # surfaces such as soil with vegetation while still being able to discriminate from water.

    extended_cloudshadow = np.where(water_land_mask == 1, 0, extended_cloudshadow)
    extended_cloudshadow = np.expand_dims(extended_cloudshadow, axis=0)

    # update cloud mask in masklist
    masklist[4] = extended_cloudshadow

    mask_obj.mask_data = masklist


# overall wrapper
def run_eniccs(
        dir_path: str,
        save_output: bool = True,
        return_mask_obj: bool = False,
        auto_optimize: bool = False,
        verbose: bool = False,
        plot: bool = False,
        smooth_output: bool = True,
        contamination: float = 0.25,
        percentile: int = 85,
        num_samples: int = 3000,
        n_jobs: int = -1,
):
    """
    Main wrapper for the EnICCS pipeline for improving EnMAP's operational
    cloud and cloud shadow masks.

    Loads hyperspectral image and masks (cloud, cloud shadow, land, water, nodata),
    refines them, trains a PLS-DA model, classifies CCS, and postprocesses the masks with
    physical constraints.

    Parameters
    ----------
    dir_path : str
        Path to the directory containing the geotiffs as provided by the data provider.
        Naming must follow operatinal convention as stated in DLR EnMAP documentation.
    save_output : bool, default=True
        Whether to save the output masks to file as two new files.
    return_mask_obj : bool, default=False
        Whether to return the final EnICCS mask object.
    auto_optimize : bool, default=False
        Whether to optimize the number of LVs for the PLS-DA model automatically with F1-score
        saturation curve fitting.
        If False, uses 10 LVs.
    verbose : bool, default=False
        Whether to print progress information during processing.
    plot : bool, default=False
        Whether to generate diagnostic plots during processing.
         - PLS-DA latent variable x F1-score curve (if auto_optimize=True)
    smooth_output : bool, default=True
        Whether to apply smoothing to the output masks.
        This is done with image morphological operations, configured to retain initial mask area.
        It is generally recommended to cover C/CS corners/edges/Indentations which are high
        uncertainty areas.
    contamination : float, default=0.25
        Proportion of outliers in the dataset for outlier detection with sklearn LOF.
        Must be in the range (0, 0.5].
    percentile : int, default=80
        Percentile threshold used for detecting missclassified cloud shadows due to unusual
        cloud to shadow distance.
        High percentile = accepting more FPs, low percentile = accepting more FNs.
        This depends on the scenes cloud formation processes, types and structure.
        The diagnostic plot can help
    num_samples : int, default=3000
        Number of samples to use for training the classification model.
        the default (3000) is conservative. low values first deteriorate model performance,
        for cloud shadows due to higher spectral variability.
    n_jobs : int, default=-1
        Number of parallel jobs to run. -1 --> all processors.

    Returns
    -------
    mask_obj : Mask or None
        The updated cloud and cloud shadow masks within a mask-class object if `return_mask_obj`
        is True, otherwise None.

    Notes
    -----
    This pipeline was developed and optimized for densely vegetated surfaces of
    tropical western Kenya. If your AOI has differing surface characteristics, changes may be
    necessary to yield satisfactory results.

    First option: Try to adjust the pipeline parameters within this function.
    Second option: Adapt mask refinement functions
    (spectral indices and/or thresholds to your target area)

    For details, we refer to the accompanying publication and the readme.
    """

    # load hyperspectral image
    spectral_image_obj = HsImage(dir_path)

    # load masks
    mask_obj = Mask(dir_path, num_samples=num_samples)

    # refine cloud and cloud shadow masks
    refine_ccs_masks(spectral_image_obj, mask_obj)

    # classify image
    mask_obj, vip_df = classify_image(
        spectral_image_obj,
        mask_obj,
        auto_optimize=auto_optimize,
        verbose=verbose,
        num_samples=num_samples,
        contamination=contamination,
        percentile=percentile,
        n_jobs=n_jobs,
        smooth_output=smooth_output,
        plot=plot
    )

    if save_output:
        filename_cloud = mask_obj.datatake_name + '_EnICCS_CLOUD'
        filename_cloudshadow = mask_obj.datatake_name + '_EnICCS_CLOUDSHADOW'

        mask_obj.save_mask_to_geotiff(
            mask_obj.new_cloud_mask,
            filename_prefix=filename_cloud
        )
        mask_obj.save_mask_to_geotiff(
            mask_obj.new_cloudshadow_mask,
            filename_prefix=filename_cloudshadow
        )

    return mask_obj if return_mask_obj else None


# mask refinement and classification preparation wrapper
def refine_ccs_masks(spectral_image_obj, mask_obj):
    """
    This function is a wrapper for refining the cloud and cloud shadow masks to improve the
    suitability for ML training.
    Cloud and cloud sadow masks are refined with spectral indices and thresholding.

    Note:
    This is the main access point for customization to surfaces with
    differing spectral characteristics.

    Parameters
    ----------
    spectral_image_obj : HsImage
        EnICCS Hyperspectral image object.
    mask_obj : Mask
        EnICCS mask object.

    Returns
    -------
    None
        Function modifies the mask_obj in-place.
    """

    # print('Refining cloud and cloud shadow masks with spectral indices')
    # improve cloud mask over land
    improve_cloud_mask_over_land(spectral_image_obj, mask_obj)

    # improve cloud shadow mask
    improve_cloud_shadow_mask(spectral_image_obj, mask_obj)

    # combine masks
    mask_obj.combine_masks()

    # buffer water mask to remove areas of high uncertainty
    mask_obj.buffer_water_mask(buffer_size=3)

    # format mask for classification
    mask_obj.format_mask_for_classification()


# classification wrapper
def classify_image(
        spectral_image_obj: 'HsImage',
        mask_obj: 'Mask',
        num_samples: int,
        percentile: int,
        contamination: float = None,
        auto_optimize: bool = False,
        verbose: bool = False,
        plot: bool = False,
        smooth_output: bool = True,
        n_jobs: int = -1,
):
    """
    Classify hyperspectral image using PLS-DA model for cloud and cloud shadow detection.

    This function trains a Partial Least Squares Discriminant Analysis (PLS-DA) model
    using refined cloud and cloud shadow masks, applies the model the
     HSimage, and postprocesses the results to generate final CCS masks.

    Parameters
    ----------
    spectral_image_obj : HsImage
        EnICCS hyperspectral image object.
    mask_obj : Mask
        EnICCS mask object with refined cloud and cloud shadow masks for training.
    num_samples : int
        Number of samples to use for training the classification model.
    percentile : int
        Percentile threshold used for cloud to shadow centroid distance based object selection.
    contamination : float, optional
        Proportion of outliers to remove during preprocessing with sklearn LOF.
    auto_optimize : bool, default=False
        Whether to optimize the number of components for the PLS-DA model automatically.
        If False, uses 10 components.
    verbose : bool, default=False
        Whether to print progress information.
    plot : bool, default=False
        Whether to generate diagnostic plots during processing.
    smooth_output : bool, default=True
        Whether to apply smoothing during postprocessing of predictions. Generally recommended.
    n_jobs : int, default=-1
        Number of parallel jobs to run. -1 --> all processors.

    Returns
    -------
    mask_obj : Mask
        Updated EnICCS mask object with new cloud and cloud shadow masks.
    vip_df : DataFrame
        Variable Importance in Projection (VIP) scores i.e. feature importance from the PLS-DA model.

    """

    # reshape image to table
    hyperspectral_2d = reshape_image_to_table(spectral_image_obj.image)

    # get pixel labels
    labeled_pixels, labels = get_pixellabels(mask_obj.classification_mask, hyperspectral_2d)

    # balance classes
    balanced_pixels, balanced_labels = balance_classes(labeled_pixels,
                                                       labels,
                                                       num_samples=num_samples)

    # remove outliers
    balanced_pixels, balanced_labels = outlier_removal(balanced_pixels,
                                                       balanced_labels,
                                                       contamination=contamination)

    # split data
    X_train, X_test, y_train, y_test = split_data(balanced_pixels, balanced_labels)

    pls_da = PLSDA_model_builder(X_train, y_train, auto_optimize=auto_optimize,
                                 verbose=verbose,
                                 plot=plot,
                                 n_jobs=n_jobs)

    # get VIP scores
    vip_df = get_vip(pls_da)
    mask_obj.vip_scores = vip_df

    # get validation report
    val_report = get_validation_report(X_test, y_test, pls_da, format=True, verbose=verbose)

    # attach to mask object
    mask_obj.validation_report = val_report

    # predict on image
    mask_obj.predicted_mask = predict_on_image(spectral_image_obj.image, pls_da)
    mask_obj.predicted_mask = np.where(mask_obj.nodata_mask == 1, 0, mask_obj.predicted_mask)

    # extract cloud and cloud shadow mask as binary masks
    mask_obj.format_predicted_masks_to_binary()

    # update water mask to remove falsely classified water pixels in native water mask
    mask_obj.resolve_cs_water_confusion()

    mask_obj.modify_cloud_shadows_based_on_centroid_distance(percentile=percentile,
                                                             verbose=verbose,
                                                             plot=plot)  # 75

    # postprocess cloudshadow to remove missclassifications along water bodies
    mask_obj.reset_cs_coastal_pixels()

    # postprocess predictions
    mask_obj.new_cloud_mask = mask_obj.prediction_postprocessing(mask_obj.new_cloud_mask,
                                                                 structure_size=3,
                                                                 buffer_size=1,
                                                                 neutral_smooth=smooth_output)

    mask_obj.new_cloudshadow_mask = mask_obj.prediction_postprocessing(mask_obj.new_cloudshadow_mask,
                                                                       structure_size=3,
                                                                       buffer_size=1,
                                                                       neutral_smooth=smooth_output)
    # reconciling cloud and cloud shadow masks, favoring cloud mask
    mask_obj.new_cloudshadow_mask[mask_obj.new_cloud_mask == 1] = 0

    mask_obj.reapply_nodata_mask()

    return mask_obj, vip_df
