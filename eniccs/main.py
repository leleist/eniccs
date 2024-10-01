import numpy as np
from .classification import (reshape_image_to_table, get_pixellabels, balance_classes, outlier_removal,
                             split_data, PLSDA_model_builder, get_VIP, validation_report, predict_on_image)
from .masks import Mask
from .hs_image import HsImage



# find undetected clouds and update cloud mask
def improve_cloud_mask_over_land(spectral_image_obj, mask_obj):
    """ This function extends the original cloud mask via the  universal CloudIndex (CI) after Zhai et al. 2018, ISPRS
    and the Cloud over Land Test (CLT). Undetected small clouds over land are detected and added to the cloud mask.

    spectral_image_obj: eniccs HsImage object
    mask_obj: eniccs Mask object
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
    with np.errstate(divide='ignore', invalid='ignore'): # catching potential divide by zero error. Nas are handeled accordingly downstream
        ci = (band_75 + 2 * band_153) / (band_6 + band_28 + band_47)

    # apply threshold to CI
    ci_threshold = 1  # small value from (0.01, 0.1, 1, 10, 100) as in Zhai et al. 2018
    ci_binary = np.zeros(spectral_image.shape[1:])
    ci_binary[np.abs(ci) < ci_threshold] = 1

    # Cloud_over_Land_Test
    clt_mask = np.zeros(spectral_image.shape[1:])
    clt_mask[(band_6 >= 0.15) & (band_75 >= 0.25) & (ci >= ci_threshold) & (ci <= 2)] = 1

    # update cloud mask in masklist
    masklist[3] = np.where(clt_mask == 1, 1, masklist[3])

    mask_obj.maskdata = masklist



def improve_cloud_shadow_mask(spectral_image_obj, mask_obj):
    """ This function extends the original cloud shadow mask. """
    spectral_image = spectral_image_obj.image
    no_data = spectral_image_obj.no_data_value
    masklist = mask_obj.mask_data


    no_data_condition = spectral_image[0, :, :] == no_data

    band_108 = spectral_image[107, :, :]  # 1070 nm, NIR
    band_45 = spectral_image[44, :, :]  # 641 nm, Red
    band_3 = spectral_image[2, :, :]  # 641 nm, Blue
    band_29 = spectral_image[28, :, :]  # 550 nm, Green

    band_108 = band_108 * 0.0001
    band_45  = band_45  * 0.0001
    band_3 = band_3 * 0.0001
    band_29 = band_29 * 0.0001

    band_108[no_data_condition] = np.nan
    band_45[no_data_condition] = np.nan
    band_3[no_data_condition] = np.nan
    band_29[no_data_condition] = np.nan

    band_108[band_108 > 1] = 1
    band_45[band_45 > 1] = 1
    band_3[band_3 > 1] = 1
    band_29[band_29 > 1] = 1

    band_108[band_108 < 0] = 0
    band_45[band_45 < 0] = 0
    band_3[band_3 < 0] = 0
    band_29[band_29 < 0] = 0


    # calculate difference index
    di = band_108 - band_45 + band_108

    # create a binary mask of DI where values between 0.015 and 0.03 are set to 1
    di_binary = np.zeros(spectral_image.shape[1:])
    di_binary[(di >= 0) & (di <= 0.3)] = 1

    # extend Cloud shadow with binary DI
    original_cloudshadow = masklist[6]
    extended_cloudshadow = np.where(di_binary == 1, 1, original_cloudshadow).astype(np.uint8)

    # remove water pixels mistakenly classified as cloud shadow
    # mask_obj.apply_binary_opening(mask_index=2, structure_size=2)
    # extended_cloudshadow = binary_erosion(extended_cloudshadow, iterations=1)

    water_land_mask = (band_45 > 0.8*band_29) # .astype(float) # exploits logic of green hump in veg. spectrum
    extended_cloudshadow = np.where(water_land_mask == 1, 0, extended_cloudshadow)

    # update cloud mask in masklist
    masklist[6] = extended_cloudshadow
    mask_obj.mask_data = masklist

    return water_land_mask, di_binary, extended_cloudshadow, original_cloudshadow # TODO: remove return values?

    # buffer water mask to exclude coastal areas due to high missclassification rate in original data

# overall wrapper

def eniccs(dir_path, save_output=True, auto_optimize=False, plot_bool=False, return_mask_obj=False):
    """ This function is the main wrapper for the ENICCS pipeline. It loads the hyperspectral image and masks, refines them, trains a PLS-DA model and classifies the image.
    after postprocessing (smoothing) the results are saved as geotiffs.
    dirpath: str, path to the directory containing the geotiffs as provided by the data provider
    save_output: bool, if True the output masks are saved to file
    auto_optimize: bool, if True the number of components for the PLS-DA model is optimized. If False, the number of components is set to 10.

    """
    # load hyperspectral image
    spectral_image_obj = HsImage(dir_path)

    # load masks
    mask_obj = Mask(dir_path)

    # refine cloud and cloud shadow masks
    refine_ccs_masks(spectral_image_obj, mask_obj)

    # classify image
    mask_obj = classify_image(spectral_image_obj, mask_obj, auto_optimize=auto_optimize, plot_bool=False)

    if save_output:
        filename_Cloud = mask_obj.datatake_name + "_EnICCS_CLOUD.tif"
        filename_CloudShadow = mask_obj.datatake_name + "_EnICCS_CLOUDSHADOW.tif"

        mask_obj.save_mask_to_geotiff(mask_obj.new_cloud_mask, filename_prefix=filename_Cloud)
        mask_obj.save_mask_to_geotiff(mask_obj.new_cloudshadow_mask, filename_prefix=filename_CloudShadow)

    if return_mask_obj:
        return mask_obj



# mask refinement and classification preparation wrapper
def refine_ccs_masks(spectral_image_obj, mask_obj):
    print("Refining cloud and cloud shadow masks with spectral indices")
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

    # save mask to geotiff
    mask_obj.save_mask_to_geotiff(mask_obj.classification_mask, filename_prefix="CLASSIFICATION_MASK")



# classification wrapper

def classify_image(spectral_image_obj, mask_obj, auto_optimize=False, plot_bool=False):
    print("Training PLS-DA model with refined cloud and cloud shadow masks")

    # reshape image to table
    hyperspectral_2D = reshape_image_to_table(spectral_image_obj.image)

    # get pixel labels
    labeled_pixels, labels = get_pixellabels(mask_obj.classification_mask, hyperspectral_2D)

    # balance classes
    balanced_pixels, balanced_labels = balance_classes(labeled_pixels, labels)

    # remove outliers
    balanced_pixels, balanced_labels = outlier_removal(balanced_pixels, balanced_labels)

    # split data
    X_train, X_test, y_train, y_test = split_data(balanced_pixels, balanced_labels)

    pls_da = PLSDA_model_builder(X_train, y_train, auto_optimize=auto_optimize, plot_bool=plot_bool)

    # get VIP scores
    VIP_df = get_VIP(pls_da) # TODO: Not used currently check notes

    # get validation report
    validation_report(X_test, y_test, pls_da)

    print("Predicting on image...")

    # predict on image
    mask_obj.predicted_mask = predict_on_image(spectral_image_obj.image, pls_da)

    # extract cloud and cloud shadow mask as binary masks
    mask_obj.format_predicted_mask_to_binary()

    # postprocess predictions
    mask_obj.prediction_postprocessing(mask_obj.new_cloud_mask, structure_size=4, buffer_size=2)
    mask_obj.prediction_postprocessing(mask_obj.new_cloudshadow_mask, structure_size=4, buffer_size=2)

    # postprocess cloudshadow to remove missclassifications
    mask_obj.reset_cs_coastal_pixels()

    mask_obj._modify_cloud_shadows_based_on_centroid_distance(percentile=75, plot_bool=plot_bool)

    # postprocess predictions
    mask_obj.prediction_postprocessing(mask_obj.new_cloud_mask, structure_size=4, buffer_size=2)
    mask_obj.prediction_postprocessing(mask_obj.new_cloudshadow_mask, structure_size=4, buffer_size=2)

    print("Done")

    return mask_obj
