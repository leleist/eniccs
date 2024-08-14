from .classification import (reshape_image_to_table, get_pixellabels, balance_classes, outlier_removal,
                             split_data, CV_optimize_n_components, get_VIP, validation_report, predict_on_image)
from .masks import mask
from .hs_image import hs_image

import numpy as np

# find undetected clouds and update cloud mask
def improve_cloud_mask_over_land(spectral_image_obj, mask_obj):
    spectral_image = spectral_image_obj.image
    no_data = spectral_image_obj.no_data_value
    masklist = mask_obj.mask_data

    # plt.imshow(masklist[3][0, :, :], cmap='gray')
    # plt.title('CI')
    # plt.colorbar()
    # plt.show()

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
    CI = (band_75 + 2 * band_153) / (band_6 + band_28 + band_47)

    # apply threshold to CI
    CI_threshold = 1  # small value from (0.01, 0.1, 1, 10, 100) as in Zhai et al. 2018
    CI_binary = np.zeros(spectral_image.shape[1:])
    CI_binary[np.abs(CI) < CI_threshold] = 1  # 1 equals cloud

    # Cloud_over_Land_Test
    CLT_mask = np.zeros(spectral_image.shape[1:])
    CLT_mask[(band_6 >= 0.15) & (band_75 >= 0.25) & (CI >= CI_threshold) & (CI <= 2)] = 1



    # update cloud mask in masklist
    masklist[3] = np.where(CLT_mask == 1, 1, masklist[3])

    # print CI
    # plt.imshow(CLT_mask, cmap='gray')
    # plt.title('CI')
    # plt.colorbar()
    # plt.show()
#
    # plt.imshow(masklist[3][0, :, :], cmap='gray')
    # plt.title('CI')
    # plt.colorbar()
    # plt.show()

    mask_obj.maskdata = masklist



def improve_cloud_shadow_mask(spectral_image_obj, mask_obj):
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
    DI = band_108 - band_45 + band_108

    # plot DI
    #plt.imshow(DI)
    #plt.colorbar()
    #plt.show()


    # create a binary mask of DI where values between 0.015 and 0.03 are set to 1
    DI_binary = np.zeros(spectral_image.shape[1:])
    DI_binary[(DI >= 0) & (DI <= 0.3)] = 1

    # extend Cloud shadow with binary DI
    original_cloudshadow = masklist[6]
    extended_cloudshadow = np.where(DI_binary == 1, 1, original_cloudshadow).astype(np.uint8)

    #plt.imshow(original_cloudshadow[0, :, :])
    #plt.colorbar()
    #plt.show()


    # remove water pixels mistakenly classified as cloud shadow
    # mask_obj.apply_binary_opening(mask_index=2, structure_size=2)
    # extended_cloudshadow = binary_erosion(extended_cloudshadow, iterations=1)

    water_land_mask = (band_45 > 0.8*band_29) # .astype(float) # exploits logic of green hump in veg. spectrum
    extended_cloudshadow = np.where(water_land_mask == 1, 0, extended_cloudshadow)


    # plot water_land_binary_mask
    # plt.figure(figsize=(10, 10))
    # plt.imshow(water_land_mask)
    # plt.title('water_land_mask')
    # plt.colorbar()
    # plt.show()
#
    # plt.imshow(extended_cloudshadow[0, :, :])
    # plt.colorbar()
    # plt.show()


    # update cloud mask in masklist
    masklist[6] = extended_cloudshadow
    mask_obj.mask_data = masklist

    # plot DI
    #plt.imshow(original_cloudshadow[0, :, :])
    #plt.colorbar()
    #plt.show()

    #plt.imshow(extended_cloudshadow[0, :, :])
    #plt.colorbar()
    #plt.show()
    print("water_land_mask shape: ", water_land_mask.shape)
    print("DI_binary shape: ", DI_binary.shape)
    print("extended_cloudshadow shape: ", extended_cloudshadow.shape)
    print("original_cloudshadow shape: ", original_cloudshadow.shape)


    return water_land_mask, DI_binary, extended_cloudshadow, original_cloudshadow

    # buffer water mask to exclude coastal areas due to high missclassification rate in original data

# overall wrapper

def eniccs(dir_path, save_output=True):
    # load hyperspectral image
    spectral_image_obj = hs_image(dir_path)

    # load masks
    mask_obj = mask(dir_path)

    # refine cloud and cloud shadow masks
    refine_ccs_masks(spectral_image_obj, mask_obj)

    # classify image
    mask_obj = classify_image(spectral_image_obj, mask_obj)

    # TODO: report accuracies etc. and write to file.

    if save_output:
        mask_obj.save_mask_to_geotiff(mask_obj.new_cloud_mask, filename_prefix="NEW_CLOUD_V2")
        mask_obj.save_mask_to_geotiff(mask_obj.new_cloudshadow_mask, filename_prefix="NEW_CLOUDSHADOW_V2")

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

def classify_image(spectral_image_obj, mask_obj):
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

    # cross validate to find optimal number of components
    f1_scores, optimal_n_components, pls_da = CV_optimize_n_components(X_train, y_train, max_components=20,
                                                                              cv=10, njobs=-1)
    # get VIP scores
    VIP_df = get_VIP(pls_da)

    # get validation report
    validation_report(X_test, y_test, pls_da)

    print("Predicting on image...")

    # predict on image
    mask_obj.predicted_mask = predict_on_image(spectral_image_obj, pls_da)

    # extract cloud and cloud shadow mask as binary masks
    mask_obj.format_predicted_mask_to_binary()



    # postprocess prediction
    mask_obj.prediction_postprocessing(mask_obj.new_cloud_mask, structure_size=3, buffer_size=2)
    mask_obj.prediction_postprocessing(mask_obj.new_cloudshadow_mask, structure_size=3, buffer_size=2)

    print("Done")

    return mask_obj

