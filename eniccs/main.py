#test









# find undetected clouds and update cloud mask
def find_cloud_over_land(spectral_image_obj, mask_obj):
    import numpy as np
    import matplotlib.pyplot as plt

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
    CI = (band_75 + 2 * band_153) / (band_6 + band_28 + band_47)

    # print CI
    plt.imshow(CI, cmap='gray')
    plt.title('CI')
    plt.colorbar()
    plt.show()


    # apply threshold to CI
    CI_threshold = 1  # small value from (0.01, 0.1, 1, 10, 100) as in Zhai et al. 2018
    CI_binary = np.zeros(spectral_image.shape[1:])
    CI_binary[np.abs(CI) < CI_threshold] = 1  # 1 equals cloud

    # Cloud_over_Land_Test
    CLT_mask = np.zeros(spectral_image.shape[1:])
    CLT_mask[(band_6 >= 0.15) & (band_75 >= 0.25) & (CI >= 0.95) & (CI <= 2)] = 1

    # update cloud mask in masklist
    masklist[3] = np.where(CLT_mask == 1, 1, masklist[3])

    # print CI
    plt.imshow(CLT_mask, cmap='gray')
    plt.title('CI')
    plt.colorbar()
    plt.show()

    mask_obj.maskdata = masklist

    return mask_obj


def find_cloud_shadow(spectral_image_obj, mask_obj):
    import numpy as np
    import matplotlib.pyplot as plt

    spectral_image = spectral_image_obj.image
    no_data = spectral_image_obj.no_data_value
    masklist = mask_obj.mask_data


    no_data_condition = spectral_image[0, :, :] == no_data

    band_108 = spectral_image[107, :, :]  # 1070 nm, NIR
    band_45 = spectral_image[44, :, :]  # 641 nm, Red
    band_3 = spectral_image[2, :, :]  # 641 nm, Blue

    band_108 = band_108 * 0.0001
    band_45  = band_45  * 0.0001
    band_3 = band_3 * 0.0001

    band_108[no_data_condition] = np.nan
    band_45[no_data_condition] = np.nan
    band_3[no_data_condition] = np.nan

    band_108[band_108 > 1] = 1
    band_45[band_45 > 1] = 1
    band_3[band_3 > 1] = 1

    band_108[band_108 < 0] = 0
    band_45[band_45 < 0] = 0
    band_3[band_3 < 0] = 0


    # calculate difference index
    DI = band_108 - band_45 + band_108

    # create a binary mask of DI where values between 0.015 and 0.03 are set to 1
    DI_binary = np.zeros(spectral_image.shape[1:])
    DI_binary[(DI >= 0) & (DI <= 0.3)] = 1

    # extend Cloud shadow with binary DI
    original_cloudshadow = masklist[6]
    extended_cloudshadow = np.where(DI_binary == 1, 1, original_cloudshadow)

    # remove water pixels mistakenly classified as cloud shadow
    mask_obj.apply_binary_opening(mask_index=2, structure_size=2)

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


    # buffer water mask to exclude coastal areas due to high missclassification rate in original data
