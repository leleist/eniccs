#test












# find undetected clouds and update cloud mask
def find_undetected_clouds(spectral_image_obj, mask_obj, cloud_threshold=1.5):
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

    band_75 = band_75 * 0.0001
    band_153 = band_153 * 0.0001
    band_6 = band_6 * 0.0001
    band_28 = band_28 * 0.0001
    band_47 = band_47 * 0.0001

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
    # CI = (band_75 + 2* band_153) / (band_6 + band_28 + band_47)
    # Suppress division by zero warnings
    with np.errstate(divide='ignore', invalid='ignore'):
        CI = (band_75 + 2 * band_153) / (band_6 + band_28 + band_47)
        CI[(band_6 + band_28 + band_47) == 0] = np.nan  # Set CI to NaN where denominator is zero

    #plot CI
    plt.imshow(CI)
    plt.colorbar()
    plt.show()

    # apply threshold to CI
    CI_threshold = cloud_threshold # 0.5  # small value from (0.01, 0.1, 1, 10, 100) as in Zhai et al. 2018
    CI_binary = np.zeros(spectral_image.shape[1:])
    CI_binary[np.abs(CI) < CI_threshold] = 1  # 1 equals cloud

    #plot CI_binary
    plt.imshow(CI_binary)
    plt.colorbar()
    plt.show()


    # update cloud mask in masklist
    masklist[0] = np.where(CI_binary == 1, 1, masklist[0])

    mask_obj.maskdata = masklist

    return mask_obj


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
