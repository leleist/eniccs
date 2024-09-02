import rasterio
import glob
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import binary_dilation, binary_opening, binary_closing, binary_erosion

class mask:
    def __init__(self, dir_path, mask_RegEx=None):
        if mask_RegEx is None:
            self.mask_RegEx = dict(Classes="/*_CLASSES.TIF", Cloud="/*_CLOUD.TIF", Cirrus="/*CIRRUS.TIF",
                              Haze="/*HAZE.TIF", Cloud_shadow="/*CLOUDSHADOW.TIF")
        else:
            self.mask_RegEx = mask_RegEx
        self.multiclass_mask = mask
        self.dir_path = dir_path
        self.mask_regEx = mask_RegEx
        self.profile = None
        self.datatake_name = None
        self.mask_data = [] # Placeholder for loaded mask data
        self.multiclass_mask = None
        self.classification_mask = None
        self.predicted_mask = None
        self.new_cloud_mask = None
        self.new_cloudshadow_mask = None

        # load all masks and combine them into a multiclass mask upon initialization
        self.load_masks()


    # function to load and collect all masks into a list
    def load_masks(self):
        template_shape = None  # To hold the shape of the first successfully loaded mask

        def load_mask_or_placeholder(path_pattern, template_shape=None):
            paths = glob.glob(path_pattern)
            if paths:
                with rasterio.open(paths[0]) as src:
                    if template_shape is None:
                        # Save the shape of the first successfully loaded mask
                        template_shape = src.read(1).shape
                        self.transform = src.transform  # save Georeferencomg information for later use
                        self.profile = src.profile
                        self.datatake_name = os.path.basename(paths[0])[0:74]
                    return src.read(), template_shape
            else:
                if template_shape is None:
                    raise ValueError("First mask cannot be a placeholder, no reference shape available.")
                # Return an array of zeros with the same shape as the first mask
                print(f"Mask not found: {path_pattern} using placeholder")
                return np.zeros((1, *template_shape)), template_shape

        # Load each mask separately

        # land and water mask
        classesmask, template_shape = load_mask_or_placeholder(self.dir_path + self.mask_RegEx["Classes"])
        nodatamask = np.zeros(classesmask.shape)
        nodatamask[classesmask == 3] = 1  # set background class to 1
        self.mask_data.append(nodatamask)
        # plt.imshow(nodatamask[0, :, :])
        # plt.colorbar()
        # plt.show()

        landmask = np.zeros(classesmask.shape)
        landmask[classesmask == 1] = 1  # set land class to 1
        self.mask_data.append(landmask)

        watermask = np.zeros(classesmask.shape)
        watermask[classesmask == 2] = 1  # set water class to 1
        self.mask_data.append(watermask)


        # Cloud mask
        cloudmask, template_shape = load_mask_or_placeholder(self.dir_path + self.mask_RegEx["Cloud"])
        self.mask_data.append(cloudmask)

        # Cirrus mask
        cirrusmask, _ = load_mask_or_placeholder(self.dir_path + self.mask_RegEx["Cirrus"], template_shape)
        cirrusmask[cirrusmask < 4] = 0  # remove thin cirrus classes
        cirrusmask[cirrusmask > 0] = 1  # merge cirrus clouds to form binary class
        self.mask_data.append(cirrusmask)

        # Other masks (Haze, Cloud_shadow, Snow) follow the same pattern
        for mask_type in ["Haze", "Cloud_shadow"]:  # , "Snow"
            mask, _ = load_mask_or_placeholder(self.dir_path + self.mask_RegEx[mask_type], template_shape)
            self.mask_data.append(mask)
        print("length of mask_data: ", len(self.mask_data))



    # function to combine all masks into a multiclass mask
    def combine_masks(self):
        print(len(self.mask_data))
        self.multiclass_mask = np.zeros(self.mask_data[0].shape)  # Initialize the multiclass mask
        for i, mask in enumerate(self.mask_data): # start=1
            # Set mask values to the current class value (i)
            self.multiclass_mask[mask != 0] = i

        # overwrite cloudshadow with water mask to address CS-Water confusion in original data for training
        self.multiclass_mask = np.where(self.mask_data[2] == 2, 2, self.multiclass_mask)

# apply binary opening (for removing flase positives from water mask)
    def apply_binary_opening(self, mask_index, structure_size=3):
        input_mask = self.mask_data[mask_index]
        input_mask = input_mask.squeeze()
        structure = np.ones((structure_size, structure_size))
        output_mask = binary_opening(input_mask, structure=structure)

        # add arbitrary third dim to make shape fit
        output_mask = np.expand_dims(output_mask, axis=0)

        # update mask data
        self.mask_data[mask_index] = output_mask.astype(np.uint8)

    # buffer water mask to exclude coastal areas due to high missclassification rate in original data
    def buffer_water_mask(self, buffer_size=3):
        water_mask = self.mask_data[2]
        buffered_water_mask_outwards = binary_dilation(water_mask, iterations=buffer_size)
        buffered_water_mask_inwards = binary_erosion(water_mask, iterations=buffer_size-1)

        # extract pixels extended through buffering
        extended_water_mask_outwards = buffered_water_mask_outwards - water_mask
        extended_water_mask_inwards = water_mask - buffered_water_mask_inwards

        #update the multiclass mask (attention, this change is not applied to the mask data list)
        self.multiclass_mask = np.where(extended_water_mask_outwards == 1, 0, self.multiclass_mask)
        self.multiclass_mask = np.where(extended_water_mask_inwards == 1, 0, self.multiclass_mask)


    def format_mask_for_classification(self):
        formatted_mask = self.multiclass_mask
        formatted_mask = np.where(formatted_mask == 1, 1, formatted_mask)
        formatted_mask = np.where(formatted_mask == 2, 1, formatted_mask)
        formatted_mask = np.where(formatted_mask == 3, 2, formatted_mask)
        formatted_mask = np.where(formatted_mask == 4, 1, formatted_mask)
        formatted_mask = np.where(formatted_mask == 5, 1, formatted_mask)
        formatted_mask = np.where(formatted_mask == 6, 3, formatted_mask)

        self.classification_mask = formatted_mask


    def prediction_postprocessing(self, binary_mask, structure_size=4, buffer_size=2):
        # TODO: make sure binray_mask is updated in self.___
        print("binary_mask shape: ", binary_mask.shape)
        binary_mask = np.squeeze(binary_mask)
        print("binary_mask shape: ", binary_mask.shape)
        # remove missclassification with water
        binary_mask = np.where(self.mask_data[2] == 1, 0, binary_mask)

        # remove noise
        binary_mask = binary_erosion(binary_mask, iterations=1)
        print("binary_mask shape: ", binary_mask.shape)
        mask_padded = binary_dilation(binary_mask, iterations=buffer_size)
        structure = np.ones((1, structure_size, structure_size))
        closed_mask = binary_closing(mask_padded, structure=structure)
        final_mask = binary_opening(closed_mask, structure=structure)
        print("done")
        binary_mask = final_mask


    def format_predicted_mask_to_binary(self):
        self.new_cloudshadow_mask = np.where(self.predicted_mask == 3, 1, 0).astype(np.uint8)
        self.new_cloud_mask = np.where(self.predicted_mask == 2, 1, 0).astype(np.uint8)


    def save_mask_to_geotiff(self, raster=None, filename_prefix=None):
        output_path = self.dir_path + "/" + filename_prefix + ".tif"
        if len(raster.shape) != 2:
            # remove arbitratry third dim
            raster_to_save = np.squeeze(raster.copy())
        else:
            raster_to_save = raster

        with rasterio.open(output_path, 'w', driver=self.profile['driver'], height=self.mask_data[1].shape[1],
                           width=self.mask_data[1].shape[2], count=1, dtype=str(self.profile['dtype']), crs=self.profile['crs'],
                           transform=self.profile['transform']) as dst:
            dst.write(raster_to_save, 1)


