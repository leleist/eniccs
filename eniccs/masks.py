import rasterio
import glob
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import binary_dilation, binary_opening, binary_closing, binary_erosion



mask_RegEx = {"Classes" : "/*_CLASSES.TIF", # water, land, backround
              "Cloud" : "/*_CLOUD.TIF",
              "Cirrus" : "/*CIRRUS.TIF",
              "Haze" : "/*HAZE.TIF",
              "Cloud_shadow" : "/*CLOUDSHADOW.TIF"
              #"Snow" : "/*SNOW.TIF",
              }

class mask:
    def __init__(self, dir_path, mask_RegEx=None):
        if mask_RegEx is None:
            mask_RegEx = dict(Classes="/*_CLASSES.TIF", Cloud="/*_CLOUD.TIF", Cirrus="/*CIRRUS.TIF",
                              Haze="/*HAZE.TIF", Cloud_shadow="/*CLOUDSHADOW.TIF")
        self.multiclass_mask = mask
        self.dir_path = dir_path
        self.mask_regEx = mask_RegEx
        self.profile = None
        self.datatake_name = None
        self.mask_data = [] # Placeholder for loaded mask data
        self.multiclass_mask = None

        # load all masks and combine them into a multiclass mask upon initialization
        self.load_masks()
        # self.combine_masks() # not usefull here because individual masks will be updated in main!

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
        classesmask, template_shape = load_mask_or_placeholder(self.dir_path + mask_RegEx["Classes"])
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
        cloudmask, template_shape = load_mask_or_placeholder(self.dir_path + mask_RegEx["Cloud"])
        self.mask_data.append(cloudmask)

        # Cirrus mask
        cirrusmask, _ = load_mask_or_placeholder(self.dir_path + mask_RegEx["Cirrus"], template_shape)
        cirrusmask[cirrusmask < 4] = 0  # remove thin cirrus classes
        cirrusmask[cirrusmask > 0] = 1  # merge cirrus clouds to form binary class
        self.mask_data.append(cirrusmask)

        # Other masks (Haze, Cloud_shadow, Snow) follow the same pattern
        for mask_type in ["Haze", "Cloud_shadow"]:  # , "Snow"
            mask, _ = load_mask_or_placeholder(self.dir_path + mask_RegEx[mask_type], template_shape)
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





    # default: saves multiclass raster to geotiff, can save any intermediate/derivetive mask as well if specified
    def save_mask_to_geotiff(self, alternative_raster=None, alternative_filename=None):
        if alternative_filename is not None:
            output_path = self.dir_path + "/" + alternative_filename + ".tif"
        else:

            output_path = self.dir_path + "/" + self.datatake_name + "_REFINED_QA_Mask.tif"

        if alternative_raster is not None:
            raster_to_save = alternative_raster

        else:
            raster_to_save = self.multiclass_mask

        with rasterio.open(output_path, 'w', driver=self.profile['driver'], height=self.multiclass_mask.shape[1],
                           width=self.multiclass_mask.shape[2], count=1, dtype=str(self.profile['dtype']), crs=self.profile['crs'],
                           transform=self.profile['transform']) as dst:
            dst.write(raster_to_save)


