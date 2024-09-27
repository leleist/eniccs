import glob
import os
import rasterio
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import binary_dilation, binary_opening, binary_closing, binary_erosion
from skimage.measure import label, regionprops
from scipy.spatial import cKDTree

class Mask:
    def __init__(self, dir_path, mask_regex=None):
        if mask_regex is None:
            self.mask_regex = dict(Classes="/*_CLASSES.TIF", Cloud="/*_CLOUD.TIF",
                                   Cirrus="/*CIRRUS.TIF", Haze="/*HAZE.TIF",
                                   Cloud_shadow="/*CLOUDSHADOW.TIF") # TODO: only load necessary ones? because they get saved individually anyways, maybe not because of div. overlaps?

        else:
            self.mask_regex = mask_regex

        self.dir_path = dir_path
        self.profile = None
        self.transform = None
        self.datatake_name = None
        self.mask_data = [] # Placeholder for loaded mask data
        self.multiclass_mask = None
        self.multiclass_mask_native = None
        self.coastal_buffer = None
        self.classification_mask = None
        self.predicted_mask = None
        self.new_cloud_mask = None
        self.new_cloudshadow_mask = None

        # load all masks and combine them into a multiclass mask upon initialization
        self.load_masks()
        # copy native multiclass mask
        self.multiclass_mask_native = self.combine_masks()


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
                        self.transform = src.transform  # save Georeference info for later use
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
        classesmask, template_shape = load_mask_or_placeholder(self.dir_path + self.mask_regex["Classes"])
        nodatamask = np.zeros(classesmask.shape)
        nodatamask[classesmask == 3] = 1  # set background class to 1
        self.mask_data.append(nodatamask)

        landmask = np.zeros(classesmask.shape)
        landmask[classesmask == 1] = 1  # set land class to 1
        self.mask_data.append(landmask)

        watermask = np.zeros(classesmask.shape)
        watermask[classesmask == 2] = 1  # set water class to 1
        self.mask_data.append(watermask)

        # Cloud mask
        cloudmask, template_shape = load_mask_or_placeholder(self.dir_path + self.mask_regex["Cloud"])
        self.mask_data.append(cloudmask)

        # Cirrus mask
        cirrusmask, _ = load_mask_or_placeholder(self.dir_path + self.mask_regex["Cirrus"], template_shape)
        cirrusmask[cirrusmask < 4] = 0  # remove thin cirrus classes
        cirrusmask[cirrusmask > 0] = 1  # merge cirrus clouds to form binary class
        self.mask_data.append(cirrusmask)

        # Other masks (Haze, Cloud_shadow, Snow) follow the same pattern
        for mask_type in ["Haze", "Cloud_shadow"]:  # , "Snow"
            mask, _ = load_mask_or_placeholder(self.dir_path + self.mask_regex[mask_type], template_shape)
            self.mask_data.append(mask)
        # print("length of mask_data: ", len(self.mask_data))

    # function to combine all masks into a multiclass mask
    def combine_masks(self):
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

        # combine both extended masks into one bianry mask
        extended_water_mask = np.where(extended_water_mask_outwards == 1, 1, 0)
        extended_water_mask = np.where(extended_water_mask_inwards == 1, 1, extended_water_mask)

        water_buffer_pixels = extended_water_mask - water_mask

        self.coastal_buffer = np.where(self.mask_data[0] == 1, 0, water_buffer_pixels)



    def format_mask_for_classification(self):
        formatted_mask = self.multiclass_mask
        formatted_mask = np.where(formatted_mask == 1, 1, formatted_mask)
        formatted_mask = np.where(formatted_mask == 2, 1, formatted_mask)
        formatted_mask = np.where(formatted_mask == 3, 2, formatted_mask)
        formatted_mask = np.where(formatted_mask == 4, 1, formatted_mask)
        formatted_mask = np.where(formatted_mask == 5, 1, formatted_mask)
        formatted_mask = np.where(formatted_mask == 6, 3, formatted_mask)

        self.classification_mask = formatted_mask

    # def prediction_postprocessing(self, binary_mask, structure_size=4, buffer_size=2):
    #     # TODO: make sure binray_mask is updated in self.___
    #     binary_mask = np.squeeze(binary_mask)
    #     # remove missclassification with water
    #     binary_mask = np.where(self.mask_data[2] == 1, 0, binary_mask)
#
    #     # remove noise
    #     binary_mask = binary_erosion(binary_mask, iterations=1)
    #     mask_padded = binary_dilation(binary_mask, iterations=buffer_size)
    #     structure = np.ones((1, structure_size, structure_size))
    #     closed_mask = binary_closing(mask_padded, structure=structure)
    #     final_mask = binary_opening(closed_mask, structure=structure)
    #     # print("done")
    #     return final_mask

    def prediction_postprocessing(self, binary_mask, structure_size=4, buffer_size=2):
        # TODO: make sure binray_mask is updated in self.___
        binary_mask = np.squeeze(binary_mask)
        # remove missclassification with water
        binary_mask = np.where(self.mask_data[2] == 1, 0, binary_mask)

        # remove noise
        binary_mask = binary_erosion(binary_mask, iterations=1)
        mask_padded = binary_mask# binary_dilation(binary_mask, iterations=buffer_size)
        structure = np.ones((1, structure_size, structure_size))
        closed_mask = binary_closing(mask_padded, structure=structure)
        final_mask = binary_opening(closed_mask, structure=structure)
        # print("done")
        return final_mask

    def reset_cs_coastal_pixels(self):
        """
        Reset CS masks for coastal pixels due to high uncertainties in native EnMAP Data.

        this function updates self.new_cloudshadow_mask
        """

        self.new_cloudshadow_mask = np.squeeze(
            np.where(self.coastal_buffer == 1, 0, self.new_cloudshadow_mask))

    def _create_combined_mask(self):

        cloud_mask = self.new_cloud_mask
        cloud_shadow_mask = self.new_cloudshadow_mask

        combined_mask = np.zeros(cloud_mask.shape, dtype=int)
        combined_mask[cloud_mask > 0] = 1  # Clouds as 1
        combined_mask[cloud_shadow_mask > 0] = 2  # Cloud shadows as 2
        return combined_mask

    def _check_if_touching(self, labeled_clouds, labeled_shadows):
        """
        Check if cloud and shadow regions are adjacent.
        Returns a boolean array where each shadow label is marked True if it touches a cloud.

        np.roll is used to apply the different shifts to the cloud mask, thereby moving it around 1 pixel in all 8 directions.
        """

        cloud_border = (labeled_clouds > 0)
        shadow_border = (labeled_shadows > 0)

        # Shift the cloud mask in all 8 directions and check for overlap with shadow mask
        touch = np.zeros_like(labeled_clouds, dtype=bool)
        for shift_x in [-1, 0, 1]:
            for shift_y in [-1, 0, 1]:
                if shift_x == 0 and shift_y == 0:
                    continue
                touch |= np.roll(cloud_border, shift=(shift_x, shift_y), axis=(0, 1)) & shadow_border

        return touch

    def _modify_cloud_shadows_based_on_centroid_distance(self, percentile="Auto", plot_bool=False):
        """


        """
        cloud_mask = self.new_cloud_mask
        cloud_shadow_mask = self.new_cloudshadow_mask

        # Label the cloud and cloud shadow masks
        labeled_clouds = label(cloud_mask)
        labeled_shadows = label(cloud_shadow_mask)

        # Extract centroids of clouds and shadows
        cloud_props = regionprops(labeled_clouds)
        shadow_props = regionprops(labeled_shadows)

        cloud_centroids = np.array([prop.centroid for prop in cloud_props])
        shadow_centroids = np.array([prop.centroid for prop in shadow_props])

        # Use KDTree for efficient nearest neighbor search
        tree = cKDTree(cloud_centroids)

        # Query the tree for the nearest cloud centroid for each shadow centroid
        distances, _ = tree.query(shadow_centroids)

        # Calculate the threshold based on the specified percentile or find it automatically
        if percentile == "Auto":
            cloud_pixel_count = np.sum(cloud_mask)
            cloudshadow_pixel_count = np.sum(cloud_shadow_mask)
            ratio = (cloud_pixel_count * 1.15 / cloudshadow_pixel_count) * 100
            percentile = min(max(ratio, 0), 100) # clamp to range [0, 100] in case of very few detected CS
        else:
            percentile = percentile


        # Calculate the threshold based on the specified percentile
        threshold = np.percentile(distances, percentile)

        # Check if any cloud shadows touch clouds
        touch_mask = self._check_if_touching(labeled_clouds, labeled_shadows)

        # Create the combined mask (0 = background, 1 = clouds, 2 = cloud shadows)
        result_mask = self._create_combined_mask()

        num_modified = 0

        # Process each shadow region
        for shadow_prop in shadow_props:
            shadow_index = shadow_prop.label - 1  # Adjust for 0-indexing
            min_distance = distances[shadow_index]

            # Get the shadow region mask
            shadow_region = (labeled_shadows == shadow_prop.label)

            # Check if any part of the shadow is touching a cloud
            touching_cloud = np.any(touch_mask[shadow_region])

            if not touching_cloud and min_distance > threshold:
                result_mask[shadow_region] = 3
                num_modified += 1

        print(f'Total cloud shadows selected for deletion: {num_modified}')

        if plot_bool:
            plt.hist(distances, bins=20, edgecolor='black')
            plt.title(f"Distribution of Shadow Distances to Nearest Cloud Centroid\nThreshold: {threshold}")
            plt.xlabel("Distance to nearest cloud centroid")
            plt.ylabel("Frequency")
            plt.show()

        # set all pixels with 3 to 0
        result_mask = np.where(result_mask == 3, 0, result_mask)

        # split the combined mask into cloud and cloudshadow mask
        self.new_cloud_mask = np.where(result_mask == 1, 1, 0)
        self.new_cloudshadow_mask = np.where(result_mask == 2, 1, 0)




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
                           width=self.mask_data[1].shape[2], count=1, dtype=str(self.profile['dtype']),
                           crs=self.profile['crs'], transform=self.profile['transform']) as dst:
            dst.write(raster_to_save, 1)
