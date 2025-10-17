import glob
import os
import rasterio
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import binary_dilation, binary_opening, binary_closing, binary_erosion
from skimage.measure import label, regionprops
from skimage.morphology import remove_small_holes
from scipy.spatial import cKDTree

class Mask:
    """
    Mask class, creates a convenient object from a set of masks.
    allows for loading and processing of cloud and cloudshadow masks.

    attributes:
    - dir_path: path to the directory containing the masks, also used for saving the EnICCS masks
    - mask_patterns: dictionary containing the patterns for the mask files in operational naming
      convention
    - profile: rasterio profile of the first loaded mask
    - transform: transform of the first loaded mask
    - datatake_name: name of the datatake in native naming convention, used for saving the
      EnICCS masks
    - mask_data: list containing the loaded masks. Note: data herein will be modified inplace.
    - multiclass_mask: combined masks into one multiclass raster for processing
    - coastal_buffer: binary mask containing the buffered coastal pixels to handle areas commonly
      misclassified in native data
    - classification_mask: improved mask, used for training
    - predicted_mask: mask predicted by the model
    - new_cloud_mask: updated cloud mask after postprocessing, binary
    - new_cloudshadow_mask: updated cloudshadow mask after postprocessing, binary
    """
    def __init__(self, dir_path, mask_patterns=None, num_samples=3000):
        if mask_patterns is None:
            self.mask_patterns = dict(Classes='_CLASSES', Cloud='_CLOUD', Cloud_shadow='CLOUDSHADOW')
        else:
            self.mask_patterns = mask_patterns

        self.dir_path = dir_path
        self.profile = None
        self.transform = None
        self.datatake_name = None
        self.mask_data = [] # Placeholder for loaded mask data
        self.nodata_mask = None
        self.multiclass_mask = None
        self.min_samples = num_samples
        self.coastal_buffer = None
        self.classification_mask = None
        self.predicted_mask = None
        self.new_cloud_mask = None
        self.new_cloudshadow_mask = None
        self.validation_report = None
        self.VIP_scores = None

        # load all masks and combine them into a multiclass mask upon initialization
        self.load_masks()

        # check if cloud and cloud shadow masks contain enough pixels
        self._check_CCS_presence()


    # function to load and collect all masks into a list
    def load_masks(self):
        """
        Loads necessary masks and appends them to the mask_data list.
        """

        def load_pattern(pattern):
            paths = []
            for ext in ['TIF', 'tif', 'TIFF', 'tiff', 'BSQ', 'bsq']:
                paths.extend(glob.glob(f"{self.dir_path}/*{pattern}.{ext}"))
            if not paths:
                raise FileNotFoundError(
                    f"Required mask file not found: *{pattern}.[TIF|tif|TIFF|tiff|BSQ|bsq] in"
                    f" {self.dir_path}")

            with rasterio.open(paths[0]) as src:
                if self.transform is None:
                    self.transform = src.transform
                    self.profile = src.profile
                    self.datatake_name = os.path.basename(paths[0])[0:85]
                return src.read()


        # Load each mask separately
        # land and water mask
        classesmask = load_pattern(self.mask_patterns['Classes'])
        nodatamask = np.zeros(classesmask.shape)
        nodatamask[classesmask == 3] = 1   # 3 is nodata value of the Classes mask file!
        self.mask_data.append(nodatamask)
        self.nodata_mask = nodatamask

        landmask = np.zeros(classesmask.shape)
        landmask[classesmask == 1] = 1  # set land class to 1
        self.mask_data.append(landmask)

        watermask = np.zeros(classesmask.shape)
        watermask[classesmask == 2] = 1  # set water class to 1
        self.mask_data.append(watermask)

        # Cloud mask
        cloudmask = load_pattern(self.mask_patterns['Cloud'])
        self.mask_data.append(cloudmask)

        # Cloud shadow mask
        cloudshadowmask = load_pattern(self.mask_patterns['Cloud_shadow'])
        self.mask_data.append(cloudshadowmask)

    def _check_CCS_presence(self):
        """
        Checks if cloud and cloud shadow masks contain enough pixels/samples.
        """
        for mask in self.mask_data[3:5]:  # cloud and cloudshadow masks
            pixelcount = np.sum(mask == 1)  # Count pixels that are exactly 1

            if pixelcount < self.min_samples:
                raise ValueError(
                    f'Insufficient cloud/shadow pixels: {pixelcount} < {self.min_samples}. '
                    f'Recommended default is 3000. Consider lowering min_samples parameter.')



    # function to combine all masks into a multiclass mask
    def combine_masks(self):
        """
        Combines all loaded masks into a multiclass mask.
        """

        self.multiclass_mask = np.zeros(self.mask_data[0].shape)
        for i, mask in enumerate(self.mask_data): # start=1
            self.multiclass_mask[mask != 0] = i

        # overwrite cloudshadow with water mask to address CS-Water confusion in original data for training
        self.multiclass_mask = np.where(self.mask_data[2] == 2, 2, self.multiclass_mask)


    # buffer water mask to exclude coastal areas due to high missclassification rate in original data
    def buffer_water_mask(self, buffer_size=3):
        """
        Specifically handles the operational water mask as it often contains false positives for
        cloudshadow in coastal areas, both within and outside the water mask.
        Updates the multiclass mask in place.

        :param buffer_size: size of the buffer in pixels
        """
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
        # todo: very weird implementation of coastal uffer. is it needed?


    def format_mask_for_classification(self):
        """
        Formats the multiclass mask into a classification mask suitable for ML training.
        """

        formatted_mask = self.multiclass_mask
        formatted_mask = np.where(formatted_mask == 1, 1, formatted_mask) # land
        formatted_mask = np.where(formatted_mask == 2, 1, formatted_mask) # water
        formatted_mask = np.where(formatted_mask == 3, 2, formatted_mask) # cloud
        formatted_mask = np.where(formatted_mask == 4, 3, formatted_mask) # cloud shadow

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
    #     return final_mask

    def prediction_postprocessing(self, binary_mask, structure_size=2, buffer_size=1, neutral_smooth=True): # TODO whats up with buffer size?
        """
        Postprocessing of the predicted mask to remove noise and smooth the output
        :param binary_mask: binary mask to be postprocessed
        :param structure_size: int, size of the structuring element for morphological operations
        :param buffer_size: size of the buffer for binary dilation
        :param neutral_smooth: boolean, whether to apply morphological shape area neutral smoothing operations

        :return: postprocessed binary mask
        """



        # TODO: make sure binray_mask is updated in self.___
        binary_mask = np.squeeze(binary_mask)

        # close small holes
        binary_mask = remove_small_holes(binary_mask.astype(bool), area_threshold=600, connectivity=1)

        # remove missclassification with water
        # binary_mask = np.where(self.mask_data[2] == 1, 0, binary_mask)

        # remove noise
        if neutral_smooth:
            #binary_mask = binary_erosion(binary_mask, iterations=1)
            mask_padded = binary_dilation(binary_mask, iterations=buffer_size)
            structure = np.ones((structure_size, structure_size))
            closed_mask = binary_closing(mask_padded, structure=structure)
            final_mask = binary_opening(closed_mask, structure=structure)
            final_mask = np.squeeze(final_mask)
        else:
            final_mask = binary_mask

        return final_mask

    def reset_cs_coastal_pixels(self):
        """
        Reset CS masks for coastal pixels due to high uncertainties in native Data.
        this method updates self.new_cloudshadow_mask
        """
        water_mask = self.mask_data[2].squeeze()
        # TODO: Check redundancy with buffer_water_mask and binary_opening. 3 are a little much from the same task?
        self.new_cloudshadow_mask = np.squeeze(
            np.where(self.coastal_buffer == 1, 0, self.new_cloudshadow_mask))
        # Update the cloud-shadow mask where there is both water and cloud shadow
        # self.new_cloudshadow_mask = np.where((water_mask == 1) & (self.new_cloudshadow_mask == 1),
                                            # 1, self.new_cloudshadow_mask)
        #

    def _resolve_cs_water_confusion(self):
        """
        Resolve cloud-shadow water confusion.
        step1: close holes in cloud shadow mask that originate from CS misclassification as water
        step2: update water mask to remove missclassified pixels
        """

        cloudshadow_mask = self.predicted_mask
        cloudshadow_mask = np.where(cloudshadow_mask == 3, 1, 0)
        cloudshadow_mask = cloudshadow_mask > 0
        cloudshadow_mask = remove_small_holes(cloudshadow_mask, area_threshold=400, connectivity=1)

        self.new_cloudshadow_mask = cloudshadow_mask.astype(np.uint8)

        water_mask = self.mask_data[2].squeeze()
        # where water == 1 and cloudshadow == 1, set water to 0

        # update water mask with no_data mask
        nodata_mask = self.nodata_mask.squeeze()
        water_mask = np.where(nodata_mask == 1, 0, water_mask)

        # resolves EnMAP native dark shadow-water confusion to a large extent
        water_mask = np.where((water_mask == 1) & (self.new_cloudshadow_mask == 1), 0, water_mask)
        # was 0 in v0.17, causing larger waterbodies to sometimes be misclassied as CS
        # changed back in v0.18.2 to 1, as it was in v0.16
        self.mask_data[2] = np.expand_dims(water_mask, axis=0)

        # set CS mask to 0 where water mask is 1
        self.new_cloudshadow_mask = np.where(water_mask == 1, 0, self.new_cloudshadow_mask)

        # recalculate coastal buffer
        self.buffer_water_mask()

    def _create_combined_mask(self):
        """
        Create a combined mask from the EnICCS cloud and cloud shadow masks.
        0 = background, 1 = clouds, 2 = cloud shadows
        This is used for cloud-cloudshadow adjacency checks during postprocessing.

        :return: combined mask
        """
        cloud_mask = self.new_cloud_mask
        cloud_shadow_mask = self.new_cloudshadow_mask

        combined_mask = np.zeros(cloud_mask.shape, dtype=int)
        combined_mask[cloud_mask > 0] = 1  # Clouds as 1
        combined_mask[cloud_shadow_mask > 0] = 2  # Cloud shadows as 2
        return combined_mask

    def _check_if_touching(self, labeled_clouds, labeled_shadows):
        """
        Check if cloud and shadow regions are adjacent.
        np.roll is used to apply the different shifts to the cloud mask,
        thereby moving it around 1 pixel in all 8 directions to check for adjacency.

        :param labeled_clouds: labeled cloud mask (binary)
        :param labeled_shadows: labeled cloud shadow mask (binary)

        :return: a boolean array where each shadow label/object is marked True if it touches a cloud.
        """

        # TODO: maybe include water where it is intersecting with cloud shadows!!!!!

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

    def _modify_cloud_shadows_based_on_centroid_distance(self, percentile=0.80, verbose=False, plot=False):
        """
        Approximates cloud-cloud shadow association based on the distance between cloud and shadow centroids.
        The distance threshold is calculated based on the specified percentile of the nearest neighbor distances
        and the ratio of cloud and cloud shadow pixels in the scene.
        Updates new_cloud_mask and new_cloudshadow_mask in place.

        :param percentile: percentile of the nearest neighbor distances to use as threshold, 'Auto' for automatic calculation
        :param plot: boolean, whether to plot the distance histogram
        :param verbose: boolean, whether to print the number of modified cloud shadows
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
        if percentile == 'auto': # TODO: Fix or remove 'auto' option
            cloud_pixel_count = np.sum(cloud_mask)
            cloudshadow_pixel_count = np.sum(cloud_shadow_mask)
            ratio = (cloud_pixel_count * 1.15 / cloudshadow_pixel_count) * 100 # initial setting: 1.15
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
        if verbose:
            print(f'Total cloud shadows selected for deletion: {num_modified}')

        if plot:
            plt.hist(distances, bins=20, edgecolor='black')
            # plt.axvline(x=threshold, color='r', linestyle='--')
            plt.title(f'Distribution of Shadow Distances to Nearest Cloud Centroid')
            plt.xlabel('Distance to nearest cloud centroid')
            plt.ylabel('Frequency')
            plt.show()

        # set all pixels with 3 to 0
        result_mask = np.where(result_mask == 3, 0, result_mask)

        # split the combined mask into cloud and cloudshadow mask
        self.new_cloud_mask = np.where(result_mask == 1, 1, 0)
        self.new_cloudshadow_mask = np.where(result_mask == 2, 1, 0)


    def _format_predicted_masks_to_binary(self):
        """
        Formats the predicted masks to binary cloud and cloudshadow masks for postprocessing.
        """
        self.new_cloudshadow_mask = np.where(self.predicted_mask == 3, 1, 0).astype(np.uint8)
        self.new_cloud_mask = np.where(self.predicted_mask == 2, 1, 0).astype(np.uint8)

    def reapply_nodata_mask(self):
        """
        Reapplies the no data mask to the cloud and cloudshadow masks.
        """
        self.new_cloud_mask = np.where(self.nodata_mask == 1, 0, self.new_cloud_mask)
        self.new_cloudshadow_mask = np.where(self.nodata_mask == 1, 0, self.new_cloudshadow_mask)

    def save_mask_to_geotiff(self, raster=None, filename_prefix=None):
        """
        Saves a raster to a geotiff file in the associated directory.
        Can also be used to save intermediate results from the Mask object.
        Uses georeferencing Info from the first loaded mask.

        :param raster: raster to save
        :param filename_prefix: prefix for the filename
        """

        output_path = self.dir_path + '/' + filename_prefix + '.tif'
        if len(raster.shape) != 2:
            # remove arbitratry third dim
            raster_to_save = np.squeeze(raster.copy()) # TODO: Check: native masks are 3D with shape (1, x, y)!
        else:
            raster_to_save = raster

        with rasterio.open(output_path, 'w', driver=self.profile['driver'], height=self.mask_data[1].shape[1],
                           width=self.mask_data[1].shape[2], count=1, dtype=str(self.profile['dtype']),
                           crs=self.profile['crs'], transform=self.profile['transform'], compression='zstd') as dst:
            dst.write(raster_to_save, 1)
