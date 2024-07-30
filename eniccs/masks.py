import rasterio
import glob
import os
import numpy as np
import matplotlib.pyplot as plt

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
            #print(f'Class {i} has value:, {i} ')
            print(mask.shape)
            print(i)
            # Set mask values to the current class value (i)
            self.multiclass_mask[mask != 0] = i
        # return self

    def save_mask_to_geotiff(self):
        # get datatake_ID and tile_ID
        base_name = self.datatake_name
        parts = base_name.split("-")  # split into parts

        datatake = parts[2][13:29]  # after EnMAP folder naming convention
        tileno = parts[2][30:33]  # after EnMAP folder naming convention

        transform = self.metadata['transform']
        output_path = self.dir_path + "/" + datatake + "_" + tileno + "_refined_mask.tif"
        with rasterio.open(output_path, 'w', driver='GTiff', height=mask.shape[1],
                           width=mask.shape[2], count=1, dtype=str(mask.dtype), crs='EPSG:4326',
                           transform=transform) as dst:
            dst.write(mask)


    def save_mask_to_geotiff_p(self):
        output_path = self.dir_path + "/" + self.datatake_name + "_REFINED_QA_Mask.tif"
        with rasterio.open(output_path, 'w', driver=self.profile['driver'], height=self.multiclass_mask.shape[1],
                           width=self.multiclass_mask.shape[2], count=1, dtype=str(self.profile['dtype']), crs=self.profile['crs'],
                           transform=self.profile['transform']) as dst:
            dst.write(self.multiclass_mask)


