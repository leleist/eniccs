import rasterio
import glob
import numpy as np


class hs_image:
    def __init__(self, dir_path, image_regEx=r"/*-SPECTRAL_IMAGE.TIF"):
        self.image = None
        self.dir_path = dir_path
        self.image_path = glob.glob(dir_path + image_regEx)
        self.image_regEx = image_regEx
        self.profile = None
        self.no_data_value = None
        self.metadata = None
        self.nodata_mask = None

        # load image upon initialization
        self.load_image()
        self.get_no_data_mask()



    def load_image(self):
        with rasterio.open(self.image_path[0]) as src:
            self.image = src.read()
            self.profile = src.profile
            self.no_data_value = src.nodata
        return self

    def get_no_data_mask(self):  # TODO: merge into land water mask?
        no_data_condition = self.image[0, :, :] == self.no_data_value  # boolean mask for no data pixels
        no_data_condition_inv = np.invert(no_data_condition)
        no_data_mask = np.zeros(self.image.shape[1:])  # TODO: check if this is the right shape
        self.nodata_mask = np.where(no_data_condition_inv, 1, no_data_mask)
        return self

