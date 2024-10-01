import glob
import rasterio
import matplotlib.pyplot as plt

class HsImage:
    """
    HsImage class
    creates a convenient object from a hyperspectral image.
    allows for easy plotting of bands
    """
    def __init__(self, dir_path, image_regex=r"/*-SPECTRAL_IMAGE.TIF"):
        self.image = None
        self.dir_path = dir_path
        self.image_path = glob.glob(dir_path + image_regex)
        self.image_regex = image_regex
        self.profile = None
        self.no_data_value = None
        self.metadata = None

        # load image upon initialization
        self._load_image()

    def _load_image(self):
        with rasterio.open(self.image_path[0]) as src:
            self.image = src.read()
            self.profile = src.profile
            self.no_data_value = src.nodata

    def plot_band(self, band=0):
        """ Plot a band of the hyperspectral image
        :param band: band ID to plot individual band
        """
        plt.imshow(self.image[band, :, :])
        plt.show()
