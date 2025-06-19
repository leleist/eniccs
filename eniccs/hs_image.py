import glob
import rasterio
import matplotlib.pyplot as plt

class HsImage:
    """
    HsImage class
    creates a convenient object from a hyperspectral image cube.
    """
    def __init__(self, dir_path, filename_pattern='-SPECTRAL_IMAGE', extensions=['TIF', 'tif', 'TIFF', 'tiff']):
        self.image = None
        self.dir_path = dir_path
        self.image_path = []
        self.filename_pattern = filename_pattern
        self.extensions = extensions
        self.profile = None
        self.no_data_value = None
        self.metadata = None

        # test trough different extensions
        for ext in extensions:
            self.image_path.extend(glob.glob(f"{dir_path}/*{filename_pattern}.{ext}"))

        if not self.image_path:
            raise FileNotFoundError(f"No files found with pattern '*{filename_pattern}' and extensions {extensions}")

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
