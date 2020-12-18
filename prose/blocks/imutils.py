from .base import Block
from astropy.io import fits
import numpy as np
from astropy.time import Time
from os import path
import imageio
from .. import visualisation as viz
from astropy.table import Table
import pandas as pd
from .. import utils, io
from astropy.stats import SigmaClip
from photutils import Background2D, MedianBackground
from .psf import cutouts


class Stack(Block):
    """Build a FITS stack image of the observation

    Parameters
    ----------
    destination : str, optional
        path of the stack image (must be a .fits file name), dfault is None and does not save
    header : dict, optional
        header base of the stack image to be saved, default is None for fresh header
    overwrite : bool, optional
        weather to overwrite file if exists, by default False
    """

    def __init__(self, destination=None, header=None, overwrite=False, **kwargs):

        super(Stack, self).__init__(**kwargs)
        self.stack = None
        self.n_images = 0
        self.stack_header = header
        self.destination = destination
        self.fits_manager = None
        self.overwrite = overwrite

        self.reference_image_path = None

    def run(self, image):
        if self.stack is None:
            self.stack = image.data

        else:
            self.stack += image.data

        self.n_images += 1

    def terminate(self):

        self.stack /= self.n_images

        if self.destination is not None:
            self.stack_header[self.telescope.keyword_image_type] = "Stack image"
            self.stack_header["BZERO"] = 0
            self.stack_header["REDDATE"] = Time.now().to_value("fits")
            self.stack_header["NIMAGES"] = self.n_images

            # changing_flip_idxs = np.array([
            #     idx for idx, (i, j) in enumerate(zip(self.fits_manager.files_df["flip"],
            #                                          self.fits_manager.files_df["flip"][1:]), 1) if i != j])
            #
            # if len(changing_flip_idxs) > 0:
            #     self.stack_header["FLIPTIME"] = self.fits_manager.files_df["jd"].iloc[changing_flip_idxs].values[0]

            stack_hdu = fits.PrimaryHDU(self.stack, header=self.stack_header)
            stack_hdu.writeto(self.destination, overwrite=self.overwrite)


class StackStd(Block):

    def __init__(self, destination=None, overwrite=False, **kwargs):
        super(StackStd, self).__init__(**kwargs)
        self.images = []
        # self.stack_header = None
        # self.destination = destination
        self.overwrite = overwrite
        self.stack_std = None

    def run(self, image, **kwargs):
        self.images.append(image.data)

    def terminate(self):
        self.images = np.array(self.images)
        # shape_divisors = utils.divisors(self.images[0].shape[1])
        # n = shape_divisors[np.argmin(np.abs(50 - shape_divisors))]
        self.stack_std = np.std(self.images, axis=0) #concatenate([np.std(im, axis=0) for im in np.split(self.images, n, axis=1)])
        # stack_hdu = fits.PrimaryHDU(self.stack_std, header=self.stack_header)
        # stack_hdu.header["IMTYPE"] = "std"
        # stack_hdu.writeto(self.destination, overwrite=self.overwrite)


class SaveReduced(Block):
    """Save reduced FITS images.

    Parameters
    ----------
    destination : str
        folder path of the images. Orignial name is used with the addition of :code:`_reduced.fits`
    overwrite : bool, optional
        weather to overwrite file if exists, by default False
    """
    def __init__(self, destination, overwrite=False, **kwargs):

        super().__init__(**kwargs)
        self.destination = destination
        self.overwrite = overwrite

    def run(self, image, **kwargs):

        new_hdu = fits.PrimaryHDU(image.data)
        new_hdu.header = image.header

        image.header["SEEING"] = new_hdu.header.get(self.telescope.keyword_seeing, "")
        image.header["BZERO"] = 0
        image.header["REDDATE"] = Time.now().to_value("fits")
        image.header[self.telescope.keyword_image_type] = "reduced"

        fits_new_path = path.join(
            self.destination,
            path.splitext(path.basename(image.path))[0] + "_reduced.fits"
        )

        new_hdu.writeto(fits_new_path, overwrite=self.overwrite)


class Video(Block):
    """Build a video of all :code:`Image.data`.

    Can be either from raw image or a :code:`int8` rgb image.

    Parameters
    ----------
    destination : str
        path of the video which format depends on the extension (e.g. :code:`.mp4`, or :code:`.gif)
    overwrite : bool, optional
        weather to overwrite file if exists, by default False
    factor : float, optional
        subsampling factor of the image, by default 0.25
    fps : int, optional
        frames per second of the video, by default 10
    from_fits : bool, optional
        Wether :code:`Image.data` is a raw fits image, by default False. If True, a z scaling is applied as well as casting to `uint8`
    """

    def __init__(self, destination, overwrite=True, factor=0.25, fps=10, from_fits=False, **kwargs):

        super().__init__(**kwargs)
        self.destination = destination
        self.overwrite = overwrite
        self.images = []
        self.factor = factor
        self.fps = fps
        self.from_fits = from_fits

    def initialize(self, *args):
        # Check if writer is available (sometimes require extra packages)
        _ = imageio.get_writer(self.destination, mode="I")

    def run(self, image, **kwargs):
        if self.from_fits:
            self.images.append(viz.gif_image_array(image.data, factor=self.factor))
        else:
            self.images.append(image.data.copy())

    def terminate(self):
        imageio.mimsave(self.destination, self.images, fps=self.fps)

    def citations(self):
        return "imageio"


from astropy.stats import sigma_clipped_stats


class RemoveBackground(Block):

    def __init__(self):
        super().__init__()
        self.stack_data = None

    def run(self, image, **kwargs):
        _, im_median, _ = sigma_clipped_stats(image.data, sigma=3.0)
        image.data = im_median


class CleanCosmics(Block):

    def __init__(self, threshold=2):
        super().__init__()
        self.stack_data = None
        self.threshold = threshold
        self.sigma_clip = SigmaClip(sigma=3.)
        self.bkg_estimator = MedianBackground()

    def initialize(self, fits_manager):
        if fits_manager.has_stack():
            self.stack_data = fits.getdata(fits_manager.get("stack")[0])
        self.std_stack = fits.getdata(path.join(fits_manager.folder, "test_std.fits"))

    def run(self, image, **kwargs):
        mask = image.data > (self.stack_data + self.std_stack * self.threshold)
        image.data[mask] = self.stack_data[mask]


class Pass(Block):
    """A Block that does nothing"""

    def __init__(self, *args, **kwargs):
        super().__init__()
        pass

    def run(self, image):
        pass


class ImageBuffer(Block):
    """Store the last Image
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.image = None

    def run(self, image):
        self.image = image


class Set(Block):
    """Set specific attribute to every image

    For example to set attributes ``a`` with the value 2 on every image (i.e Image.a = 2):
    
    .. code-block:: python

        from prose import blocks

        set_block = blocks.Set(a=2)

    Parameters
    ----------
    kwargs : kwargs
        keywords argument and values to be set on every image
    """
    def __init__(self, name=None, **kwargs):
        super().__init__(name=name)
        self.kwargs = kwargs

    def run(self, image):
        image.__dict__.update(self.kwargs)


class Cutouts(Block):

    def __init__(self, size=21, **kwargs):
        super().__init__(**kwargs)
        self.size = size

    def run(self, image, **kwargs):
        image.cutouts_idxs, image.cutouts = cutouts(image.data, image.stars_coords, size=self.size)


class Flip(Block):
    def __init__(self, reference_image, **kwargs):
        super().__init__(**kwargs)
        self.reference_image = reference_image
        self.reference_flip_value = None

    def initialize(self, *args):
        self.reference_flip_value = self.reference_image.header.get(self.telescope.keyword_flip, None)

    def run(self, image, **kwargs):
        flip_value = image.header.get(self.telescope.keyword_flip, None)
        if flip_value != self.reference_flip_value:
            image.data = image.data[::-1, ::-1]
