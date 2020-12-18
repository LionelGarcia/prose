from .. import Unit, blocks, io, Block, Telescope
from . import io as bio
import os
from os import path
from astropy.io import fits
from ..console_utils import INFO_LABEL
import numpy as np
import time
from ..diagnostics.show_stars import ShowStars


class Reduction:
    """A reduction unit producing a reduced FITS folder

    Parameters
    ----------
    fits_manager : prose.FitsManager
        Fits manager of the observation. Should contain a single obs
    destination : str, optional
        Destination of the newly created folder, by default beside the folder given to FitsManager
    reference : float, optional
        Reference image to use for alignment from 0 (first image) to 1 (last image), by default 1/2
    overwrite : bool, optional
        wether to overwrtie existing products, by default False
    n_images : int, optional
        number of images to process, by default None for all images
    calibration : bool, optional
        weather to perform calibration, by default True (if False images are still trimmed)
    ignore_telescope: bool, optional
        whether to load a default telescope if telescope not recognised, by default False
    """

    def __init__(
            self,
            fits_manager=None,
            destination=None,
            reference=1 / 2,
            overwrite=False,
            calibration=True,
            flats=None,
            bias=None,
            darks=None,
            alignment=blocks.XYShift,
            psf=blocks.FastGaussian,
            ignore_telescope=False):

        self.fits_manager = fits_manager
        self.destination = destination
        self.overwrite = overwrite
        self.calibration = calibration

        # set on prepare
        self.stack_path = None
        self.gif_path = None

        self.flats = flats
        self.bias = bias
        self.darks = darks

        self.prepare()

        if not ignore_telescope:
            assert self.fits_manager.telescope.name != "Unknown", \
                "Telescope has not been recognised, to load a default one set ignore_telescope=True (kwargs)"

        # set reference file
        reference_id = int(reference * len(self.files))
        self.reference_fits = self.files[reference_id]

        self.reference_unit = None
        self.reduction_unit = None

        assert psf is None or issubclass(psf, Block), "psf must be a subclass of Block"
        self.psf = psf

        assert alignment is None or issubclass(alignment, Block), "alignment must be a subclass of Block"
        self.alignment = alignment

    def run(self):

        self.reference_unit = Unit([
            blocks.Calibration(self.darks, self.flats, self.bias,
                               name="calibration"),
            blocks.Trim(name="trimming"),
            blocks.SegmentedPeaks(n_stars=50, name="detection"),
            blocks.ImageBuffer(name="buffer")
        ], self.reference_fits, telescope=self.fits_manager.telescope, show_progress=False)

        self.reference_unit.run()

        ref_image = self.reference_unit.buffer.image
        calibration_block = self.reference_unit.calibration

        self.reduction_unit = Unit([
            blocks.Pass() if not self.calibration else calibration_block,
            blocks.Trim(name="trimming", skip_wcs=True),
            blocks.Flip(ref_image, name="flip"),
            blocks.SegmentedPeaks(n_stars=50, name="detection"),
            self.alignment(ref_image.stars_coords, name="shift"),
            blocks.Align(ref_image.data, name="alignment"),
            self.psf(name="fwhm"),
            blocks.Stack(self.stack_path, header=ref_image.header, overwrite=self.overwrite, name="stack"),
            blocks.SaveReduced(self.destination, overwrite=self.overwrite, name="saving"),
            ShowStars(),
            blocks.Video(self.gif_path, name="video", from_fits=False)
        ], self.files, telescope=self.fits_manager.telescope, name="Reduction")

        self.reduction_unit.run()

    def prepare(self):
        """
        This will prepare the `self.destination` containing the:

        - ``self.stack_path``
        - ``self.gif_path``

        Returns
        -------

        """

        # Either the input is a FitsManager and the following append:
        if self.fits_manager is not None:
            if self.fits_manager.unique_obs:
                self.fits_manager.set_observation(0, future=100000)
            else:
                _ = self.fits_manager.calib
                raise AssertionError("Multiple observations found")

            if self.destination is None:
                self.destination = path.join(self.fits_manager.folder, self.fits_manager.products_denominator)

            self.stack_path = f"{path.join(self.destination, self.fits_manager.products_denominator)}_stack.fits"
            self.gif_path = f"{path.join(self.destination, self.fits_manager.products_denominator)}_movie.gif"

            self.files = self.fits_manager.images
            self.darks = self.fits_manager.darks
            self.flats = self.fits_manager.flats
            self.bias = self.fits_manager.bias
        else:
            self.stack_path = path.join(self.destination, "stack.fits")
            self.gif_path = path.join(self.destination, "movie.gif")

        if path.exists(self.stack_path) and not self.overwrite:
            raise AssertionError("stack {} already exists, consider using the 'overwrite' kwargs".format(self.stack_path))

        if not path.exists(self.destination):
            os.mkdir(self.destination)

        self.files = self.fits_manager.images

    def __repr__(self):
        return f"{self.reference_unit}\n{self.reduction_unit}"


class Photometry:
    """Base unit for Photometry

    Parameters
    ----------
    fits_manager : prose.FitsManager
         FitsManager of the observation. Should contain a single obs. One of ``fits_manager`` or ``files`` and ``stack` should  be provided
    files : list of str, optional
        List of files to process. One of ``fits_manager`` or ``files`` and ``stack`` should  be provided
    stack: str, optional
        Path of the stack image. Should be specified if ``files`` is specified.
    overwrite : bool, optional
        whether to overwrite existing products, by default False
    n_stars : int, optional
        max number of stars to take into account, by default 500
    ignore_telescope: bool, optional
        whether to load a default telescope if telescope not recognised, by default False
    """

    def __init__(self,
                 fits_manager=None,
                 files=None,
                 stack=None,
                 overwrite=False,
                 n_stars=500,
                 detection=blocks.Pass,
                 psf=blocks.Gaussian2D,
                 ignore_telescope=False):

        self.fits_manager = fits_manager
        self.overwrite = overwrite
        self.n_stars = n_stars
        self.reference_detection_unit = None
        self.photometry_unit = None
        self.destination = None

        # preparing inputs and outputs
        self.destination = None
        self.stack_path = None
        self.phot_path = None
        self.files = None
        self.telescope = None
        self.prepare(fits_manager=fits_manager, files=files, stack=stack)

        if not ignore_telescope:
            assert self.fits_manager.telescope.name != "Unknown", \
                "Telescope has not been recognised, to load a default one set ignore_telescope=True (kwargs)"

        assert psf is None or issubclass(psf, Block), "psf must be a subclass of Block"
        self.psf = psf

        assert detection is None or issubclass(detection, Block), "psf must be a subclass of Block"
        self.detection = detection(cross_match=True)

    def run_reference_detection(self):
        self.reference_detection_unit = Unit([
            blocks.DAOFindStars(n_stars=self.n_stars, name="detection"),
            self.psf(name="fwhm"),
            blocks.ImageBuffer(name="buffer"),
        ], self.stack_path, telescope=self.fits_manager.telescope, show_progress=False)

        self.reference_detection_unit.run()
        stack_image = self.reference_detection_unit.buffer.image
        ref_stars = stack_image.stars_coords
        fwhm = stack_image.fwhm

        print("{} detected stars: {}".format(INFO_LABEL, len(ref_stars)))
        print("{} global psf FWHM: {:.2f} (pixels)".format(INFO_LABEL, np.mean(fwhm)))

        time.sleep(0.5)

        return stack_image, ref_stars, fwhm

    def run(self, destination=None):
        if self.phot_path is None:
            assert destination is not None, "You must provide a destination"
        if destination is not None:
            self.phot_path = destination.replace(".phot", "") + ".phot"

        self._check_phot_path()

        self.run_reference_detection()
        self.photometry_unit.run()

    def _check_phot_path(self):
        if path.exists(self.phot_path) and not self.overwrite:
            raise OSError("{} already exists".format(self.phot_path))

    def prepare(self, fits_manager=None, files=None, stack=None):
        """
        Check that stack and observation is present and set ``self.phot_path``

        """
        if self.fits_manager is not None:
            if isinstance(self.fits_manager, str):
                self.fits_manager = io.FitsManager(self.fits_manager, image_kw="reduced", verbose=False)
            elif isinstance(self.fits_manager, io.FitsManager):
                if self.fits_manager.image_kw != "reduced":
                    print(f"Warning: image keyword is '{self.fits_manager.image_kw}'")

            self.destination = self.fits_manager.folder

            if not self.fits_manager.unique_obs:
                _ = self.fits_manager.observations
                raise AssertionError("Multiple observations found")
            else:
                self.fits_manager.set_observation(0, future=100000)

            self.phot_path = path.join(
                self.destination, "{}.phot".format(self.fits_manager.products_denominator))

            self.files = self.fits_manager.images
            self.stack_path = self.fits_manager.stack

            self.telescope = self.fits_manager.telescope

            self._check_phot_path()

        elif files is not None:
            assert stack is not None, "'stack' should be specified if 'files' is specified"

            self.stack_path = stack
            self.files = files
            self.telescope = Telescope(self.stack_path)

    def __repr__(self):
        return f"{self.reference_detection_unit}\n{self.photometry_unit}"


class AperturePhotometry(Photometry):
    """Aperture Photometry unit

    Parameters
    ----------
    fits_manager : prose.FitsManager
         FitsManager of the observation. Should contain a single obs. One of ``fits_manager`` or ``files`` and ``stack` should  be provided
    files : list of str, optional
        List of files to process. One of ``fits_manager`` or ``files`` and ``stack`` should  be provided
    stack: str, optional
        Path of the stack image. Should be specified if ``files`` is specified.
    overwrite : bool, optional
        whether to overwrite existing products, by default False
    n_stars : int, optional
        max number of stars to take into account, by default 500
    apertures : list or np.ndarray, optional
        Apertures radii to be used. If None, by default np.arange(0.1, 10, 0.25)
    r_in : int, optional
        Radius of the inner annulus to be used in pixels, by default 5
    r_out : int, optional
        Radius of the outer annulus to be used in pixels, by default 8
    fwhm_scale : bool, optional
        wheater to multiply ``apertures``, ``r_in`` and ``r_out`` by the global fwhm, by default True
    sigclip : float, optional
        Sigma clipping factor used in the annulus, by default 3. No effect if :class:`~prose.blocks.SEAperturePhotometry` is used
    psf : Block, optional
        PSF modeling Block (mainly used to estimate fwhm and scale aperture if ``fwhm_scale`` is ``True``), by default :class:`~prose.blocks.Gaussian2D`
    photometry : Block, optional
        aperture photometry Block, by default :class:`~prose.blocks.PhotutilsAperturePhotometry`
    centroid : Block, optional
        centroid computing Block, by default None to keep centroid fixed
    ignore_telescope: bool, optional
        whether to load a default telescope if telescope not recognised, by default False
    """

    def __init__(self,
                 fits_manager=None,
                 files=None,
                 stack=None,
                 overwrite=False,
                 n_stars=500,
                 apertures=None,
                 r_in=5,
                 r_out=8,
                 fwhm_scale=True,
                 sigclip=3.,
                 psf=blocks.Gaussian2D,
                 photometry=blocks.PhotutilsAperturePhotometry,
                 centroid=None,
                 ignore_telescope=False,
                 detection=blocks.Pass):
                 
        if apertures is None:
            apertures = np.arange(0.1, 10, 0.25)

        super().__init__(
            fits_manager=fits_manager,
            files=files,
            stack=stack,
            overwrite=overwrite,
            n_stars=n_stars,
            psf=psf,
            ignore_telescope= ignore_telescope,
            detection=detection
        )

        # Blocks
        assert centroid is None or issubclass(centroid, Block), "centroid must be a subclass of Block"
        self.centroid = centroid
        # ==
        assert photometry is None or issubclass(photometry, Block), "photometry must be a subclass of Block"
        self.photometry = photometry(
            apertures=apertures,
            r_in=r_in,
            r_out=r_out,
            sigclip=sigclip,
            fwhm_scale=fwhm_scale,
            name="photometry",
            set_once=isinstance(self.detection, blocks.Pass)
        )

    def run_reference_detection(self):
        stack_image, ref_stars, fwhm = super().run_reference_detection()
        self.detection.cross_match = ref_stars

        self.photometry_unit = Unit([
            blocks.Set(stars_coords=ref_stars, name="set stars"),
            blocks.Set(fwhm=fwhm, name="set fwhm"),
            self.detection,
            blocks.Pass() if not isinstance(self.centroid, Block) else self.centroid,
            self.photometry,
            bio.SavePhot(self.phot_path, header=stack_image.header, stack=stack_image.data, name="saving")
        ], self.files, telescope=self.telescope, name="Photometry")


class PSFPhotometry(Photometry):
    """PSF Photometry unit (not tested, use not recommended)

    Parameters
    ----------
    fits_manager : prose.FitsManager
         FitsManager of the observation. Should contain a single obs. One of ``fits_manager`` or ``files`` and ``stack` should  be provided
    files : list of str, optional
        List of files to process. One of ``fits_manager`` or ``files`` and ``stack`` should  be provided
    stack: str, optional
        Path of the stack image. Should be specified if ``files`` is specified.
    overwrite : bool, optional
        whether to overwrite existing products, by default False
    n_stars : int, optional
        max number of stars to take into account, by default 500
    psf : Block, optional
        PSF modeling Block (mainly used to estimate fwhm and scale aperture if ``fwhm_scale`` is ``True``), by default :class:`~prose.blocks.Gaussian2D`
    photometry : Block, optional
        aperture photometry Block, by default :class:`~prose.blocks.PhotutilsAperturePhotometry`
    ignore_telescope: bool, optional
        whether to load a default telescope if telescope not recognised, by default False
    """

    def __init__(self,
                 fits_manager=None,
                 files=None,
                 stack=None,
                 overwrite=False,
                 n_stars=500,
                 psf=blocks.Gaussian2D,
                 photometry=blocks.PhotutilsPSFPhotometry,
                 ignore_telescope=False):

        super().__init__(
            fits_manager=fits_manager,
            files=files,
            stack=stack,
            overwrite=overwrite,
            n_stars=n_stars,
            psf=psf,
            ignore_telescope=ignore_telescope
        )

        # Blocks
        assert photometry is None or issubclass(photometry, Block), "photometry must be a subclass of Block"
        self.photometry = photometry

    def run_reference_detection(self):
        stack_image, ref_stars, fwhm = super().run_reference_detection()

        self.photometry_unit = Unit([
            blocks.Set(stars_coords=ref_stars, name="set stars"),
            blocks.Set(fwhm=fwhm, name="set fwhm"),
            self.photometry(fwhm),
            bio.SavePhot(
                self.phot_path,
                header=fits.getheader(self.stack_path),
                stack=fits.getdata(self.stack_path),
                name="saving")
        ], self.files, telescope=self.telescope, name="Photometry")