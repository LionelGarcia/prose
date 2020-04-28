from os import path
import datetime
import pandas as pd
from astropy.io import fits
import numpy as np
from tabulate import tabulate
from collections import OrderedDict
from tqdm import tqdm
from prose import utils
from prose.telescope import Telescope
import glob
from prose import CONFIG
import warnings
from prose.lightcurves import LightCurves


def phot2dict(filename, format="fits"):
    if format == "fits":
        hdu = fits.open(filename)
        dictionary = {h.name.lower(): h.data for h in hdu}
        dictionary["header"] = hdu[0].header
    
    return dictionary


def trapphot_phot2dict():
    pass


def get_files(
    ext,
    folder,
    deepness=0,
    return_folders=False,
    single_list_removal=True,
    none_for_empty=False,
):
    """

    Return files of specific extension in the specified folder and sub-folders

    Parameters
    ----------
    folder : str
        Folder to be analyzed
    deepness : int
        Number how sub-folder layer to look into.
        0 (default) will look into current folder
        1 will look into current folder and its sub-folders
        2 will look into current folder, its sub-folders and their sub-folders
        ... etc

    Returns
    -------
    list of fits files

    """
    files = []
    for deepness in range(deepness + 1):
        files += glob.iglob(
            path.join(folder, "*/" * deepness + "*{}".format(ext)), recursive=False
        )

    files = [path.abspath(f) for f in files]

    if return_folders:
        folders = [path.dirname(file) for file in files]
        if single_list_removal and len(folders) == 1:
            return folders[0]
        else:
            return folders
    else:
        if single_list_removal and len(files) == 1:
            return files[0]
        elif len(files) == 0 and none_for_empty:
            return None
        else:
            return files


class FitsManager:
    def __init__(self, folder=None, verbose=True, telescope_kw="TELESCOP", deepness=1, light_kw="light"):
        self.deepness = deepness
        self._temporary_files_headers = None
        self._temporary_files_paths = None
        self.files_df = None
        self.verbose = verbose
        self.telescope_kw = telescope_kw
        self.light_kw = light_kw

        self.folder = folder

        self.config = CONFIG

        self.check_telescope_file()
        self.telescope = Telescope()
        self.build_files_df(self.folder)
        self._original_files_df = self.files_df.copy()

        assert len(self._original_files_df) != 0, "No data found"

    def build_files_df(self, folder):

        self._get_files_headers(folder)
        paths = []
        dates = []
        telescope = []
        types = []
        targets = []
        filters = []
        combined = []
        complete_date = []
        dimensions = []
        flip = []
        jd = []

        _types = ["flat", "dark", "bias", self.light_kw]

        last_telescope_name = "fake_00000"
        _temporary_telescope = Telescope()

        if self.verbose:
            _tqdm = tqdm
        else:
            _tqdm = lambda x: x

        for i, header in enumerate(_tqdm(self._temporary_files_headers)):

            telescope_name = header[self.telescope_kw].lower()
            if telescope_name != last_telescope_name:
                _temporary_telescope.load(
                    CONFIG.match_telescope_name(telescope_name)
                )

            _path = self._temporary_files_paths[i]
            _complete_date = utils.format_iso_date(header.get(_temporary_telescope.keyword_observation_date), night_date=False)
            _date = utils.format_iso_date(header.get(_temporary_telescope.keyword_observation_date), night_date=True)
            _telescope = _temporary_telescope.name
            _target = header.get(_temporary_telescope.keyword_object, "")
            _type = header.get(_temporary_telescope.keyword_image_type, "").lower()
            _flip = header.get(_temporary_telescope.keyword_flip, "")
            
            if _temporary_telescope.keyword_light_images.lower() in _type:
                _type = "light"
            elif _temporary_telescope.keyword_dark_images.lower() in _type:
                _type = "dark"
            elif _temporary_telescope.keyword_bias_images.lower() in _type:
                _type = "bias"
            elif "stack" in _type:
                _type = "stack"
            elif _type == self.light_kw:
                _type = self.light_kw

            _filter = header.get(_temporary_telescope.keyword_filter, "")
            _combined = "{}_{}_{}_{}_{}".format(
                _date.strftime("%Y%m%d"), _type, _telescope, _target, _filter,
            )
            _dimensions = "{}x{}".format(header["NAXIS1"], header["NAXIS2"])
            _jd = header.get(_temporary_telescope.keyword_julian_date, "")

            paths.append(_path)
            dates.append(_date)
            telescope.append(_telescope)
            types.append(_type)
            targets.append(_target)
            filters.append(_filter)
            combined.append(_combined)
            complete_date.append(_complete_date)
            dimensions.append(_dimensions)
            flip.append(_flip)
            jd.append(_jd)

        self.files_df = pd.DataFrame(
            {
                "date": dates,
                "complete_date": complete_date,
                "path": paths,
                "telescope": telescope,
                "dimensions": dimensions,
                "type": types,
                "target": targets,
                "filter": filters,
                "combined": combined,
                "flip": flip,
                "jd": jd
            }
        )

        self.sort_by_date()

    def check_telescope_file(self):
        """
        Check for telescope.id file in folder and copy it to specphot config folder

        Returns
        -------

        """
        id_files = get_files(".id", self.folder, single_list_removal=False)

        if len(id_files) > 0:
            assert (
                len(id_files) == 1
            ), "Multiple .id files in your folder, please clean up"
            self.config.save_telescope_file(id_files[0])

    def reset(self):
        self.files_df = self._original_files_df.copy()

    def get(
        self,
        im_type=None,
        telescope=None,
        date=None,
        filter=None,
        target=None,
        return_conditions=False,
    ):

        if not filter:
            filter = None
        if not target:
            target = None
        if not telescope:
            telescope = None

        conditions = pd.Series(np.ones(len(self.files_df)).astype(bool))
        if im_type is not None:
            conditions = conditions & self.files_df["type"].str.contains(im_type)
        if date is not None:
            if isinstance(date, datetime.date):
                date = date.strftime("%Y%m%d")

            conditions = conditions & (
                self.files_df["date"].apply(lambda _d: _d.strftime("%Y%m%d")) == date
            )
        if telescope is not None:
            conditions = conditions & (
                self.files_df["telescope"]
                .str.lower()
                .str.contains(telescope.lower() + "*")
            )
        if filter is not None:
            conditions = conditions & (
                self.files_df["filter"].str.lower().str.contains(filter.lower() + "*")
            )
        if target is not None:
            conditions = conditions & (
                self.files_df["target"]
                .str.lower()
                .str.contains(target.replace("+", "\+").lower() + "*")
            )

        if return_conditions:
            return conditions
        else:
            return self.files_df.loc[conditions]["path"].values

    def set_telescope(self, name=None):
        """
        Set telescope object

        Parameters
        ----------
        name : str
            name of the telescope to set
        """
        self.telescope.load(self.config.match_telescope_name(name))

    def keep(
        self,
        telescope=None,
        date=None,
        im_filter=None,
        target=None,
        keep_closest_calibration=True,
        check_telescope=True,
        calibration_date_limit=0,
    ):
        self.files_df = self.files_df.loc[
            self.get(
                return_conditions=True,
                telescope=telescope,
                filter=im_filter,
                target=target,
                date=date,
            )
        ]

        obs_telescopes = np.unique(self.files_df["telescope"])
        assert len(obs_telescopes) != 0, "No observation found"
        assert (
            len(obs_telescopes) == 1
        ), "Multiple observations found, please add constraints"

        obs_telescope = np.unique(self.files_df["telescope"])[0]
        obs_dimensions = np.unique(self.files_df["dimensions"])[0]

        self.set_telescope(obs_telescope)

        if keep_closest_calibration:

            # date of the kept observation
            obs_date = np.unique(self.files_df["date"])[0]

            if not check_telescope:
                obs_telescope = None
            dark = self.find_closest_calibration(
                obs_date,
                "dark",
                obs_dimensions,
                obs_telescope,
                days_limit=calibration_date_limit,
            )
            bias = self.find_closest_calibration(
                obs_date,
                "bias",
                obs_dimensions,
                obs_telescope,
                days_limit=calibration_date_limit,
            )
            flat = self.find_closest_calibration(
                obs_date,
                "flat",
                obs_dimensions,
                obs_telescope,
                days_limit=calibration_date_limit,
            )

            self.files_df = pd.concat([self.files_df, dark, bias, flat])
            self.sort_by_date()

    def find_closest_calibration(
        self, observation_date, im_type, obs_dimensions, telescope=None, days_limit=0
    ):
        """

        Parameters
        ----------
        observation_date : "YYYYmmdd"
            date of the observation from which closest calibration data need to be found

        im_type : "bias", "dark" or "flat"
            calibration type

        obs_dimensions: pixels x pixels
            dimensions of the images from observation, example: 2000x2000

        telescope : str (default: None)
            telescope from which closest calibration data need to be found

        Returns
        -------

        """
        original_df = self._original_files_df.copy()

        # Find all dark
        condition = original_df["type"].str.contains(im_type + "*")

        condition_checker = bool(len(original_df[condition]))

        if not condition_checker:
            raise ValueError("No '{}' calibration could be retrieved".format(im_type))

        # Check telescope
        if telescope is not None:
            condition = condition & original_df["telescope"].str.lower().str.contains(
                telescope.lower() + "*"
            )

        condition_checker = bool(len(original_df.loc[condition]))

        if not condition_checker:
            raise ValueError(
                "No '{}' calibration from {} could be retrieved. Common error when calibration "
                "files do not provide telescope information (use "
                "--no-check if using CLI)".format(im_type, telescope)
            )

        # Check dimensions
        condition = condition & (original_df["dimensions"] == obs_dimensions)
        condition_checker = bool(len(original_df.loc[condition]))

        if not condition_checker:
            raise ValueError(
                "Could not find calibration images of {} pixels for {}".format(
                    obs_dimensions, telescope
                )
            )

        calibration = original_df.loc[condition]

        # sorted calibration rows
        sorted_calib = calibration.loc[
            (observation_date - calibration["date"])
            >= datetime.timedelta(days=-days_limit)
        ].dropna(
            subset=["date"]
        )  # We only look for files prior or during the day of observation
        closest_combined = sorted_calib.iloc[
            (observation_date - sorted_calib["date"]).argsort()
        ]["combined"]

        if len(closest_combined) == 0:
            raise AssertionError(
                "Calibration could not be found. Here are available data:\n{}\nlooking for "
                "calibration {} days in the {} (check --days-limit parameter)".format(
                    self.describe("calib", original=True, return_string=True),
                    np.abs(days_limit),
                    "future" if days_limit >= 0 else "past",
                )
            )

        closest_combined = closest_combined.iloc[0]

        calibration = original_df.loc[
            original_df["combined"].str.contains(closest_combined + "*") & condition
        ]

        return calibration

    def sort_by_date(self):
        self.files_df = self.files_df.sort_values(["complete_date"]).reset_index(
            drop=True
        )

    def has_calibration(self):
        return (
            len(self.get("dark")) > 0
            and len(self.get("flat")) > 0
            and len(self.get("bias")) > 0
        )

    def _get_files_headers(self, folder):
        self._temporary_files_paths = get_files(".f*ts", folder, deepness=self.deepness)
        self._temporary_files_headers = []

        if self.verbose:
            _tqdm = tqdm
        else:
            _tqdm = lambda x: x

        for f in _tqdm(self._temporary_files_paths):
            self._temporary_files_headers.append(fits.getheader(f))

    def describe(self, table_format="obs", return_string=False, original=False):

        if original:
            files_df = self._original_files_df.copy()
        else:
            files_df = self.files_df.copy()

        if "obs" in table_format:
            headers = ["index", "date", "telescope", "target", "filter", "quantity"]

            observations = self.observations()
            rows = OrderedDict(observations[headers].to_dict(orient="list"))

            table_string = tabulate(rows, headers, tablefmt="fancy_grid")

        elif "calib" in table_format:
            headers = [
                "date",
                "telescope",
                "type",
                "target",
                "dimensions",
                "filter",
                "quantity",
            ]

            multi_index_obs = files_df.pivot_table(
                index=["date", "telescope", "type", "target", "dimensions", "filter"],
                aggfunc="size",
            )

            single_index_obs = (
                multi_index_obs.reset_index()
                .rename(columns={0: "quantity"})
                .reset_index(level=0)
            )
            rows = OrderedDict(single_index_obs[headers].to_dict(orient="list"))

            table_string = tabulate(rows, tablefmt="fancy_grid", headers="keys",)

        elif "files" in table_format:
            headers = [
                "index",
                "date",
                "telescope",
                "type",
                "dimensions",
                "target",
                "filter",
            ]
            rows = OrderedDict(files_df.reset_index()[headers].to_dict(orient="list"))
            table_string = tabulate(rows, tablefmt="fancy_grid", headers="keys")

        else:
            raise ValueError(
                "{} is not an accepted format. Accepted format are 'obs', 'calib' and 'files'".format(
                    table_format
                )
            )

        if return_string:
            return table_string
        else:
            print(table_string)

    def trim(self, image):
        if isinstance(image, np.ndarray):
            pass
        elif isinstance(image, str):
            if path.exists(image) and image.lower().endswith((".fts", ".fits")):
                image = fits.getdata(image)
        else:
            raise ValueError("{} should be a numpy array or a fits file")

        return image[
            self.telescope.trimming[1] : -self.telescope.trimming[1],
            self.telescope.trimming[0] : -self.telescope.trimming[0],
        ]

    def observations(self):
        light_rows = self.files_df.loc[self.files_df["type"].str.contains(self.light_kw)]
        observations = (
            light_rows.pivot_table(
                index=["date", "telescope", "target", "dimensions", "filter"],
                aggfunc="size",
            )
            .reset_index()
            .rename(columns={0: "quantity"})
            .reset_index(level=0)
        )

        return observations

    @property
    def products_denominator(self):
        single_obs = self.observations()

        assert len(single_obs) == 1, "Multiple observations found"

        single_obs = single_obs.iloc[0]

        return "{}_{}_{}_{}".format(
            self.telescope.name,
            single_obs["date"].strftime("%Y%m%d"),
            single_obs["target"],
            single_obs["filter"],
        )

    def set_observation(
            self,
            observation_id,
            check_calib_telescope=True,
            keep_closest_calibration=False,
            calibration_date_limit=0):

        observations = self.observations()

        assert observation_id in observations["index"], "index {} do not match any observation".format(observation_id)

        obs = observations[observations["index"] == observation_id].iloc[0]

        self.keep(
            telescope=obs["telescope"],
            date=obs["date"],
            im_filter=obs["filter"],
            target=obs["target"],
            check_telescope=check_calib_telescope,
            keep_closest_calibration=keep_closest_calibration,
            calibration_date_limit=calibration_date_limit
        )


def fits_keyword_values(fits_files, keywords, default_value=None, verbose=False):
    """

    Get the values of specific keywords in a list of fits files

    Parameters
    ----------
    fits_files: list(str)
        List of fits files (string path)
    default_value : any
        Value to be returned if keyword is not found (default is None). If None return a keyError instead
    keywords: list(str)
        List of keywords

    Returns
    -------
    List of keyword values for any files with shape (number_of_keywords, number_of_files)

    """
    if not verbose:
        _tqdm = lambda l, **kwargs: l
    else:
        _tqdm = tqdm

    header_values = []

    for f in _tqdm(fits_files):
        fits_header = fits.getheader(f)
        if default_value is None:
            header_values.append([fits_header[keyword] for keyword in keywords])
        else:
            header_values.append([fits_header[keyword] for keyword in keywords])

    # If only one keyword the list is flattened
    if type(keywords) == str:
        header_values = [hv for hvs in header_values for hv in hvs]

    return header_values


def save_phot_fits(phot, destination=None):
    """
    Save data into a ``.phots`` file at specified destination. File name is : ``Telescope_date(YYYYmmdd)_target_filter``. For more info check :doc:`/notes/phots-structure`
    
    Parameters
    ----------
    destination : str path, optional
        path of destination where to save file, by default None
    """
    if destination is None:
        if phot.folder  is not None:
            destination = phot.photometry_path
        else:
            raise ValueError("destination must be specified")
    else:
        destination = path.join(
            destination, "{}.phots".format(phot.products_denominator)
        )

    phot.photometry_path = destination

    header = fits.PrimaryHDU(header=fits.getheader(phot.phot_file))

    header.header.update({
        "TARGETID": phot.target["id"],
        "TELESCOP": phot.telescope.name,
        "OBSERVAT": phot.telescope.name,
        "FILTER": phot.filter,
        "NIMAGES": phot.n_images
    })

    hdu_list = [
        header,
        fits.ImageHDU(phot.light_curves.as_array()[0], name="photometry"),
        fits.ImageHDU(phot.stars_coords, name="stars"),
        fits.ImageHDU(phot.comparison_stars, name="comparison stars"),
        fits.ImageHDU(phot.apertures, name="apertures"),
        fits.ImageHDU(phot.artificial_lc, name="artificial lcs"),
        fits.ImageHDU(phot._time, name="jd"),
        fits.ImageHDU(phot.bjd_tdb, name="bjd")
    ]

    for keyword in [
        "fwhm", "sky", "dx", "dy", "airmass",
        (phot.telescope.keyword_exposure_time.lower(), "exptime"),
        (phot.telescope.keyword_julian_date.lower(), "jd"),
    ]:
        if isinstance(keyword, str):
            if keyword in phot.data:
                hdu_list.append(fits.ImageHDU(phot.data[keyword], name=keyword))
        elif isinstance(keyword, tuple):
            if keyword[0] in phot.data:
                hdu_list.append(
                    fits.ImageHDU(phot.data[keyword[0]], name=keyword[1])
                )

    if phot.differential_light_curves is not None:
        lcs, lcs_errors = phot.differential_light_curves.as_array()
        hdu_list.append(fits.ImageHDU(lcs, name="lightcurves"))
        hdu_list.append(fits.ImageHDU(lcs_errors, name="lightcurves errors"))

    hdu = fits.HDUList(hdu_list)
    hdu.writeto(destination, overwrite=True)


def load_phot_fits(phot, phots_path, sort_stars=True):
    phot_dict = phot2dict(phots_path)

    header = phot_dict["header"]
    phot.n_images = phot_dict.get("nimages", None)
    
    # Loading telescope, None if name doesn't match any 
    telescope = Telescope()
    telescope_name = header.get(telescope.keyword_observatory, None)
    found = telescope.load(CONFIG.match_telescope_name(telescope_name))
    phot.telescope = telescope if found else None
    if phot.telescope is not None:
        ra = header.get(phot.telescope.keyword_ra, None)
        dec = header.get(phot.telescope.keyword_dec, None)
        phot.target["radec"] = [ra, dec]

    # Loading info
    phot.filter = header.get(phot.telescope.keyword_filter, None)
    phot.observation_date = utils.format_iso_date(
        header.get(phot.telescope.keyword_observation_date, None))
    phot.target["name"] = header.get(phot.telescope.keyword_object, None)

    # Loading time and exposure
    phot._time = phot_dict.get("jd", None)
    if phot._time is not None: phot._compute_bjd()

    phot.exposure = header.get(phot.telescope.keyword_exposure_time)
    if phot.exposure is None:
        phot.exposure = np.min(np.diff(phot.time))
        warnings.warn("Exposure not found in headers, computed from time")

    # Loading fluxes and sort by flux if specified
    fluxes = phot_dict.get("photometry", None)
    assert fluxes is not None
    fluxes_error = phot_dict.get("photometry errors", None)
    star_mean_flux = np.mean(np.mean(fluxes, axis=0), axis=1)
    if sort_stars:
        sorted_stars = np.argsort(star_mean_flux)[::-1]
    else:
        sorted_stars = np.arange(0, np.shape(fluxes)[1])
    fluxes = fluxes[:, sorted_stars, :]
    if fluxes_error is not None: fluxes_error = fluxes_error[:, sorted_stars, :]

    # Loading stars, target, apertures
    phot.stars_coords = phot_dict.get("stars", None)[sorted_stars]
    phot.apertures = phot_dict.get("apertures", None)
    target_id = header.get("targetid", None)
    if target_id is not None:
        phot.target["id"]= sorted_stars[target_id]
    
    # Loading light curves
    lcs = phot_dict.get("lightcurves", None)
    lcs_error = phot_dict.get("lightcurves errors", None)
    if lcs is not None: lcs = lcs[:, sorted_stars, :]
    if lcs_error is not None: lcs_error = lcs_error[:, sorted_stars, :]
    comparison_stars = phot_dict.get("comparison stars", None)
    phot.artificial_lcs = phot_dict.get("artificial lcs", None)
    if comparison_stars is not None: phot.comparison_stars = sorted_stars[comparison_stars]
    
    # Loading all known systematics
    for key in ["fwhm", "sky", "dx", "dy", "airmass", "exptime"]:
        phot.data[key] = phot_dict.get(key, None)
    phot._data_as_attributes()

    time = phot.time
    a, s, f = fluxes.shape # saved as (apertures, stars, fluxes) for conveniance

    # Photometry into LightCurve objects
    if fluxes_error is None:
        fluxes_error = np.empty(np.shape(fluxes))
        for i, ape in enumerate(phot.apertures):
            fluxes_error[i, :] = phot.error(
            fluxes[i, :], np.pi * ape ** 2, method="scinti"
        )
    phot.light_curves = LightCurves(
        time, np.moveaxis(fluxes, 1, 0), np.moveaxis(fluxes_error, 1, 0))
    phot.light_curves.apertures = phot.apertures

    # Differential photometry into LightCurve objects
    if lcs is not None:
        phot.differential_light_curves = LightCurves(time, np.moveaxis(lcs, 1, 0), np.moveaxis(lcs_error, 1, 0))
        best_aperture_id = phot.differential_light_curves[phot.target["id"]]._best_aperture_id
        phot.differential_light_curves.set_best_aperture_id(best_aperture_id)

    # self._compute_fluxes_errors()