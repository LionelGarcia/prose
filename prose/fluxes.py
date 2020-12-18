import numpy as np
from . import utils
import matplotlib.pyplot as plt
from . import visualisation as viz
from scipy.optimize import curve_fit
import xarray as xr
from astropy.stats import sigma_clip

# TODO: check if working properly with a single aperture/lc
# TODO: a rms reject method and a sigma clip method in LightCurves
# TODO: test old broeg2005 method and see if returns same results (or simlar)


def differential_photometry(fluxes, errors, comps, return_alc=False):
    np.seterr(divide="ignore")  # Ignore divide by 0 warnings here

    comps = np.array(comps)

    # Normalization
    # ==============
    mean_fluxes = np.nanmean(fluxes, 2)[:, :, None]
    fluxes = fluxes.copy() / mean_fluxes
    errors = errors.copy() / mean_fluxes

    n_apertures, n_stars, n_points = np.shape(fluxes)

    # Compute the final light-curves
    # ==============================
    art_lc = np.array([fluxes[a, comps].mean(0) for a in range(n_apertures)])
    art_error = np.array([np.sqrt(errors[a, comps] ** 2).sum(0) / len(comps) for a in range(n_apertures)])
    lcs = fluxes / art_lc[:, None]
    lcs /= np.median(lcs, 2)[:, :, None]
    lcs_errors = np.sqrt(errors ** 2 + art_error[:, None, :] ** 2)

    np.seterr(divide="warn")  # Set warnings back

    return [lcs, lcs_errors, art_lc] if return_alc else [lcs, lcs_errors]


def nu(n, sw, sr):
    return (sw ** 2) / n + sr ** 2


def pont2006(x, y, n=30, plot=True, return_error=False):
    dt = np.median(np.diff(x))
    ns = np.arange(1, n)
    binned = [utils.fast_binning(x, y, dt * _n)[1] for _n in ns]
    _nu = [np.var(b) for b in binned]
    (sw, sr), pcov = curve_fit(nu, ns, _nu)
    if plot:
        plt.plot(ns, np.std(y) / np.sqrt(ns), ":k", label=r"$\sigma / \sqrt{n}$")
        plt.plot(ns, np.sqrt(nu(ns, sw, sr)), "k", label="fit")
        plt.plot(ns, np.sqrt(_nu), "d", markerfacecolor="w", markeredgecolor="k")
        plt.xlabel("n points")
        plt.ylabel("$\\nu^{1/2}(n)$")
        plt.legend()
    if return_error:
        return sw, sr, np.sqrt(np.diag(pcov))
    else:
        return sw, sr


def broeg2005(
    fluxes,
    errors,
    target,
    keep="float",
    max_iteration=50,
    tolerance=1e-8,
    n_comps=500,
    exclude=None,
    sigclip=3,
):

    np.seterr(divide="ignore")  # Ignore divide by 0 warnings here

    if exclude is not None:
        _exclude = np.array(exclude)
    else:
        _exclude = []

    _time = np.arange(fluxes.shape[-1])

    original_fluxes = fluxes.copy()
    initial_n_stars = np.shape(original_fluxes)[1]

    if sigclip is not None:
        bad_flux = np.abs(original_fluxes - np.nanmean(original_fluxes, 2)[:, :, None]) >\
                   sigclip * np.nanstd(original_fluxes, 2)[:, :, None]
        original_fluxes[bad_flux] = np.repeat(np.nanmean(original_fluxes, 2)[:, :, None], len(_time), -1)[bad_flux]

    # Normalization
    # -------------

    mean_fluxes = np.nanmean(fluxes, 2)[:, :, None]
    fluxes = fluxes.copy() / mean_fluxes
    errors = errors.copy() / mean_fluxes

    # Cleaning
    # --------
    # To keep track of which stars are clean and get a light curve (if not containing infs, nans or zeros)
    clean_stars = np.arange(0, initial_n_stars)

    # Data cleaning, excluding stars with no finite flux (0, nan or inf)
    if not np.all(np.isfinite(fluxes)):
        idxs = np.unique(np.where(np.logical_not(np.isfinite(fluxes)))[1])
        fluxes = np.delete(fluxes, idxs, axis=1)
        errors = np.delete(errors, idxs, axis=1)
        clean_stars = np.delete(clean_stars, idxs)
    if not np.all(np.isfinite(errors)):
        idxs = np.unique(np.where(np.logical_not(np.isfinite(errors)))[1])
        fluxes = np.delete(fluxes, idxs, axis=1)
        errors = np.delete(errors, idxs, axis=1)
        clean_stars = np.delete(clean_stars, idxs)

    n_apertures, n_stars, n_points = np.shape(fluxes)

    i = 0
    evolution = 1e25
    lcs = None
    weights = None
    last_weights = np.zeros((n_apertures, n_stars))
    # record = []

    # Broeg 2004 algorithm to find weights of comp stars
    # --------------------------------------------------
    while evolution > tolerance and i < max_iteration:
        if i == 0:
            weights = 1 / np.mean(errors ** 2, axis=2)
        else:
            # This metric is preferred from std to optimize over white noise and not red noise
            std = utils.std_diff_metric(lcs)
            weights = 1 / std

        # Keep track of weights
        evolution = np.std(
            np.abs(np.mean(weights, axis=1) - np.mean(last_weights, axis=1))
        )
        # record.append(evolution)
        last_weights = weights

        weighted_fluxes = fluxes * weights[:, :, None]
        art_lc = np.sum(weighted_fluxes, 1) / weights.sum(1)[:, None]
        lcs = fluxes / art_lc[:, None, :]

        i += 1

    # Setting target weight to 0
    weights[:, target] = 0
    # Setting weights of stars to exclude to 0
    if exclude is not None:
        weights[:, _exclude] = 0

    ordered_stars = np.argsort(weights, axis=1)[:, ::-1]
    # Remove target and excluded stars (at the end of ordered since weight = 0
    ordered_stars = ordered_stars[:, 0:-(1 + len(_exclude))]

    # Find the best number of comp stars to keep if None
    # --------------------------------------------------
    if isinstance(keep, str):

        k_range = np.arange(1, np.min([np.shape(fluxes)[1], n_comps]))

        weighted_fluxes = fluxes * weights[:, :, None]
        ordered_weighted_fluxes = np.array([weighted_fluxes[a, ordered_stars[a], :] for a in range(n_apertures)])
        ordered_summed_weighted_fluxes = np.array([ordered_weighted_fluxes[:, 0:k, :].sum(1) for k in k_range])
        ordered_summed_weights = np.array([weights[:, 0:k].sum(1) for k in k_range])

        metric = 1e18
        last_metric = np.inf
        i = 0
        _keep = None

        while metric < last_metric:
            _keep = k_range[i]
            last_metric = metric
            _art_lc = ordered_summed_weighted_fluxes[i] / ordered_summed_weights[i][:, None]
            _lcs = fluxes / _art_lc[:, None, :]
            metric = np.std([utils.std_diff_metric(f) for f in _lcs[:, target, :]])

            i += 1

    elif keep is None:
        _keep = np.min([np.shape(fluxes)[1], n_comps])
        keep = "float"
    else:
        _keep = keep
        keep = "float" if isinstance(keep, float) else "int"

    # Compute the final lightcurves
    # -----------------------------
    if keep == "float":
        # Using the weighted sum
        ordered_weights = np.array([weights[a, ordered_stars[a, :]] for a in range(n_apertures)])
    elif keep == "int":
        # Using a simple mean
        ordered_weights = np.ones((n_apertures, _keep))
    else:
        raise AssertionError(f"Unknown issue: keep has value {keep}")

    keep = int(_keep)

    ordered_fluxes = np.array([fluxes[a, ordered_stars[a], :] for a in range(n_apertures)])
    ordered_errors = np.array([errors[a, ordered_stars[a], :] for a in range(n_apertures)])

    best_art_lc = (ordered_fluxes[:, 0:keep, :] * ordered_weights[:, 0:keep, None]).sum(1) / ordered_weights[:, 0:keep].sum(1)[:, None]
    best_art_error = (ordered_errors[:, 0:keep, :] ** 2 * ordered_weights[:, 0:keep, None] ** 2).sum(
        1) / ordered_weights[:, 0:keep].sum(1)[:, None]

    lcs = np.zeros(np.shape(original_fluxes))
    lcs_errors = np.zeros(np.shape(original_fluxes))

    for a in range(n_apertures):
        for s, cs in enumerate(clean_stars):
            lcs[a, cs, :] = fluxes[a, s] / best_art_lc[a, :]
            lcs_errors[a, cs, :] = np.sqrt(
                errors[a, s] ** 2 + best_art_error[a, :] ** 2
            )

    # Return
    # ------------------------------
    np.seterr(divide="warn")  # Set warnings back
    info = {
        "comps": np.array(ordered_stars[:, 0:keep]),
        "weights": np.array(ordered_weights[:, 0:keep]),
        "alc": best_art_lc
    }

    return lcs, lcs_errors, info


def to_xarray(time, fluxes, errors):
    pass



class Fluxes:

    def __init__(self, xarray):
        if isinstance(xarray, str):
            self.xarray = xr.load_dataset(xarray)
        else:
            self.xarray = xarray

    def __getattr__(self, name):
        if name in self.xarray:
            return self.xarray[name].values
        elif name in self.xarray.attrs:
            return self.xarray.attrs[name]
        else:
            return self.xarray.__getattr__(name)

    @property
    def target(self):
        return self.xarray.target

    @target.setter
    def target(self, value):
        self.xarray.attrs['target'] = value
        self._pick_best_aperture()

    @property
    def aperture(self):
        return self.xarray.aperture

    @aperture.setter
    def aperture(self, value):
        self.xarray.attrs['aperture'] = value

    def _repr_html_(self):
        return self.xarray._repr_html_()

    def __repr__(self):
        return self.xarray.__repr__()

    def __str__(self):
        return self.xarray.__str__()

    def __iter__(self):
        return self.xarray.__iter__()

    def mask(self, mask, dim="time"):
        new_self = self.copy()
        new_self.xarray = new_self.xarray.where(xr.DataArray(mask, dims=dim), drop=True)
        return new_self

    @staticmethod
    def _binn(var, *bins):
        if var.dtype.name != 'object':
            if "time" in var.dims:
                if "errors" in var.name:
                    return xr.concat(
                        [(np.sqrt(np.power(var.isel(time=b), 2).sum(dim="time")) / len(b)).expand_dims('time', -1) for b
                         in bins], dim="time")
                else:
                    return xr.concat([var.isel(time=b).mean(dim="time").expand_dims('time', -1) for b in bins],
                                     dim="time")
            else:
                return var
        else:
            return xr.DataArray(np.ones(len(bins)) * -1, dims="time")

    def binn(self, dt, std=False, keep_coords=True):
        x = self.xarray.copy()
        bins = utils.index_binning(self.time, dt)
        new_time = np.array([self.time[b].mean() for b in bins])

        x = x.map(self._binn, args=(bins), keep_attrs=True)

        if std:
            if "fluxes" in self.xarray:
                flu = self.xarray.fluxes.copy()
                x['errors'] = xr.concat(
                    [(flu.isel(time=b).std(dim="time") / np.sqrt(len(b))).expand_dims('time', -1) for b in bins],
                    dim="time")
            if "raw_fluxes" in self.xarray:
                raw_flu = self.xarray.raw_fluxes.copy()
                x['raw_errors'] = xr.concat(
                    [(raw_flu.isel(time=b).std(dim="time") / np.sqrt(len(b))).expand_dims('time', -1) for b in bins],
                    dim="time")

        x.coords['time'] = new_time

        if keep_coords:
            for name, value in self.xarray.coords.items():
                if "time" not in value.dims:
                    x.coords[name] = value

        new_self = self.copy()
        new_self.xarray = x

        return new_self

    @property
    def flux(self):
        return self.xarray.fluxes.isel(apertures=self.aperture, star=self.target).values

    @property
    def error(self):
        return self.xarray.errors.isel(apertures=self.aperture, star=self.target).values

    def __copy__(self):
        return Fluxes(self.xarray.copy())

    def copy(self):
        return Fluxes(self.xarray.copy())

    def _pick_best_aperture(self, method="stddiff", return_criterion=False):
        if len(self.apertures) > 1:
            if method == "stddiff":
                criterion = utils.std_diff_metric(self.xarray.fluxes.sel(star=self.target))
            elif method == "stability":
                criterion = utils.stability_aperture(self.xarray.fluxes.sel(star=self.target))
            elif method == "pont2006":
                criterion = []
                for a in range(len(self.apertures)):
                    self.aperture = a
                    criterion.append(self.pont2006(plot=False)[0])
            else:
                raise ValueError("{} is not a valid method".format(method))

            self.aperture = np.argmin(criterion)

            if return_criterion:
                return criterion
        else:
            self.aperture = 0

    def where(self, *args, **kwargs):
        new_self = self.copy()
        new_self.xarray = self.xarray.where(*args, **kwargs)
        return new_self

    def pont2006(self, plot=True):
        return pont2006(self.time, self.xarray.fluxes.isel(apertures=self.aperture, star=self.target).values, plot=plot)

    @staticmethod
    def _rename_raw(obj):
        obj.xarray = obj.xarray.rename({
            "fluxes": "raw_fluxes",
            "errors": "raw_errors"
        })

    @staticmethod
    def _reset_raw(obj):
        if "raw_fluxes" in obj:
            obj.xarray = obj.xarray.drop_vars(("fluxes", "errors"))
            obj.xarray = obj.xarray.rename({
                "raw_fluxes": "fluxes",
                "raw_errors": "errors"
            })

    # Differential photometry methods
    # ===============================
    def diff(self, comps, keep_raw=True):
        new_self = self.copy()
        self._reset_raw(new_self)
        diff_fluxes, diff_errors, alc = differential_photometry(new_self.fluxes, new_self.errors, comps,
                                                                return_alc=True)
        dims = self.xarray.fluxes.dims

        if keep_raw:
            self._rename_raw(new_self)

        new_self.xarray['fluxes'] = (dims, diff_fluxes)
        new_self.xarray['errors'] = (dims, diff_errors)

        # Since we reset ncomps, older vars with ncomp in dims are removed
        new_self.xarray = new_self.xarray.drop_vars(
            [name for name, value in new_self.xarray.items() if 'ncomps' in value.dims])

        new_self.xarray['comps'] = (("ncomps"), comps)
        new_self.xarray['alc'] = (('apertures', 'time'), alc)
        new_self._pick_best_aperture()

        return new_self

    def broeg2005(self, keep='float', keep_raw=True, exclude=None):
        new_self = self.copy()
        self._reset_raw(new_self)
        diff_fluxes, diff_errors, info = broeg2005(new_self.fluxes, new_self.errors, self.target, keep=keep, exclude=exclude)
        dims = self.xarray.fluxes.dims

        if keep_raw:
            self._rename_raw(new_self)

        new_self.xarray['fluxes'] = (dims, diff_fluxes)
        new_self.xarray['errors'] = (dims, diff_errors)

        # Since we reset ncomps, older vars with ncomp in dims are removed
        new_self.xarray = new_self.xarray.drop_vars(
            [name for name, value in new_self.xarray.items() if 'ncomps' in value.dims])

        new_self.xarray['comps'] = (("apertures", "ncomps"), info['comps'])
        new_self.xarray['weights'] = (("apertures", "ncomps"), info['weights'])
        new_self.xarray['alc'] = (('apertures', 'time'), info['alc'])
        new_self._pick_best_aperture()

        return new_self

    # io
    # ==

    @staticmethod
    def load(filepath):
        return Fluxes(xr.load_dataset(filepath))

    def save(self, filepath):
        self.xarray.to_netcdf(filepath)

    # Plotting
    # ========

    def plot(self, which="None", bins=0.005, color="k", std=True):
        binned = self.binn(bins, std=std)
        plt.plot(self.time, self.flux, ".", c="gainsboro", zorder=0, alpha=0.6)
        plt.errorbar(binned.time, binned.flux, yerr=binned.error, fmt=".", zorder=1, color=color, alpha=0.8)


class LightCurves:
    """
    Object holding time-series of flux and data from multiple stellar observations. Act as a list of ``LightCurve``
    """
    def __init__(self, *lightcurves, **kwargs):
        if len(lightcurves) == 1:
            # Probably list(Lightucurves)
            self._lightcurves = lightcurves[0]
        elif len(lightcurves) == 3:
            # Probably list([time, fluxes, errors])
            time, fluxes, errors = lightcurves
            self._lightcurves = [
                LightCurve(time, flux, error) for flux, error in zip(fluxes, errors)
            ]
        self.apertures = None
        self.best_aperture_id = None

    def __getitem__(self, key):
        return self._lightcurves[key]

    def __len__(self):
        return len(self._lightcurves)

    def __iter__(self):
        return iter(self._lightcurves)

    @property
    def time(self):
        return np.hstack(self.times)

    @property
    def flux(self):
        return np.hstack(self.fluxes)

    @property
    def data(self):
        keys = [list(lc.data.keys()) for lc in self]
        n_keys = [len(k) for k in keys]
        if len(np.unique(n_keys)) != 1:
            raise AssertionError("LightCurves data should have same keys to be retrieved")

        n_keys = np.unique(n_keys)
        unique_keys = np.unique(keys)

        if len(unique_keys) != n_keys:
            raise AssertionError("LightCurves data should have same keys to be retrieved")

        return {key: np.hstack([lc.data[key] for lc in self]) for key in unique_keys}

    @property
    def error(self):
        return np.hstack(self.errors)

    @property
    def times(self):
        return [lc.time for lc in self._lightcurves]

    @property
    def fluxes(self):
        return [lc.flux for lc in self._lightcurves]

    @property
    def errors(self):
        return [lc.error for lc in self._lightcurves]

    def set_best_aperture_id(self, i):
        """
        Set a commonly share best aperture for all LightCurves

        Parameters
        ----------
        i : int
            index of the best aperture
        """
        if i is not None:
            for lc in self._lightcurves:
                lc.best_aperture_id = i
            self.best_aperture_id = i

    def from_ndarray(self, fluxes, errors):
        pass

    def as_array(self, aperture=None):
        """
        Return an ndarray with shape (n_apertures, n_lightcurves, n_times)

        Returns
        -------
        np.ndarray
        """
        if aperture is None:
            array = np.array([lc.fluxes for lc in self._lightcurves])
            error_array = np.array([lc.errors for lc in self._lightcurves])
            s, a, n = array.shape
            return np.moveaxis(array, 0, 1), np.moveaxis(error_array, 0, 1)
        else:
            array = np.array([lc.fluxes[aperture] for lc in self._lightcurves])
            error_array = np.array([lc.errors[aperture] for lc in self._lightcurves])
            return array, error_array

    def mask(self, mask):
        lcs = []
        for lc in self._lightcurves:
            _, _, idxs = np.intersect1d(lc.time, self.time, return_indices=True)
            sub_mask = mask[idxs]
            lcs.append(lc.mask(sub_mask))
        lcs = LightCurves(lcs)
        lcs.set_best_aperture_id(self.best_aperture_id)
        return lcs

    def sigma_clip(self, **kwargs):
        sigclip_mask = sigma_clip(self.flux, **kwargs).mask
        return self.mask(sigclip_mask)

    def binned(self, bins=0.005):
        return LightCurves([lc.binned(bins) for lc in self])

    def plot(self, **kwargs):
        viz.plot_lcs([[lc.time, lc.flux] for lc in self], **kwargs)

    def save(self, name):
        np.save("{}.lcs".format(name), [{
            "fluxes": lc.fluxes,
            "errors": lc.errors,
            "data": lc.data,
            "time": lc.time,
            "best_aperture_id": lc.best_aperture_id
        } for lc in self], allow_pickle=True)

    @staticmethod
    def load(filename):
        data = np.load(filename, allow_pickle=True)
        lcs = []
        for d in data:
            lc = LightCurve(d["time"], d["fluxes"], d["errors"], d["data"])
            lc.best_aperture_id = d["best_aperture_id"]
            lcs.append(lc)

        return LightCurves(lcs)

    def folded(self, t0, period):
        folded_time = utils.fold(self.time, t0, period)
        idxs = np.argsort(folded_time)
        folded_time = folded_time[idxs]
        flux = self.flux[idxs]
        error = self.error[idxs]

        lc = LightCurve(folded_time, [flux], [error])

        return lc