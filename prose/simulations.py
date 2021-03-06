import numpy as np
from . import viz, utils, Observation, models
from photutils.psf import extract_stars
from astropy.table import Table
from astropy.nddata import NDData
import celerite2 as celerite
from prose.blocks.registration import distances
from os import path
from astropy.time import Time
import os
from tqdm import tqdm
from astropy.io import fits
from datetime import datetime
import xarray as xr
import matplotlib.pyplot as plt
import warnings


def fits_image(data, header, destination):

    header = dict(
        TELESCOP=header.get("TELESCOP", "fake"),
        EXPTIME=header.get("EXPTIME", 1),
        FILTER=header.get("FILTER", ""),
        OBJECT=header.get("OBJECT", "prose"),
        IMAGETYP=header.get("IMAGETYP", "light"),
        AIRMASS=header.get("AIRMASS", 1),
        JD=header.get("JD", 0),
        RA=header.get("RA", 12.84412),
        DEC=header.get("DEC", -22.85886),
    )
    header['DATE-OBS'] = header.get("DATE-OBS", Time(datetime.now()).to_value("fits"))
    hdu = fits.PrimaryHDU(data, header=fits.Header(header))
    hdu.writeto(destination, overwrite=True)


def cutouts(image, stars, size):
    stars_tbl = Table(stars.T, names=["x", "y"])
    stars = extract_stars(NDData(data=image), stars_tbl, size=size)
    return stars


def sim_signal(time, amp=1e-3, w=10.):
    kernel = celerite.terms.SHOTerm(S0=1, Q=1, w0=2 * np.pi / w)
    gp = celerite.GaussianProcess(kernel)
    gp.compute(time)
    return 1 + utils.rescale(gp.sample()) * amp / 2


def random_stars(k, shape, sort=True):
    positions = np.random.rand(k, 2) * shape
    return positions


def random_fluxes(k, time, peak=65000, max_amp=1e-2, sort=True):
    fluxes = np.random.beta(1e-2, 100, size=k)
    diff_amplitudes = np.random.beta(1, 8, size=k)

    if sort:
        idxs = np.argsort(fluxes)[::-1]
        fluxes = fluxes[idxs]
        diff_amplitudes = diff_amplitudes[idxs]

    diff_amplitudes /= diff_amplitudes.max()
    diff_amplitudes *= max_amp

    fluxes /= fluxes.max()
    fluxes *= peak
    fluxes += 15

    fluxes = np.repeat(fluxes[:, np.newaxis], len(time), axis=1)
    fluxes *= np.array([sim_signal(time, amp=a) for a in diff_amplitudes])

    return fluxes


def protopapas2005(t, t0, duration, depth, c=20, period=1):
    _t = period * np.sin(np.pi * (t - t0) / period) / (np.pi * duration)
    return (1 - depth) + (depth / 2) * (
            2 - np.tanh(c * (_t + 1 / 2)) + np.tanh(c * (_t - 1 / 2)))


class ObservationSimulation:

    def __init__(self, shape, telescope, n=51):
        if isinstance(shape, (tuple, list)):
            self.shape = shape
        else:
            self.shape = (shape, shape)
        self.telescope = telescope
        self.n = n
        self.x, self.y = np.indices((n, n))

    def set_psf(self, fwhm, theta, beta, model="moffat"):
        self.beta = beta
        self.theta = theta * np.pi / 180
        self.sigma = np.array(fwhm) / self.sigma_to_fwhm
        if model is "moffat":
            self.psf_model = self.moffat_psf
        elif model is "gaussian":
            self.psf_model = self.gaussian_psf

    def moffat_psf(self, a, x, y):
        # https://pixinsight.com/doc/tools/DynamicPSF/DynamicPSF.html
        dx_ = self.x - x
        dy_ = self.y - y
        dx = dx_ * np.cos(self.theta) + dy_ * np.sin(self.theta)
        dy = -dx_ * np.sin(self.theta) + dy_ * np.cos(self.theta)
        sx, sy = self.sigma

        return a / np.power(1 + (dx / sx) ** 2 + (dy / sy) ** 2, self.beta)

    def gaussian_psf(self, a, x, y):
        dx = self.x - x
        dy = self.y - y
        sx, sy = self.sigma
        a = (np.cos(self.theta) ** 2) / (2 * sx ** 2) + (np.sin(self.theta) ** 2) / (2 * sy ** 2)
        b = -(np.sin(2 * self.theta)) / (4 * sx ** 2) + (np.sin(2 * self.theta)) / (4 * sy ** 2)
        c = (np.sin(self.theta) ** 2) / (2 * sx ** 2) + (np.cos(self.theta) ** 2) / (2 * sy ** 2)
        im = a * np.exp(-(a * dx ** 2 + 2 * b * dx * dy + c * dy ** 2))
        return im

    def field(self, i):
        image = np.zeros(self.shape)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cuts = cutouts(image, self.positions[:, :, i].T, self.n)
        fluxes = self.fluxes * self.atmosphere[np.newaxis, :]
        for c, f in zip(cuts, fluxes[:, i]):
            image[c.slices[0], c.slices[1]] += self.psf_model(f, *c.cutout_center)

        return image

    def remove_stars(self, idxs):
        k = self.positions.shape[0]
        self.positions = self.positions[np.setdiff1d(np.arange(k), idxs), :, :]
        self.fluxes = self.fluxes[np.setdiff1d(np.arange(k), idxs), :]

    @property
    def sigma_to_fwhm(self):
        return 2 * np.sqrt(np.power(2, 1 / self.beta) - 1)

    def add_stars(self, k, time, atmosphere=6e-2, peak=65000, positions=None, fluxes=None):
        # Generating time series
        self.time = time
        if positions is None:
            positions = np.repeat(random_stars(k, np.min(self.shape))[:, :, np.newaxis], len(time), axis=2)
        self.positions = positions
        if fluxes is None:
            fluxes = random_fluxes(k, time, peak=peak)
        self.fluxes = fluxes

        if atmosphere is not None:
            # Atmosphere signal
            self.atmosphere = sim_signal(time, w=0.5, amp=atmosphere)
        else:
            self.atmosphere = np.ones_like(self.time)

    def image(self, i, sky, noise=True):
        image = self.field(i)

        background = sky + np.random.normal(scale=np.sqrt(sky), size=self.shape)
        read_noise = np.random.normal(scale=self.telescope.read_noise, size=self.shape)
        photon_noise = np.random.normal(scale=np.sqrt(image), size=self.shape)

        if noise:
            image += background + read_noise + photon_noise

        return image

    def set_star(self, i, position, diff_flux=None):
        self.target = i
        self.positions[i, :, :] = np.repeat(np.array(position)[:, np.newaxis], len(self.time), axis=1)
        if diff_flux is not None:
            peak = self.fluxes[i, :].mean()
            self.fluxes[i, :] = peak * diff_flux

    def set_target(self, i, diff_flux=None):
        self.target = i
        self.set_star(i, np.array(self.shape) / 2, diff_flux)

    def plot(self, n, photon_noise=True, atmosphere=True, **kwargs):
        fluxes = self.fluxes * (self.atmosphere[np.newaxis, :] if atmosphere else 1)
        viz.plot_lcs([(self.time, np.random.normal(f, np.sqrt(f) if photon_noise else 0, size=len(self.time))) for f in
                      fluxes[0:n]], **kwargs)

    def clean_around_target(self, radius):
        close_by = np.setdiff1d(np.argwhere(
            np.array(distances(self.positions[:, :, 0].T, self.positions[self.target, :, 0])) < radius).flatten(),
                                self.target)
        self.remove_stars(close_by)

    def save_fits(self, destination, calibration=False, verbose=True):

        progress = lambda x: tqdm(x) if verbose else x

        if not path.exists(destination):
            os.makedirs(destination)

        for i, time in enumerate(progress(self.time)):
            date = Time(datetime(2020, 3, 1, int(i / 60), i % 60)).to_value("fits")
            im = self.image(i, 300)
            fits_image(im,
                       {'TELESCOP': self.telescope.name, 'JD': time, 'DATE-OBS': date, "FILTER": "a"},
                       path.join(destination, f"fake-im-{i}.fits"))

        if calibration:
            fits_image(np.zeros_like(im),
                       {'TELESCOP': self.telescope.name, 'JD': time, 'DATE-OBS': date, "IMAGETYP": "dark"},
                       path.join(destination, f"fake-dark.fits"))

            fits_image(np.zeros_like(im),
                       {'TELESCOP': self.telescope.name, 'JD': time, 'DATE-OBS': date, "IMAGETYP": "bias"},
                       path.join(destination, f"fake-C001-bias.fits"))

            for i in range(0, 4):
                fits_image(np.ones_like(im),
                           {'TELESCOP': self.telescope.name, 'JD': time, 'DATE-OBS': date, "IMAGETYP": "flat", "FILTER": "a"},
                           path.join(destination, f"fake-flat-{i}.fits"))


def observation_to_model(time, t0=0.1, r=0.06417):
    # transit signal
    # lc = xo_lightcurve(time, period=0.7, r=7e-2, t0=0.1, plot=True)
    flux = models.transit(time, 0.1, 0.06417, 5.2e-3, c=20).flatten() + 1 
    error = 2e-3
    flux += error*np.random.randn(len(time))

    # Sky correlated signal
    sky = sim_signal(time, w=0.2)
    sky_flux = 5*sky + 0.35*sky**2
    sky_flux -= sky_flux.mean()

    # dy correlated signal
    dy = sim_signal(time, w=0.1)
    dy_flux = 2*dy + 0.35*dy**2
    dy_flux -= dy_flux.mean()

    flux += dy_flux + sky_flux

    x = xr.Dataset(dict(
        diff_fluxes=xr.DataArray(flux[None, None, :], dims=("apertures", "star", "time")),
        diff_errors=xr.DataArray(error*np.ones_like(time)[None, None, :], dims=("apertures", "star", "time")),
        fluxes=xr.DataArray(np.ones_like(time)[None, None, :], dims=("apertures", "star", "time")),
        errors=xr.DataArray(np.ones_like(time)[None, None, :], dims=("apertures", "star", "time")),
        sky=xr.DataArray(sky, dims="time"),
        dy=xr.DataArray(dy, dims="time")
    ), attrs=dict(
        telescope="Saint-Ex",
        aperture=0,
        target=0,
    ), coords=dict(
        time=time
    ))
    
    return Observation(x)


try:
    import exoplanet as xo
except:
    pass


def xo_lightcurve(time, period=3, r=0.1, t0=0, plot=False):
    orbit = xo.orbits.KeplerianOrbit(period=0.7, t0=0.1)
    light_curve = xo.LimbDarkLightCurve([0.1, 0.4]).get_light_curve(orbit=orbit, r=r, t=time).eval() + 1

    if plot:
        plt.plot(time, light_curve, color="C0", lw=2)
        plt.ylabel("relative flux")
        plt.xlabel("time [days]")

    return light_curve.flatten()
