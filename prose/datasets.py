import numpy as np
import matplotlib.pyplot as plt
import os
from astropy.io import fits
from os import path
from astropy.stats import gaussian_fwhm_to_sigma
from datetime import datetime
from astropy.time import Time
import pandas as pd


def psf_model(x, y, height, xo, yo, sx, sy, theta, m):
    dx = x - xo
    dy = y - yo
    a = (np.cos(theta)**2)/(2*sx**2) + (np.sin(theta)**2)/(2*sy**2)
    b = -(np.sin(2*theta))/(4*sx**2) + (np.sin(2*theta))/(4*sy**2)
    c = (np.sin(theta)**2)/(2*sx**2) + (np.cos(theta)**2)/(2*sy**2)
    psf = height * np.exp(-(a * dx ** 2 + 2 * b * dx * dy + c * dy ** 2))
    return psf + m


def create_image(peaks, stars, n):
    x, y = np.indices((n, n))
    image = np.zeros((n, n))

    background = np.random.normal(256, 6, size=(n, n))
    psf_x, psf_y = 2*gaussian_fwhm_to_sigma, 2.2*gaussian_fwhm_to_sigma

    image += background

    for star, peak in zip(stars, peaks):
        image += psf_model(x, y, peak, star[0], star[1], psf_x, psf_y, 0, 0)

    return image


def generate_prose_reduction_dataset(destination, n_images=80, moving=None):

    # moving_exmaple: [5, [0,40], [75, 60]]

    if not path.exists(destination):
        os.mkdir(destination)

    n = 80
    x, y = np.indices((n, n))

    # df = pd.read_csv("/Users/lionelgarcia/Code/prose_env/lena.csv", sep=";")
    # x = df["x"].values
    # y = df["y"].values
    # y -= np.mean(y)
    # y /= np.std(y)
    # y *= 4e-3
    # y += 1

    x = np.linspace(0, 1 / 24, n_images) + 2459000
    y = np.random.normal(1, 1.5e-3, size=len(x)) + np.sin(x * 2 * np.pi * 24 / 0.5) * 6e-3

    p = np.poly1d([0.3, 0.6, 0.8], r=True)
    sky = (p(x) / np.mean(p(x)))
    sky /= np.std(sky)

    telescope_name = "fake_telescope"

    stars = [
        [n / 2, n / 2],
        [n / 7, n / 5],
        [n / 1.2, n / 6],
        [n / 4, n / 1.3],
        [n / 1.5, n / 3],
        [n / 5.5, n / 1.2],
        [n / 10, n / 3.2],
        [n / 2.3, n / 5.6],
        [n / 2, n / 1.2],
        [n / 5., n / 1.4]
    ]

    all_peaks = np.array([
        16424,
        12524,
        12554,
        1324,
        140,
        940,
        214,
        30434,
        234,
        342
    ]).astype("float")

    all_peaks = all_peaks.T * np.random.normal(1, 1e-4, (len(x), len(all_peaks))).astype("float")
    all_peaks *= np.array(1 + 1e-2 * sky)[:, None]

    all_peaks[:, 0] *= y
    all_stars_coords = np.array([stars for i in range(n_images)])

    all_peaks.astype("int")
    if moving is not None:
        moving_star, p0, p1 = moving

        x_pos = np.linspace(p0[0], p1[0], n_images)
        y_pos = p0[1] + x_pos*(p1[1]-p0[1])/np.max(x_pos)

        all_stars_coords[:, moving_star, 0] = x_pos
        all_stars_coords[:, moving_star, 1] = y_pos

    for i, (peaks, stars_coords) in enumerate(zip(all_peaks, all_stars_coords)):
        im = create_image(peaks, stars_coords, n)
        hdu = fits.PrimaryHDU(im)
        hdu.header["TELESCOP"] = telescope_name
        hdu.header["EXPTIME"] = 1
        hdu.header["FILTER"] = "I+z"
        hdu.header["OBJECT"] = "prose"
        hdu.header["IMAGETYP"] = "light"
        hdu.header["AIRMASS"] = 1
        hdu.header["JD"] = x[i]
        hdu.header["RA"] = 12.84412
        hdu.header["DEC"] = -22.85886
        hdu.header["DATE-OBS"] = Time(datetime(2020, 3, 1, int(i / 60), i % 60)).to_value("fits")
        hdu.writeto(path.join(destination, "fake-C001-00{}.fits").format(hdu.header["DATE-OBS"]), overwrite=True)

    hdu = fits.PrimaryHDU(np.zeros_like(im))
    hdu.header["IMAGETYP"] = "dark"
    hdu.header["TELESCOP"] = telescope_name
    hdu.header["JD"] = x[0]
    hdu.header["EXPTIME"] = 1
    hdu.header["DATE-OBS"] = Time(datetime(2020, 3, 1, int(i / 60), i % 60)).to_value("fits")
    hdu.writeto(path.join(destination, "fake-C001-dark.fits"), overwrite=True)

    hdu = fits.PrimaryHDU(np.zeros_like(im))
    hdu.header["IMAGETYP"] = "bias"
    hdu.header["TELESCOP"] = telescope_name
    hdu.header["JD"] = x[0]
    hdu.header["EXPTIME"] = 1
    hdu.header["DATE-OBS"] = Time(datetime(2020, 3, 1, int(i / 60), i % 60)).to_value("fits")
    hdu.writeto(path.join(destination, "fake-C001-bias.fits"), overwrite=True)

    hdu = fits.PrimaryHDU(np.ones_like(im))
    hdu.header["IMAGETYP"] = "flat"
    hdu.header["TELESCOP"] = telescope_name
    hdu.header["JD"] = x[0]
    hdu.header["EXPTIME"] = 1
    hdu.header["FILTER"] = "I+z"
    hdu.header["DATE-OBS"] = Time(datetime(2020, 3, 1, int(i / 60), i % 60)).to_value("fits")
    hdu.writeto(path.join(destination, "fake-C001-flat0.fits"), overwrite=True)
    hdu.writeto(path.join(destination, "fake-C001-flat1.fits"), overwrite=True)
    hdu.writeto(path.join(destination, "fake-C001-flat2.fits"), overwrite=True)

    return path.abspath(destination)