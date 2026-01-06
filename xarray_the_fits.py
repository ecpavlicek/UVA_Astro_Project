import os
from astropy.io import fits
import numpy as np
from scipy.signal import medfilt
from scipy.interpolate import interp1d
import xarray as xr

def extract_spectrum(hdul):
    hd = hdul['EXR'].header
    spec_data = hdul['EXR1D'].data
    spec_err_data = hdul['EXR1D_ERROR'].data

    # Find the spectrum with the highest mean flux
    mean_fluxes = [np.mean(s) for s in spec_data]
    best_index = np.argmax(mean_fluxes)

    spec = spec_data[best_index]
    spec_err = spec_err_data[best_index]
    #5400-10000, 3200-5900    Angstrom
    wl_min = 3200
    wl_max = 5900
    n_wl = len(spec)
    wl = np.linspace(wl_min, wl_max, n_wl)

    # Bad pixel correction
    spec_filt = medfilt(spec.astype(float), kernel_size=5)
    bad_pix_mask = np.where((np.abs(spec - spec_filt) > 3 * spec_err))
    good_pix_mask = np.ones(len(spec), dtype=bool)
    good_pix_mask[bad_pix_mask] = False

    interp_func = interp1d(wl[good_pix_mask], spec[good_pix_mask],
                           bounds_error=False, fill_value="extrapolate")
    spec_fixed = interp_func(wl)

    # Get observation time
    obs_time = hdul[0].header['HIERARCH EL DATE OBS']

    return wl, spec_fixed, spec_err, obs_time

def process_fits_folder_to_netcdf(folder_path, output_path="spectra.nc"):
    flux_list = []
    error_list = []
    time_list = []
    wavelength = None

    fits_files = sorted([f for f in os.listdir(folder_path) if f.lower().endswith('.fits')])

    for filename in fits_files:
        filepath = os.path.join(folder_path, filename)
        with fits.open(filepath) as hdul:
            wl, spec_fixed, spec_err, obs_time = extract_spectrum(hdul)
            if wavelength is None:
                wavelength = wl
            flux_list.append(spec_fixed)
            error_list.append(spec_err)
            time_list.append(obs_time)

    flux_arr = np.stack(flux_list)
    error_arr = np.stack(error_list)
    wavelength = np.array(wavelength)

    try:
        time_arr = np.array(time_list, dtype="datetime64[s]")
    except:
        time_arr = np.array(time_list, dtype=object)

    dataset = xr.Dataset(
        data_vars={
            "flux": (["time", "wavelength"], flux_arr,),
            "flux_error": (["time", "wavelength"], error_arr),
        },
        coords={
            "wavelength": ("wavelength", wavelength, {"units": "angstrom"}),
            "time": ("time", time_arr),
        },
        attrs={"source": "folder of FITS"},
    )

    dataset = dataset.sortby("time")
    dataset.to_netcdf(output_path, engine="h5netcdf")
    return output_path

process_fits_folder_to_netcdf(r"D:/Pavlicek, Emma/WD1202/blue/MODS1B", "WD1202_blue_mods1b_spectra.h5")
