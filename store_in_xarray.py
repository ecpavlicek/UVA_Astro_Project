import numpy as np
import xarray as xr

wl_min = 3200  # angstroms
wl_max = 5900  # angstroms
n_wl = 5193
wl = np.linspace(wl_min, wl_max, n_wl)

"""
spec = hdul["EXR1D"].data[4, :]
spec_err = hdul["EXR1D_ERROR"].data[4, :]
"""

spec = np.zeros((n_wl,))
spec_err = np.zeros((n_wl,))

# xr.Variable -> data=np.array, dimension(s)=list[str], attributes (e.g. name, units)=dict[str, str]
# xr.DataArray -> Variable + coordinate
# coordinate -> Variable but it represents the dimension
# xr.Dataset -> dict[str, DataArray]

wavelength: xr.Variable = xr.Variable("wavelength", wl, attrs={"units": "angstrom"})
spectrum: xr.DataArray = xr.DataArray(
    data=spec,  # np.array
    dims=["wavelength"],
    coords={"wavelength": wavelength},
    attrs={"units": ""},
    name="flux",
)
spectrum_error: xr.DataArray = xr.DataArray(
    data=spec_err,  # np.array
    dims=["wavelength"],
    coords={"wavelength": wavelength},
    attrs={"units": ""},
    name="flux_error",
)

dataset: xr.Dataset = xr.Dataset(
    data_vars={"flux": spectrum, "flux_error": spectrum_error},
    # coords={"wavelength": wl},
    attrs={"date_taken": "2021-11-01"},
)

dataset.to_netcdf("test_spectrum.nc")


spectrum_at_time_0: xr.DataArray = xr.DataArray(
    data=spec,  # np.array
    dims=["wavelength"],
    coords={"wavelength": wavelength},
    attrs={"units": ""},
    name="flux",
)
spectrum_error_at_time_0: xr.DataArray = xr.DataArray(
    data=spec_err,  # np.array
    dims=["wavelength"],
    coords={"wavelength": wavelength},
    attrs={"units": ""},
    name="flux_error",
)

spectrum_at_time_1: xr.DataArray = xr.DataArray(
    data=spec,  # np.array
    dims=["wavelength"],
    coords={"wavelength": wavelength},
    attrs={"units": ""},
    name="flux",
)
spectrum_error_at_time_1: xr.DataArray = xr.DataArray(
    data=spec_err,  # np.array
    dims=["wavelength"],
    coords={"wavelength": wavelength},
    attrs={"units": ""},
    name="flux_error",
)

spectrum_at_time_2: xr.DataArray = xr.DataArray(
    data=spec,  # np.array
    dims=["wavelength"],
    coords={"wavelength": wavelength},
    attrs={"units": ""},
    name="flux",
)
spectrum_error_at_time_2: xr.DataArray = xr.DataArray(
    data=spec_err,  # np.array
    dims=["wavelength"],
    coords={"wavelength": wavelength},
    attrs={"units": ""},
    name="flux_error",
)

time: xr.Variable = xr.Variable(dims=["time"], data=[0, 1, 2])

spectrum_by_time = xr.concat(
    [spectrum_at_time_0, spectrum_at_time_1, spectrum_at_time_2],
    dim="time",
).assign_coords(time=time)

spectrum_by_time.to_netcdf("test_spectra_by_time.nc")
