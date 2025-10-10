import datetime
from pathlib import Path

import astropy_healpix.healpy as hp
import numpy as np
import spiceypy
import xarray as xr
from imap_processing.ena_maps.utils.coordinates import CoordNames
from imap_processing.quality_flags import ImapPSETUltraFlags
from imap_processing.spice.time import str_to_et
from imap_processing.ultra.l1c.ultra_l1c_pset_bins import build_energy_bins
from spacepy import pycdf
from spacepy.pycdf import CDF, Var

from tests.test_helpers import get_run_local_data_path

DEFAULT_RECT_SPACING_DEG_L1C = 0.5
DEFAULT_HEALPIX_NSIDE_L1C = 128

# TODO: Add ability to mock with/without energy dim to exposure_factor
# The Helio frame L1C will have the energy dimension, but the spacecraft frame will not.
def create_example_ultra_l1c_pset(
    nside: int = DEFAULT_HEALPIX_NSIDE_L1C,
    stripe_center_lat: int = 0,
    width_scale: float = 10.0,
    counts_scaling_params: tuple[int, float] = (100, 0.01),
    peak_exposure: float = 1000.0,
    timestr: str = "2025-01-01T00:00:00",
    head: str = "45",
    energy_dependent_exposure: bool = False,
) -> xr.Dataset:
    """
    Mock the L1C PSET product with recognizable but unrealistic counts.

    See the docstring for mock_l1c_pset_product_rectangular for more details about
    the structure of the dataset.
    The rectangular and Healpix mocked datasets are very similar in structure, though
    the actual values at a given latitude and longitude may be different. This is only
    meant to provide a recognizable structure for L2 testing purposes.

    The counts are generated along a stripe, centered at a given latitude.
    This stripe can be thought of as a 'vertical' line if the lon/az axis is plotted
    as the x-axis and the lat/el axis is plotted as the y-axis. See the figure below.

    ^  Elevation/Latitude
    |
    |                   00000000000000000000                     |
    |               0000000000000000000000000000                 |
    |           0000000000000000000000000000000000000            |
    |        0000000000000000000000000000000000000000000         |
    |      00000000000000000000000000000000000000000000000       |
    |     0000000000000000000000000000000000000000000000000       \
    |    222222222222222222222222222222222222222222222222222       \
    |    444444444444444444444444444444444444444444444444444        \
    |    666666666666666666666666666666666666666666666666666         |
    |     4444444444444444444444444444444444444444444444444         /
    |      22222222222222222222222222222222222222222222222         /
    |        0000000000000000000000000000000000000000000          /
    |           0000000000000000000000000000000000000            |
    |               0000000000000000000000000000                 |
    |                   00000000000000000000                     |
    --------------------------------------------------------->
    Azimuth/Longitude ->

    Fig. 1: Example of the '90' sensor head stripe on a HEALPix grid

    Parameters
    ----------
    nside : int, optional
        The HEALPix nside parameter (default is 128).
    stripe_center_lat : int, optional
        The center latitude of the stripe in degrees (default is 0).
    width_scale : float, optional
        The width of the stripe in degrees (default is 10 degrees).
    counts_scaling_params : tuple[int, float], optional
        The parameters for the binomial distribution of counts (default is (100, 0.01)).
        The 0th element is the number of trials to draw,
        the 1st element scales the probability of success for each trial.
    peak_exposure : float, optional
        The peak exposure time (default is 1000.0).
    timestr : str, optional
        The time string for the epoch (default is "2025-01-01T00:00:00").
    head : str, optional
        The sensor head (either '45' or '90') (default is '45').
    energy_dependent_exposure : bool, optional
        Whether the exposure time is energy dependent (default is False).
        If True, the exposure time will have an additional energy dimension.
        All the exposure times will be the same for each energy bin.
    """
    energy_intervals, energy_bin_midpoints, _ = build_energy_bins()
    energy_bin_delta = np.diff(energy_intervals, axis=1).squeeze()
    num_energy_bins = len(energy_bin_midpoints)
    npix = hp.nside2npix(nside)
    counts = np.zeros(npix)
    exposure_time = np.zeros(npix)

    # Get latitude for each healpix pixel
    pix_indices = np.arange(npix)
    lon_pix, lat_pix = hp.pix2ang(nside, pix_indices, lonlat=True)

    counts = np.zeros(shape=(num_energy_bins, npix))

    # Calculate probability based on distance from target latitude
    lat_diff = np.abs(lat_pix - stripe_center_lat)
    prob_scaling_factor = counts_scaling_params[1] * np.exp(
        -(lat_diff**2) / (2 * width_scale**2)
    )
    # Generate counts using binomial distribution
    rng = np.random.default_rng(seed=42)
    counts = np.array(
        [
            rng.binomial(n=counts_scaling_params[0], p=prob_scaling_factor)
            for _ in range(num_energy_bins)
        ]
    )

    # Generate exposure times using gaussian distribution, but wider
    prob_scaling_factor_exptime = counts_scaling_params[1] * np.exp(
        -(lat_diff**2) / (2 * (3 * width_scale) ** 2)
    )
    exposure_time = (
        peak_exposure
        * (prob_scaling_factor_exptime / prob_scaling_factor_exptime.max())
    )[np.newaxis, :]

    # Exposure time/factor can optionally be energy dependent
    if energy_dependent_exposure:
        # Add energy dimension to exposure time as axis 1
        exposure_time = np.repeat(
            exposure_time[:, np.newaxis, :], num_energy_bins, axis=1
        )
        exposure_dims = [
            CoordNames.TIME.value,
            CoordNames.ENERGY_ULTRA_L1C.value,
            CoordNames.HEALPIX_INDEX.value,
        ]
    else:
        exposure_dims = [
            CoordNames.TIME.value,
            CoordNames.HEALPIX_INDEX.value,
        ]

    # Ensure counts are integers
    counts = counts.astype(int)
    # add an epoch dimension
    counts = np.expand_dims(counts, axis=0)
    sensitivity = np.ones_like(counts)

    # Determine the epoch, which is TT time in nanoseconds since J2000 epoch
    tdb_et = str_to_et(timestr)
    tt_j2000ns = spiceypy.unitim(tdb_et, "ET", "TT") * 1e9

    logical_source = f"imap_ultra_l1c_{head}sensor-spacecraftpset"
    logical_file_id = (
        f"{logical_source}_{timestr[:4]}{timestr[5:7]}{timestr[8:10]}-repointNNNNN_vNNN"
    )

    pset_product = xr.Dataset(
        {
            "counts": (
                [
                    CoordNames.TIME.value,
                    CoordNames.ENERGY_ULTRA_L1C.value,
                    CoordNames.HEALPIX_INDEX.value,
                ],
                counts,
            ),
            "background_rates": (
                [
                    CoordNames.TIME.value,
                    CoordNames.ENERGY_ULTRA_L1C.value,
                    CoordNames.HEALPIX_INDEX.value,
                ],
                np.full_like(counts, 0.05, dtype=float),
            ),
            "exposure_factor": (
                exposure_dims,  # special case: optionally energy dependent exposure
                exposure_time,
            ),
            "sensitivity": (
                [
                    CoordNames.TIME.value,
                    CoordNames.ENERGY_ULTRA_L1C.value,
                    CoordNames.HEALPIX_INDEX.value,
                ],
                sensitivity,
            ),
            CoordNames.AZIMUTH_L1C.value: (
                [CoordNames.HEALPIX_INDEX.value],
                lon_pix,
            ),
            CoordNames.ELEVATION_L1C.value: (
                [CoordNames.HEALPIX_INDEX.value],
                lat_pix,
            ),
            "energy_bin_delta": (
                [CoordNames.ENERGY_ULTRA_L1C.value],
                energy_bin_delta,
            ),
            "quality_flags": (
                [CoordNames.TIME.value, CoordNames.HEALPIX_INDEX.value],
                np.full((1, npix), ImapPSETUltraFlags.NONE.value, dtype=np.uint16),
            ),
        },
        coords={
            CoordNames.TIME.value: [
                tt_j2000ns,
            ],
            CoordNames.ENERGY_ULTRA_L1C.value: xr.DataArray(
                energy_bin_midpoints, dims=(CoordNames.ENERGY_ULTRA_L1C.value,)
            ),
            CoordNames.HEALPIX_INDEX.value: pix_indices,
        },
        attrs={
            "Logical_file_id": logical_file_id,
            "Logical_source": logical_source,
            "Data_version": "001",
        },
    )

    return pset_product


def _write_ultra_l1c_cdf_with_parents(
        out_path: Path = get_run_local_data_path("ultra/fake_l1c_psets/test_pset.cdf"),
        date: str = "2025-09-01T00:00:00"):
    out_xarray = create_example_ultra_l1c_pset(nside=1, timestr=date, energy_dependent_exposure=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.unlink(missing_ok=True)

    with CDF(str(out_path), readonly=False, masterpath="") as cdf:
        cdf.new("counts", out_xarray["counts"].values)
        cdf.new("exposure_factor", out_xarray["exposure_factor"].values)
        cdf.new("background_rates", np.full_like(out_xarray["counts"], 0.001))
        cdf.new("observation_time", np.full_like(out_xarray["counts"], 1))
        cdf.new("quality_flags", out_xarray["quality_flags"].values)
        cdf.new("sensitivity", out_xarray["sensitivity"].values)
        cdf.new(CoordNames.ELEVATION_L1C.value, out_xarray[CoordNames.ELEVATION_L1C.value].values, recVary=False)
        cdf.new(CoordNames.AZIMUTH_L1C.value, out_xarray[CoordNames.AZIMUTH_L1C.value].values, recVary=False)
        cdf.new("epoch", out_xarray[CoordNames.TIME.value].values, type=pycdf.const.CDF_TIME_TT2000.value)
        cdf.new(CoordNames.ENERGY_ULTRA_L1C.value, out_xarray[CoordNames.ENERGY_ULTRA_L1C.value].values, recVary=False)
        cdf.new("energy_bin_delta", np.full_like(out_xarray[CoordNames.ENERGY_ULTRA_L1C.value].values, 1),
                recVary=False)
        cdf.new(CoordNames.HEALPIX_INDEX.value, out_xarray[CoordNames.HEALPIX_INDEX.value].values, recVary=False)

        cdf[CoordNames.ENERGY_ULTRA_L1C.value].attrs["VAR_TYPE"] = "support_data"
        cdf[CoordNames.HEALPIX_INDEX.value].attrs["VAR_TYPE"] = "support_data"
        _add_depends(cdf["counts"], [CoordNames.ENERGY_ULTRA_L1C.value, CoordNames.HEALPIX_INDEX.value], "epoch")
        _add_depends(cdf["exposure_factor"], [CoordNames.ENERGY_ULTRA_L1C.value, CoordNames.HEALPIX_INDEX.value],
                     "epoch")
        _add_depends(cdf["sensitivity"], [CoordNames.ENERGY_ULTRA_L1C.value, CoordNames.HEALPIX_INDEX.value], "epoch")
        _add_depends(cdf["background_rates"], [CoordNames.ENERGY_ULTRA_L1C.value, CoordNames.HEALPIX_INDEX.value],
                     "epoch")
        _add_depends(cdf["observation_time"], [CoordNames.ENERGY_ULTRA_L1C.value, CoordNames.HEALPIX_INDEX.value],
                     "epoch")
        _add_depends(cdf["quality_flags"], [CoordNames.ENERGY_ULTRA_L1C.value, CoordNames.HEALPIX_INDEX.value], "epoch")
        _add_depends(cdf["energy_bin_delta"], [CoordNames.ENERGY_ULTRA_L1C.value])

        for var in cdf:
            if cdf[var].type() == pycdf.const.CDF_TIME_TT2000.value:
                cdf[var].attrs['FILLVAL'] = datetime.datetime.fromisoformat("9999-12-31T23:59:59.999999999")
            elif cdf[var].type() == pycdf.const.CDF_INT8.value:
                cdf[var].attrs['FILLVAL'] = -9223372036854775808
            elif cdf[var].type() == pycdf.const.CDF_FLOAT.value or pycdf.const.CDF_DOUBLE.value:
                cdf[var].attrs['FILLVAL'] = -1e31


def _add_depends(var: Var, depends: list[str], depend_0: str = None):
    if depend_0 is not None:
        var.attrs[f"DEPEND_0"] = depend_0
    for i, dep in enumerate(depends):
        var.attrs[f"DEPEND_{i + 1}"] = dep


if __name__ == "__main__":
    _write_ultra_l1c_cdf_with_parents()
