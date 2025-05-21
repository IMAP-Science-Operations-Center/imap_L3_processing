import datetime

import astropy_healpix.healpy as hp
import numpy as np
import xarray as xr
from imap_processing.ena_maps.utils.coordinates import CoordNames
from imap_processing.spice.time import str_to_et
from imap_processing.ultra.l1c.ultra_l1c_pset_bins import build_energy_bins
from spacepy import pycdf
from spacepy.pycdf import CDF

from imap_l3_processing.spice_wrapper import spiceypy
from tests.test_helpers import get_run_local_data_path

DEFAULT_RECT_SPACING_DEG_L1C = 0.5
DEFAULT_HEALPIX_NSIDE_L1C = 16


def create_example_ultra_l2_map(  # noqa: PLR0913
        nside: int = DEFAULT_HEALPIX_NSIDE_L1C,
        stripe_center_lat: int = 0,
        width_scale: float = 180.0,
        counts_scaling_params: tuple[int, float] = (100, 0.01),
        peak_exposure: float = 1000.0,
        timestr: str = "2025-09-01T00:00:00",
        head: str = "45",
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
    """
    _, energy_bin_midpoints, _ = build_energy_bins()
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
        -(lat_diff ** 2) / (2 * width_scale ** 2)
    )
    # Generate counts using binomial distribution
    rng = np.random.default_rng(seed=42)
    counts = np.array(
        [
            rng.binomial(n=counts_scaling_params[0], p=prob_scaling_factor)
            for _ in range(num_energy_bins)
        ]
    )

    # Generate exposure times using gaussian distribution
    exposure_time = peak_exposure * (prob_scaling_factor / prob_scaling_factor.max())
    exposure_time = np.expand_dims(exposure_time, axis=(0, 1))
    exposure_time = np.repeat(exposure_time, num_energy_bins, axis=1)

    # Ensure counts are integers
    counts = counts.astype(int)
    # add an epoch dimension
    counts = np.expand_dims(counts, axis=0)
    sensitivity = np.ones_like(counts)

    # Determine the epoch, which is TT time in nanoseconds since J2000 epoch
    tdb_et = str_to_et(timestr)
    tt_j2000ns = spiceypy.unitim(tdb_et, "ET", "TT") * 1e9

    pset_product = xr.Dataset(
        {
            "counts": (
                [
                    CoordNames.TIME.value,
                    CoordNames.ENERGY_ULTRA.value,
                    CoordNames.HEALPIX_INDEX.value,
                ],
                counts,
            ),
            "exposure_time": (
                [
                    CoordNames.TIME.value,
                    CoordNames.ENERGY_ULTRA.value,
                    CoordNames.HEALPIX_INDEX.value
                ],
                exposure_time,
            ),
            "sensitivity": (
                [
                    CoordNames.TIME.value,
                    CoordNames.ENERGY_ULTRA.value,
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
        },
        coords={
            CoordNames.TIME.value: [
                tt_j2000ns
            ],
            CoordNames.ENERGY_ULTRA.value: energy_bin_midpoints,
            CoordNames.HEALPIX_INDEX.value: pix_indices,
        },
        attrs={
            "Logical_file_id": (
                f"imap_ultra_l2_{head}sensor-map_{timestr[:4]}"
                f"{timestr[5:7]}{timestr[8:10]}-repointNNNNN_vNNN"
            ),
            "Logical_source": f"imap_ultra_l2_{head}sensor-map",
            "Data_version": "v001"
        },
    )

    pset_product["counts"].attrs["VAR_TYPE"] = "data"
    pset_product["exposure_time"].attrs["VAR_TYPE"] = "data"
    pset_product["sensitivity"].attrs["VAR_TYPE"] = "data"
    pset_product[CoordNames.AZIMUTH_L1C.value].attrs["VAR_TYPE"] = "data"
    pset_product[CoordNames.ELEVATION_L1C.value].attrs["VAR_TYPE"] = "data"

    pset_product.coords[CoordNames.TIME.value].attrs["VAR_TYPE"] = "support_data"
    pset_product.coords[CoordNames.ENERGY_ULTRA.value].attrs["VAR_TYPE"] = "support_data"
    pset_product.coords[CoordNames.HEALPIX_INDEX.value].attrs["VAR_TYPE"] = "support_data"

    return pset_product


def _write_ultra_l2_cdf_with_parents(out_path=get_run_local_data_path("ultra/fake_l2_maps/test_l2_map.cdf")):
    out_xarray = create_example_ultra_l2_map(nside=2)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.unlink(missing_ok=True)

    with CDF(str(out_path), readonly=False, masterpath="") as cdf:
        cdf.new("ena_intensity", out_xarray["counts"].values)
        cdf.new("exposure_factor", out_xarray["exposure_time"].values)
        cdf.new("sensitivity", out_xarray["sensitivity"].values)
        cdf.new("latitude", out_xarray[CoordNames.ELEVATION_L1C.value].values, recVary=False)
        cdf.new("longitude", out_xarray[CoordNames.AZIMUTH_L1C.value].values, recVary=False)
        cdf.new("epoch", out_xarray[CoordNames.TIME.value].values, recVary=False,
                type=pycdf.const.CDF_TIME_TT2000.value)
        cdf.new("energy", out_xarray[CoordNames.ENERGY_ULTRA.value].values, recVary=False)
        cdf.new("pixel_index", out_xarray[CoordNames.HEALPIX_INDEX.value].values, recVary=False)
        cdf.new("epoch_delta", np.array([0]))
        cdf.new("energy_delta_plus", np.full_like(out_xarray[CoordNames.ENERGY_ULTRA.value].values, 1))
        cdf.new("energy_delta_minus", np.full_like(out_xarray[CoordNames.ENERGY_ULTRA.value].values, 1))
        cdf.new("energy_label", [str(val) for val in out_xarray[CoordNames.ENERGY_ULTRA.value].values])
        cdf.new("obs_date", np.full(out_xarray["counts"].shape,
                                    spiceypy.unitim(datetime.datetime.now().timestamp(), "ET", "TT") * 1e9),
                type=pycdf.const.CDF_TIME_TT2000.value)
        cdf.new("obs_date_range", np.full_like(out_xarray["counts"].values, 1))
        cdf.new("solid_angle", np.full_like(out_xarray[CoordNames.HEALPIX_INDEX.value].values, 1))
        cdf.new("ena_intensity_stat_unc", np.full_like(out_xarray["counts"].values, 1))
        cdf.new("ena_intensity_sys_err", np.full_like(out_xarray["counts"].values, 1))
        cdf.new("pixel_index_label", [str(val) for val in out_xarray[CoordNames.HEALPIX_INDEX.value].values])

        for var in cdf:
            if cdf[var].type() == pycdf.const.CDF_TIME_TT2000.value:
                cdf[var].attrs['FILLVAL'] = datetime.datetime.fromisoformat("9999-12-31T23:59:59.999999999")
            elif cdf[var].type() == pycdf.const.CDF_INT8.value:
                cdf[var].attrs['FILLVAL'] = -9223372036854775808
            elif cdf[var].type() == pycdf.const.CDF_FLOAT.value or pycdf.const.CDF_DOUBLE.value:
                cdf[var].attrs['FILLVAL'] = -1e31


if __name__ == "__main__":
    _write_ultra_l2_cdf_with_parents()
