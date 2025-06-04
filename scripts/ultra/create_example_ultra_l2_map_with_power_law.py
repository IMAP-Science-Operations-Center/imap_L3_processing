from datetime import datetime
from pathlib import Path

import numpy as np
from spacepy import pycdf
from spacepy.pycdf import CDF

from imap_l3_processing.constants import TT2000_EPOCH
from tests.test_helpers import get_test_data_path


def _create_example_ultra_l2_map_with_power_law(out_path: Path, parents: list[str] = None):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.unlink(missing_ok=True)
    nside = 2
    num_lon = 60
    lon = np.linspace(0, 360, 60)
    num_lat = 30
    lat = np.linspace(-90, 90, 30)
    number_of_energies = 24
    energies = np.linspace(1, 50, number_of_energies)
    ena_intensities = np.full((1, number_of_energies, num_lon, num_lat), -1e31)

    power_law_1 = np.vectorize(lambda e: 10 * np.power(e, -2))
    power_law_2 = np.vectorize(lambda e: 1.5 * np.power(e, -3.5))

    energy_breakpoint = 15  # 15 keV
    for i in range(num_lon):
        for j in range(num_lat):
            breakpoint_index = np.searchsorted(energies, energy_breakpoint, side="right")
            ena_intensities[0, :breakpoint_index, i, j] = power_law_1(energies[:breakpoint_index])
            ena_intensities[0, breakpoint_index:, i, j] = power_law_2(energies[breakpoint_index:])

    ena_intensities_delta = np.full_like(ena_intensities, 0.0001)

    with CDF(str(out_path), readonly=False, masterpath="") as cdf:
        cdf.new("ena_intensity", ena_intensities)
        cdf.new("exposure_factor", np.full_like(ena_intensities, 1))
        cdf.new("sensitivity", np.full_like(ena_intensities, 1))
        cdf.new("latitude", lat, recVary=False)
        cdf.new("latitude_delta", np.full_like(lat, 2), recVary=False)
        cdf.new("longitude", lon, recVary=False)
        cdf.new("longitude_delta", np.full_like(lon, 2), recVary=False)
        cdf.new("epoch", np.array([datetime.now()]), type=pycdf.const.CDF_TIME_TT2000.value)
        cdf.new("energy", energies, recVary=False)
        cdf.new("epoch_delta", np.array([1]))
        cdf.new("energy_delta_plus", np.full_like(energies, 0.1))
        cdf.new("energy_delta_minus", np.full_like(energies, 0.1))
        cdf.new("energy_label", [str(val) for val in energies]),
        cdf.new("obs_date",
                np.full_like(ena_intensities, (datetime.now() - TT2000_EPOCH).total_seconds() * 1e9),
                type=pycdf.const.CDF_TIME_TT2000.value)
        cdf.new("obs_date_range", np.full_like(ena_intensities, 1))
        cdf.new("solid_angle", np.full_like(ena_intensities, 1), recVary=False)
        cdf.new("ena_intensity_stat_unc", ena_intensities_delta)
        cdf.new("ena_intensity_sys_err", np.full_like(ena_intensities, 1))
        cdf.new("latitude_label", lat.astype(str), recVary=False)
        cdf.new("longitude_label", lon.astype(str), recVary=False)

        if parents is not None:
            cdf.attrs['Parents'] = parents

        for var in cdf:
            if cdf[var].type() == pycdf.const.CDF_TIME_TT2000.value:
                cdf[var].attrs['FILLVAL'] = datetime.fromisoformat("9999-12-31T23:59:59.999999999")
            elif cdf[var].type() == pycdf.const.CDF_INT8.value:
                cdf[var].attrs['FILLVAL'] = -9223372036854775808
            elif cdf[var].type() == pycdf.const.CDF_FLOAT.value or pycdf.const.CDF_DOUBLE.value:
                cdf[var].attrs['FILLVAL'] = -1e31


if __name__ == "__main__":
    _create_example_ultra_l2_map_with_power_law(get_test_data_path('ultra') / 'fake_ultra_map_data.cdf')
