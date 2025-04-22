import argparse
import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
from imap_processing.ena_maps.utils.spatial_utils import build_solid_angle_map
from spacepy import pycdf
from spacepy.pycdf import CDF

from imap_l3_processing.constants import FIVE_MINUTES_IN_NANOSECONDS, ONE_SECOND_IN_NANOSECONDS, SECONDS_PER_DAY

parser = argparse.ArgumentParser()
parser.add_argument("map_folder")

args = parser.parse_args()

parent = Path(args.map_folder)

data = defaultdict(lambda: {})
for folder in parent.iterdir():
    for sub_folder in folder.iterdir():
        for file in sub_folder.iterdir():
            energy_string = re.search(r"_(\d*p\d*)keV", file.name).group(1)
            energy = float(energy_string.replace('p', '.'))
            data[sub_folder.name][energy] = np.loadtxt(file, delimiter=",").T
    assert (data['flux'].keys() == data['sigma'].keys())
    energy = np.array([key for key in sorted(data['flux'].keys())])
    intensity = np.array([data['flux'][key] for key in sorted(data['flux'].keys())])
    ena_intensity = intensity.reshape(1, *intensity.shape)
    energy_delta_plus = np.full(energy.shape, 1.0)
    energy_delta_minus = np.full(energy.shape, 1.0)
    energy_label = energy.astype(str)
    ena_intensity_stat_unc = np.array([data['sigma'][key] for key in sorted(data['sigma'].keys())])[np.newaxis, ...]
    ena_intensity_sys_err = ena_intensity_stat_unc / 2

    epoch = np.array([datetime.now()])
    epoch_delta = np.array([FIVE_MINUTES_IN_NANOSECONDS])
    exposure_factor = np.full(ena_intensity.shape, 1.0)
    lat = np.arange(-88.0, 92.0, 4.0)
    lon = np.arange(0.0, 360.0, 4.0)
    pixel_index = np.arange(ena_intensity.shape[1])
    pixel_index_label = [f"Pixel index {x}" for x in pixel_index]

    obs_date = np.full(ena_intensity.shape, datetime.now())
    obs_date_range = np.full(ena_intensity.shape, ONE_SECOND_IN_NANOSECONDS * SECONDS_PER_DAY * 2)

    solid_angle = build_solid_angle_map(4)

    pathname = f"/Users/harrison/Development/imap_L3_processing/tests/test_data/ultra/fake_l2_maps/{folder.name}.cdf"
    Path(pathname).unlink(missing_ok=True)
    with CDF(pathname, '') as cdf:
        cdf.col_major(True)

        cdf.new("Epoch", epoch, type=pycdf.const.CDF_TIME_TT2000)
        cdf.new("energy", energy, recVary=False, type=pycdf.const.CDF_FLOAT)
        cdf.new("latitude", lat, recVary=False, type=pycdf.const.CDF_FLOAT)
        cdf.new("longitude", lon, recVary=False, type=pycdf.const.CDF_FLOAT)
        cdf.new("ena_intensity", ena_intensity, recVary=True, type=pycdf.const.CDF_FLOAT)
        cdf.new("ena_intensity_stat_unc", ena_intensity_stat_unc, recVary=True, type=pycdf.const.CDF_FLOAT)
        cdf.new("ena_intensity_sys_err", ena_intensity_sys_err, recVary=True, type=pycdf.const.CDF_FLOAT)
        cdf.new("exposure_factor", exposure_factor, recVary=True, type=pycdf.const.CDF_FLOAT)
        cdf.new("obs_date", obs_date, recVary=True, type=pycdf.const.CDF_TIME_TT2000)
        cdf.new("obs_date_range", obs_date_range, recVary=True, type=pycdf.const.CDF_INT8)
        cdf.new("solid_angle", solid_angle, recVary=True, type=pycdf.const.CDF_FLOAT)
        cdf.new("epoch_delta", epoch_delta, recVary=True, type=pycdf.const.CDF_INT8)
        cdf.new("energy_delta_plus", energy_delta_plus, recVary=False, type=pycdf.const.CDF_FLOAT)
        cdf.new("energy_delta_minus", energy_delta_minus, recVary=False, type=pycdf.const.CDF_FLOAT)
        cdf.new("energy_label", energy_label, recVary=False, type=pycdf.const.CDF_CHAR)
        cdf.new("pixel_index", pixel_index, recVary=False, type=pycdf.const.CDF_INT8)
        cdf.new("pixel_index_label", pixel_index_label, recVary=False, type=pycdf.const.CDF_CHAR)

        for metadata_var_names in ["pixel_index_label", "energy_label"]:
            cdf[metadata_var_names].attrs["VAR_TYPE"] = "metadata"

        for var in cdf:
            if cdf[var].type() == pycdf.const.CDF_TIME_TT2000.value:
                cdf[var].attrs['FILLVAL'] = datetime.fromisoformat("9999-12-31T23:59:59.999999999")
            elif cdf[var].type() == pycdf.const.CDF_INT8.value:
                cdf[var].attrs['FILLVAL'] = -9223372036854775808
            elif cdf[var].type() == pycdf.const.CDF_FLOAT.value:
                cdf[var].attrs['FILLVAL'] = -1e31
