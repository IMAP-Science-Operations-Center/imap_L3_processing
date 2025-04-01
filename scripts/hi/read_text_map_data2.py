import argparse
import re
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
from spacepy.pycdf import CDF

from imap_l3_processing.constants import FIVE_MINUTES_IN_NANOSECONDS

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
            data[sub_folder.name][energy] = np.loadtxt(file, delimiter=",")
    assert (data['flux'].keys() == data['sigma'].keys())
    energy = np.array([key for key in data['flux'].keys()])
    flux = np.array([data['flux'][key] for key in sorted(data['flux'].keys())]).T
    sigma = np.array([data['sigma'][key] for key in sorted(data['sigma'].keys())]).T
    variance = sigma ** 2

    flux = flux.reshape(1, *flux.shape)
    variance = variance.reshape(1, *variance.shape)

    epoch = np.array([datetime.now()])
    bin_boundaries = np.full_like(energy, 1)
    counts = np.full_like(flux, 12)
    counts_uncertainty = np.full_like(flux, 13)
    epoch_delta = np.array([FIVE_MINUTES_IN_NANOSECONDS])
    exposure = np.full(flux.shape[:-1], 1)
    sensitivity = np.full_like(flux, 11)
    lat = np.arange(-88, 92, 4)
    lon = np.arange(0, 360, 4)

    pathname = f"C://Users//Harrison//Development//imap_L3_processing//tests//test_data//{folder.name}.cdf"
    with CDF(pathname, '') as cdf:
        cdf.col_major(True)
        cdf["Epoch"] = epoch
        cdf.new("bin", energy, recVary=False)
        cdf.new("bin_boundaries", bin_boundaries, recVary=False)
        cdf["counts"] = counts
        cdf["counts_uncertainty"] = counts_uncertainty
        cdf["epoch_delta"] = epoch_delta
        cdf["exposure"] = exposure
        cdf["flux"] = flux
        cdf.new("lat", lat, recVary=False)
        cdf.new("lon", lon, recVary=False)
        cdf["sensitivity"] = sensitivity
        cdf["variance"] = variance

        for var in cdf:
            cdf[var].attrs['FILLVAL'] = 1e-31

print("hello")
