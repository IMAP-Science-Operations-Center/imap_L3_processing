import argparse
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
from spacepy.pycdf import CDF

from imap_l3_processing.constants import FIVE_MINUTES_IN_NANOSECONDS

parser = argparse.ArgumentParser()
parser.add_argument("map_folder")

args = parser.parse_args()

folder = Path(args.map_folder)


def extract_data(lines: list[str]):
    start_row, dimension_string = lines[0][2:10].split(':')
    data_lines = lines[int(start_row):]
    data = np.fromstring('\t'.join(data_lines), dtype=float, sep='\t')
    return data.reshape(tuple(int(x) for x in dimension_string.split('x'))).T


fluxes = []
energies = []
variances = []

for file in folder.iterdir():
    if file.parts[-1].endswith("flux.txt") and not "mono" in file.parts[-1]:
        group_file_name_prefix = '-'.join(file.parts[-1].split('-')[:-1])

        with open(file.parent / (group_file_name_prefix + "-flux.txt"), "r") as flux_file:
            fluxes.append(extract_data(flux_file.readlines()))
        with open(file.parent / (group_file_name_prefix + "-fvar.txt"), "r") as variance_file:
            variances.append(extract_data(variance_file.readlines()))

rng = np.random.default_rng()
pathname = f"C://Users//Harrison//Development//imap_L3_processing//tests//test_data//{folder.name}.cdf"

with CDF(pathname, '') as cdf:
    cdf.col_major(True)

    flux_data = np.stack(fluxes, axis=-1)
    flux_data = np.array([flux_data])
    variance_data = np.stack(variances, axis=-1)
    variance_data = np.array([variance_data])

    # energies = np.unique(energies)
    energies = np.array([0.71, 1.11, 1.74, 2.73, 4.29])
    energies = energies[energies != 0]
    energy_data = np.stack(energies, axis=-1)

    data_shape = flux_data.shape

    epoch = np.array([datetime(2000, 1, 1)])
    bin_boundaries = np.arange(len(energy_data))
    counts = rng.random(data_shape)
    counts_uncertainty = rng.random(data_shape)
    epoch_delta = np.array([FIVE_MINUTES_IN_NANOSECONDS])
    exposure = rng.random(flux_data.shape[:-1])

    number_of_lats = flux_data.shape[2]
    number_of_lons = flux_data.shape[1]
    lat_delta = 180 / number_of_lats
    lon_delta = 360 / number_of_lons
    lat = np.arange(number_of_lats) * lat_delta + (lat_delta / 2)
    lon = np.arange(number_of_lons) * lon_delta + (lon_delta / 2)

    sensitivity = rng.random(data_shape)

    cdf["Epoch"] = epoch
    cdf.new("bin", energy_data, recVary=False)
    cdf.new("bin_boundaries", bin_boundaries, recVary=False)
    cdf["counts"] = counts
    cdf["counts_uncertainty"] = counts_uncertainty
    cdf["epoch_delta"] = epoch_delta
    cdf["exposure"] = exposure
    cdf["flux"] = flux_data
    cdf.new("lat", lat, recVary=False)
    cdf.new("lon", lon, recVary=False)
    cdf["sensitivity"] = sensitivity
    cdf["variance"] = variance_data
