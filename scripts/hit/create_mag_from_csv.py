import sys

import numpy as np
from spacepy.pycdf import CDF


def create_mag(csv_path, l2_path, output_path, offset):
    with CDF(l2_path) as cdf:
        epoch = cdf["epoch"][...]
    epoch_len = len(epoch)
    csv_data = np.loadtxt(csv_path, delimiter=",", skiprows=1)

    mag_data = csv_data[offset:offset+epoch_len, -3:]

    with CDF(output_path, create=True) as output_cdf:
        output_cdf['epoch'] = epoch
        output_cdf['vectors'] = mag_data
        output_cdf['vectors'].attrs["FILLVAL"] = -1e31


if __name__ == '__main__':
    # USAGE: python create_mag_from_csv.py <path to csv> <path to l2 cdf> <output path> <offset>
    csv_path, l2_path, output_path, offset = sys.argv[1:]
    create_mag(csv_path, l2_path, output_path, int(offset))