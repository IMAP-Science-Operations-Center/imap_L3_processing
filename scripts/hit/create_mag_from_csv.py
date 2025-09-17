import sys

import numpy as np
import spiceypy
from spacepy.pycdf import CDF


def create_mag(csv_path, l2_path, output_path, offset):
    with CDF(l2_path) as cdf:
        epoch = cdf["epoch"][...]
    epoch_len = len(epoch)
    csv_data = np.loadtxt(csv_path, delimiter=",", skiprows=1)

    mag_data = csv_data[offset:offset+epoch_len, -3:]
    mag_sclk = csv_data[offset:offset+epoch_len, 6]
    mag_epochs = [spiceypy.et2datetime(spiceypy.scs2e(-43, str(sclk))) for sclk in mag_sclk]
    print(mag_sclk[0])
    print(mag_epochs[:5])
    print(epoch[:5])

    with CDF(output_path, create=True) as output_cdf:
        output_cdf['epoch'] = epoch
        output_cdf['vectors'] = mag_data
        output_cdf['vectors'].attrs["FILLVAL"] = -1e31


if __name__ == '__main__':
    spiceypy.furnsh("spice_kernels/imap_sclk_0000.tsc")
    spiceypy.furnsh("spice_kernels/naif0012.tls")
    # USAGE: python create_mag_from_csv.py <path to csv> <path to l2 cdf> <output path> <offset>
    csv_path, l2_path, output_path, offset = sys.argv[1:]
    create_mag(csv_path, l2_path, output_path, int(offset))

    # python scripts/hit/create_mag_from_csv.py instrument_team_data/hit/hit_l1a_sample2_fakeMAG.csv tests/test_data/hit/imap_hit_l2_macropixel-intensity_20100106_v001.cdf tests/test_data/hit/mag_data_for_hit_20100106 900
    # python scripts/hit/create_mag_from_csv.py instrument_team_data/hit/hit_l1a_sample2_fakeMAG.csv tests/test_data/hit/imap_hit_l2_macropixel-intensity_20100107_v001.cdf tests/test_data/hit/mag_data_for_hit_20100107 2340