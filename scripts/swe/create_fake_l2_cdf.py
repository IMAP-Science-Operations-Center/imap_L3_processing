import traceback
from pathlib import Path

import numpy as np
import pyhdf
import pyhdf.SD
from spacepy.pycdf import CDF


def create_fake_l2_cdf(input_hdf: str, cdf_filename: str):
    hdf = pyhdf.SD.SD(input_hdf)
    counts = hdf.select("DNSWE_COUNT")[:]
    counts = np.moveaxis(counts, 3, 2).reshape((-1, 20, 30, 7))
    print(counts.shape)
    with CDF(cdf_filename, readonly=False) as cdf:
        energies = cdf["energy"][:20]
        energies = energies[:, np.newaxis, np.newaxis]
        psd = counts / np.square(energies)
        cdf_data = cdf["phase_space_density_spin_sector"][:]
        cdf["phase_space_density_spin_sector"][:] = np.zeros_like(cdf_data)
        cdf["phase_space_density_spin_sector"][:, :20, :, :] = psd[:6]

        flux = psd * energies
        cdf_data = cdf["flux_spin_sector"][:]
        cdf["flux_spin_sector"][:] = np.zeros_like(cdf_data)
        cdf["flux_spin_sector"][:, :20, :, :] = flux[:6]
        print(flux.shape)
        print(np.mean(flux[0], axis=(1, 2)))


if __name__ == "__main__":
    path = Path(__file__)
    hdf_path = path.parent.parent.parent / "instrument_team_data/swe/ACE_LV1_1999-159.swepam.hdf"
    cdf_file_path = path.parent.parent.parent / "tests" / "test_data" / "swe" / "imap_swe_l2_sci-with-ace-data_20240510_v002.cdf"

    try:
        create_fake_l2_cdf(str(hdf_path), str(cdf_file_path))
    except Exception as e:
        traceback.print_exc()
