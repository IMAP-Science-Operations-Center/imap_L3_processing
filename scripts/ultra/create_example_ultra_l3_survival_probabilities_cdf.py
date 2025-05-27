import re
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
from astropy_healpix import HEALPix
from spacepy import pycdf
from spacepy.pycdf import CDF

from tests.test_helpers import get_test_data_path


def create_survival_probabilities_file(glows_file_path: Path, date_for_file: datetime, cdf_file_path: Path):
    with open(glows_file_path) as input_data:
        energy_line = [line for line in input_data.readlines() if line.startswith("#energy_grid")]
        assert len(energy_line) == 1
        match = [float(x) for x in re.findall(r"\s+([0-9.]+)", energy_line[0])]
        energy_units = re.search(r'\[(.+)]', energy_line[0]).group(1)

        energies = np.array(match)

    spin_angle_and_survival_probabilities = np.loadtxt(glows_file_path)
    output_survival_probabilities = np.full((len(energies), 3072), 0, dtype=np.float64)
    healpix_index = spin_angle_and_survival_probabilities[:, 0]
    output_healpix_index = np.arange(0, 3072)
    healpix = HEALPix(nside=16)
    lons, lats = healpix.healpix_to_lonlat(output_healpix_index)

    latitude = lats.value
    longitude = lons.value
    skips = 0
    for i in range(0, 3072):
        if i in healpix_index:
            output_survival_probabilities[:, i] = spin_angle_and_survival_probabilities[i - skips, 3:-1].T
        else:
            skips += 1
            output_survival_probabilities[:, i] = 0

    # Removing last column as it appears to be extra
    survival_probabilities = spin_angle_and_survival_probabilities[:, 3:-1].T

    with CDF(str(cdf_file_path), '') as c:
        c.new("epoch", [date_for_file], pycdf.const.CDF_TIME_TT2000)
        c.new("epoch_delta", [12 * 60 * 60 * 1e9], pycdf.const.CDF_INT8)
        c.new("energy", energies, pycdf.const.CDF_FLOAT, recVary=False)
        c['energy'].attrs["UNITS"] = energy_units
        c.new("latitude", latitude, pycdf.const.CDF_FLOAT, recVary=False)
        c.new("longitude", longitude, pycdf.const.CDF_FLOAT, recVary=False)
        c.new("healpix_index", output_healpix_index, pycdf.const.CDF_INT2, recVary=False)
        c.new("probability_of_survival", np.array(output_survival_probabilities)[np.newaxis, ...],
              pycdf.const.CDF_FLOAT)

        c["energy"].attrs["FILLVAL"] = -1e31
        c["latitude"].attrs["FILLVAL"] = -1e31
        c["longitude"].attrs["FILLVAL"] = -1e31
        c["healpix_index"].attrs["FILLVAL"] = -32768
        c["probability_of_survival"].attrs["FILLVAL"] = -1e31

    return survival_probabilities


if __name__ == "__main__":
    path = Path(__file__)
    input_file_path = path.parent.parent.parent / "instrument_team_data" / "glows" / "probSur.Imap.Ul.V0_2009.000.dat"

    start_date = datetime(year=2025, month=9, day=1)
    num_psets_to_generate = 1

    for i in range(num_psets_to_generate):
        date_to_set = start_date + timedelta(days=i)
        filename = f"imap_glows_l3e_survival-probabilities-ultra_{date_to_set.strftime('%Y%m%d')}_v001.cdf"
        cdf_file_path = get_test_data_path(f"ultra/fake_l3e_survival_probabilities/{filename}")

        cdf_file_path.unlink(missing_ok=True)
        create_survival_probabilities_file(input_file_path, date_to_set, cdf_file_path)
