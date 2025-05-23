import shutil
import zipfile
from pathlib import Path

import numpy as np

from imap_l3_processing.codice.l3.lo.constants import CODICE_LO_NUM_AZIMUTH_BINS, CODICE_LO_NUM_ESA_STEPS
from imap_l3_processing.codice.l3.lo.direct_events.science.mass_species_bin_lookup import MassSpeciesBinLookup
from tests.test_helpers import get_run_local_data_path, get_test_data_path

HEADER_TEXT = """# Efficiency Factors for the CoDICE-Lo Instrument for each energy (128 ESA steps) and position (24 values), energy is  assumed to be in ESA step order"""


def create_efficiency_lookup(mass_species_csv_path: Path, output_dir: Path = get_run_local_data_path("codice")) -> Path:
    output_ancillary_path = output_dir / "imap_codice_lo-efficiency-factors_20241110_v001.zip"
    mass_species_lookup = MassSpeciesBinLookup.read_from_csv(mass_species_csv_path)

    species_to_generate = mass_species_lookup._range_to_species["sw_species"] + \
                          mass_species_lookup._range_to_species["nsw_species"]

    temporary_output_dir = output_dir / "lo-efficiency-factors"
    temporary_output_dir.mkdir(parents=True, exist_ok=True)

    for species in species_to_generate:
        output_path = temporary_output_dir / f"{species}-efficiency.csv"
        open(output_path, "a").close()

        efficiency_data = np.ones((CODICE_LO_NUM_ESA_STEPS, CODICE_LO_NUM_AZIMUTH_BINS), dtype=np.float64)
        np.savetxt(output_path, efficiency_data, delimiter=",", header=HEADER_TEXT)

    with zipfile.ZipFile(output_ancillary_path, 'w') as zipf:
        for file in temporary_output_dir.iterdir():
            zipf.write(file, file.name)

    shutil.rmtree(temporary_output_dir)

    return output_ancillary_path


if __name__ == "__main__":
    create_efficiency_lookup(get_test_data_path("codice/imap_codice_lo-mass-species-bin-lookup_20241110_v001.csv"))
