from pathlib import Path

import numpy as np

from imap_l3_processing.codice.l3.lo.constants import CODICE_LO_NUM_AZIMUTH_BINS, CODICE_LO_NUM_ESA_STEPS
from imap_l3_processing.codice.l3.lo.direct_events.science.mass_species_bin_lookup import MassSpeciesBinLookup
from tests.test_helpers import get_run_local_data_path, get_test_data_path

HEADER_TEXT = """# Efficiency Factors for the CoDICE-Lo Instrument for each energy (128 ESA steps) and position (24 values), energy is  assumed to be in ESA step order"""


def create_efficiency_lookup(species: str, output_dir: Path = get_run_local_data_path("codice")) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"imap_codice_lo-{species}-efficiency_20241110_v001.csv"
    open(output_path, "a").close()

    efficiency_data = np.ones((CODICE_LO_NUM_ESA_STEPS, CODICE_LO_NUM_AZIMUTH_BINS), dtype=np.float64)
    np.savetxt(output_path, efficiency_data, delimiter=",", header=HEADER_TEXT)

    return output_path


if __name__ == "__main__":

    mass_species_bin_path = get_test_data_path(
        'codice/imap_codice_lo-mass-species-bin-lookup_20241110_v001.csv')
    for species in MassSpeciesBinLookup.read_from_csv(mass_species_bin_path).species:
        print(create_efficiency_lookup(species))
