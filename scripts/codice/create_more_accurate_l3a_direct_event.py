from pathlib import Path

import numpy as np
from spacepy.pycdf import CDF

from imap_l3_processing.codice.l3.lo.direct_events.science.angle_lookup import SpinAngleLookup
from imap_l3_processing.codice.l3.lo.direct_events.science.energy_lookup import EnergyLookup
from imap_l3_processing.codice.l3.lo.direct_events.science.mass_species_bin_lookup import MassSpeciesBinLookup
from tests.test_helpers import get_test_data_path


def create_more_accurate_l3a_direct_events_cdf(template_cdf: Path):
    mass_species_path = get_test_data_path("codice/imap_codice_lo-mass-species-bin-lookup_20241110_v001.csv")
    mass_species_bin_lookup = MassSpeciesBinLookup.read_from_csv(mass_species_path)

    energy_lookup = EnergyLookup.read_from_csv(
        get_test_data_path('codice/imap_codice_lo-esa-to-energy-per_charge_20241110_v001.csv'))
    energies = energy_lookup.bin_centers
    
    spin_angle_lut = SpinAngleLookup()

    rng = np.random.default_rng()

    with CDF(str(template_cdf), readonly=False) as cdf:
        event_buffer_shape = cdf["mass"].shape

        position = rng.integers(1, 25, size=event_buffer_shape)
        cdf["position"] = position

        cdf["spin_angle"] = rng.choice(spin_angle_lut.bin_centers, size=event_buffer_shape)

        species = rng.integers(0, mass_species_bin_lookup.get_num_species(), size=event_buffer_shape)

        mass_ranges = np.array(mass_species_bin_lookup.mass_ranges)
        lower_masses = mass_ranges[species, 0]
        upper_masses = mass_ranges[species, -1]
        masses = rng.random(event_buffer_shape) * (upper_masses - lower_masses) + lower_masses

        mass_per_charge_ranges = np.array(mass_species_bin_lookup.mass_per_charge)
        lower_mpc = mass_per_charge_ranges[species, 0]
        upper_mpc = mass_per_charge_ranges[species, -1]
        mpc = rng.random(event_buffer_shape) * (upper_mpc - lower_mpc) + lower_mpc

        cdf["mass"] = masses
        cdf["mass_per_charge"] = mpc
        cdf["event_energy"] = rng.choice(energies, size=event_buffer_shape)
        cdf["num_events"] = np.full_like(cdf["num_events"], 1000)

        return template_cdf


if __name__ == "__main__":
    create_more_accurate_l3a_direct_events_cdf(
        get_test_data_path("codice/imap_codice_l3a_lo-direct-events_20241110_v000.cdf"))
