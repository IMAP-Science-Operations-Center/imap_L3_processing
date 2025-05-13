import numpy as np
from spacepy.pycdf import CDF

from imap_l3_processing.codice.l3.lo.direct_events.science.angle_lookup import SpinAngleLookup
from imap_l3_processing.codice.l3.lo.direct_events.science.mass_species_bin_lookup import MassSpeciesBinLookup, \
    EventDirection
from tests.test_helpers import get_test_data_path


def create_more_accurate_l3a_direct_events_cdf(template_cdf):
    mass_species_path = get_test_data_path("codice/species_mass_bins.csv")
    mass_species_bin_lookup = MassSpeciesBinLookup.read_from_csv(mass_species_path)

    spin_angle_lut = SpinAngleLookup()

    rng = np.random.default_rng()

    sw_apds = [1, 2, 3, 23, 24]
    nsw_apds = [a for a in range(1, 25) if a not in sw_apds]

    with CDF(str(template_cdf), readonly=False) as cdf:
        event_buffer_shape = cdf["mass"].shape
        event_direction = np.where(rng.random(event_buffer_shape) > 0.5, EventDirection.Sunward,
                                   EventDirection.NonSunward)

        sw_apd_id = rng.choice(sw_apds, size=event_buffer_shape)
        nsw_apd_id = rng.choice(nsw_apds, size=event_buffer_shape)
        cdf["apd_id"] = np.where(event_direction == EventDirection.Sunward, sw_apd_id, nsw_apd_id)

        cdf["spin_angle"] = rng.choice(spin_angle_lut.bin_centers, size=event_buffer_shape)

        num_sw_species = len(mass_species_bin_lookup._range_to_species["sw_species"])
        num_nsw_species = len(mass_species_bin_lookup._range_to_species["nsw_species"])

        sw_species = rng.integers(0, high=num_sw_species, size=event_buffer_shape)
        sw_lower_masses = mass_species_bin_lookup._range_to_species["sw_mass_ranges"][sw_species, 0]
        sw_upper_masses = mass_species_bin_lookup._range_to_species["sw_mass_ranges"][sw_species, -1]
        sw_masses = rng.random(event_buffer_shape) * (sw_upper_masses - sw_lower_masses) + sw_lower_masses

        sw_lower_mpc = mass_species_bin_lookup._range_to_species["sw_mass_per_charge_ranges"][sw_species, 0]
        sw_upper_mpc = mass_species_bin_lookup._range_to_species["sw_mass_per_charge_ranges"][sw_species, -1]
        sw_mpc = rng.random(event_buffer_shape) * (sw_upper_mpc - sw_lower_mpc) + sw_lower_mpc

        nsw_species = rng.integers(0, high=num_nsw_species, size=event_buffer_shape)
        nsw_lower_masses = mass_species_bin_lookup._range_to_species["nsw_mass_ranges"][nsw_species, 0]
        nsw_upper_masses = mass_species_bin_lookup._range_to_species["nsw_mass_ranges"][nsw_species, -1]
        nsw_masses = rng.random(event_buffer_shape) * (nsw_upper_masses - nsw_lower_masses) + nsw_lower_masses

        nsw_lower_mpc = mass_species_bin_lookup._range_to_species["nsw_mass_per_charge_ranges"][nsw_species, 0]
        nsw_upper_mpc = mass_species_bin_lookup._range_to_species["nsw_mass_per_charge_ranges"][nsw_species, -1]
        nsw_mpc = rng.random(event_buffer_shape) * (nsw_upper_mpc - nsw_lower_mpc) + nsw_lower_mpc

        cdf["mass"] = np.where(event_direction == EventDirection.Sunward, sw_masses, nsw_masses)
        cdf["mass_per_charge"] = np.where(event_direction == EventDirection.Sunward, sw_mpc, nsw_mpc)

        return template_cdf
