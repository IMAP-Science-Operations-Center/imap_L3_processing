import unittest

from imap_l3_processing.codice.l3.lo.direct_events.science.mass_species_bin_lookup import MassSpeciesBinLookup, \
    SpeciesMassRange
from tests.test_helpers import get_test_data_path


class TestMassSpeciesBinLookup(unittest.TestCase):
    def test_reads_in_csv(self):
        csv_path = get_test_data_path('codice/species_mass_bins.csv')
        mass_species_bin_lookup = MassSpeciesBinLookup.read_from_csv(csv_path)

        # @formatter:off
        self.assertEqual(SpeciesMassRange(lower_mass=0.0,upper_mass=1.5, lower_mass_per_charge=0.7, upper_mass_per_charge=1.2), mass_species_bin_lookup.h_plus_nsw)
        self.assertEqual(SpeciesMassRange(lower_mass=2.5,upper_mass=5.0, lower_mass_per_charge=1.5, upper_mass_per_charge=2.5), mass_species_bin_lookup.he_plus_plus_nsw)
        self.assertEqual(SpeciesMassRange(lower_mass=10.0,upper_mass=14.0, lower_mass_per_charge=2.7, upper_mass_per_charge=3.2), mass_species_bin_lookup.c_plus_4_nsw)
        self.assertEqual(SpeciesMassRange(lower_mass=10.0,upper_mass=14.0, lower_mass_per_charge=2.2, upper_mass_per_charge=2.7), mass_species_bin_lookup.c_plus_5_nsw)
        self.assertEqual(SpeciesMassRange(lower_mass=10.0,upper_mass=14.0, lower_mass_per_charge=1.8, upper_mass_per_charge=2.2), mass_species_bin_lookup.c_plus_6_nsw)
        self.assertEqual(SpeciesMassRange(lower_mass=14.0,upper_mass=18.0, lower_mass_per_charge=3.0, upper_mass_per_charge=3.5), mass_species_bin_lookup.o_plus_5_nsw)
        self.assertEqual(SpeciesMassRange(lower_mass=14.0,upper_mass=18.0, lower_mass_per_charge=2.4, upper_mass_per_charge=3.0), mass_species_bin_lookup.o_plus_6_nsw)
        self.assertEqual(SpeciesMassRange(lower_mass=14.0,upper_mass=18.0, lower_mass_per_charge=2.1, upper_mass_per_charge=2.4), mass_species_bin_lookup.o_plus_7_nsw)
        self.assertEqual(SpeciesMassRange(lower_mass=14.0,upper_mass=18.0, lower_mass_per_charge=1.8, upper_mass_per_charge=2.1), mass_species_bin_lookup.o_plus_8_nsw)
        self.assertEqual(SpeciesMassRange(lower_mass=18.0,upper_mass=22.0, lower_mass_per_charge=2.2, upper_mass_per_charge=5.0), mass_species_bin_lookup.ne_nsw)
        self.assertEqual(SpeciesMassRange(lower_mass=22.0,upper_mass=26.0, lower_mass_per_charge=2.0, upper_mass_per_charge=5.0), mass_species_bin_lookup.mg_nsw)
        self.assertEqual(SpeciesMassRange(lower_mass=26.0,upper_mass=30.0, lower_mass_per_charge=2.0, upper_mass_per_charge=6.0), mass_species_bin_lookup.si_nsw)
        self.assertEqual(SpeciesMassRange(lower_mass=45.0,upper_mass=65.0, lower_mass_per_charge=3.2, upper_mass_per_charge=4.5), mass_species_bin_lookup.fe_lowq_nsw)
        self.assertEqual(SpeciesMassRange(lower_mass=45.0,upper_mass=65.0, lower_mass_per_charge=4.5, upper_mass_per_charge=10.0), mass_species_bin_lookup.fe_highq_nsw)
        self.assertEqual(SpeciesMassRange(lower_mass=2.5,upper_mass=5.0, lower_mass_per_charge=3.5, upper_mass_per_charge=5.0), mass_species_bin_lookup.he_plus_nsw)
        self.assertEqual(SpeciesMassRange(lower_mass=6.0,upper_mass=25.0, lower_mass_per_charge=10.2, upper_mass_per_charge=20.0), mass_species_bin_lookup.cno_plus_nsw)
        self.assertEqual(SpeciesMassRange(lower_mass=0.0,upper_mass=1.5, lower_mass_per_charge=0.7, upper_mass_per_charge=1.2), mass_species_bin_lookup.h_plus_sw)
        self.assertEqual(SpeciesMassRange(lower_mass=2.5,upper_mass=5.0, lower_mass_per_charge=1.5, upper_mass_per_charge=2.5), mass_species_bin_lookup.he_plus_plus_sw)
        self.assertEqual(SpeciesMassRange(lower_mass=10.0,upper_mass=14.0, lower_mass_per_charge=1.8, upper_mass_per_charge=3.2), mass_species_bin_lookup.c_sw)
        self.assertEqual(SpeciesMassRange(lower_mass=14.0,upper_mass=18.0, lower_mass_per_charge=1.8, upper_mass_per_charge=3.5), mass_species_bin_lookup.o_sw)
        self.assertEqual(SpeciesMassRange(lower_mass=18.0,upper_mass=22.0, lower_mass_per_charge=2.2, upper_mass_per_charge=5.0), mass_species_bin_lookup.ne_sw)
        self.assertEqual(SpeciesMassRange(lower_mass=22.0,upper_mass=30.0, lower_mass_per_charge=2.0, upper_mass_per_charge=5.0), mass_species_bin_lookup.si_mg_sw)
        self.assertEqual(SpeciesMassRange(lower_mass=45.0,upper_mass=65.0, lower_mass_per_charge=3.2, upper_mass_per_charge=10.0),mass_species_bin_lookup.fe_sw)
        self.assertEqual(SpeciesMassRange(lower_mass=2.5,upper_mass=5.0, lower_mass_per_charge=3.5, upper_mass_per_charge=5.0), mass_species_bin_lookup.he_plus_sw)
        self.assertEqual(SpeciesMassRange(lower_mass=6.0,upper_mass=25.0, lower_mass_per_charge=10.2, upper_mass_per_charge=20.0), mass_species_bin_lookup.cno_plus_sw)
        # @formatter:on
