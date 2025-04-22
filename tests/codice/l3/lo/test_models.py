import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from unittest.mock import Mock, sentinel

import numpy as np
from spacepy.pycdf import CDF

from imap_l3_processing.codice.l3.lo.models import CodiceLoL2Data, CodiceLoL3aPartialDensityDataProduct, \
    CodiceLoL2DirectEventData, \
    CodiceLoL2bPriorityRates, PriortyEvent, EnergyAndSpinAngle


class TestModels(unittest.TestCase):
    def test_lo_l2_sectored_intensities_read_from_cdf(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cdf_file_path = Path(tmpdir) / "test_cdf.cdf"
            rng = np.random.default_rng()
            with CDF(str(cdf_file_path), readonly=False, masterpath="") as cdf_file:
                epoch = np.array([datetime(2010, 1, 1), datetime(2010, 1, 2)])
                epoch_delta = np.repeat(len(epoch), 2)
                energy = np.geomspace(2, 1000)
                spin_sector = np.linspace(0, 360, 24)
                ssd_id = np.linspace(0, 360, 16)
                h_intensities = rng.random((len(epoch), len(energy), len(spin_sector), len(ssd_id)))
                he_intensities = rng.random((len(epoch), len(energy), len(spin_sector), len(ssd_id)))
                c4_intensities = rng.random((len(epoch), len(energy), len(spin_sector), len(ssd_id)))
                c5_intensities = rng.random((len(epoch), len(energy), len(spin_sector), len(ssd_id)))
                c6_intensities = rng.random((len(epoch), len(energy), len(spin_sector), len(ssd_id)))
                o5_intensities = rng.random((len(epoch), len(energy), len(spin_sector), len(ssd_id)))
                o6_intensities = rng.random((len(epoch), len(energy), len(spin_sector), len(ssd_id)))
                o7_intensities = rng.random((len(epoch), len(energy), len(spin_sector), len(ssd_id)))
                o8_intensities = rng.random((len(epoch), len(energy), len(spin_sector), len(ssd_id)))
                mg_intensities = rng.random((len(epoch), len(energy), len(spin_sector), len(ssd_id)))
                si_intensities = rng.random((len(epoch), len(energy), len(spin_sector), len(ssd_id)))
                fe_low_intensities = rng.random((len(epoch), len(energy), len(spin_sector), len(ssd_id)))
                fe_high_intensities = rng.random((len(epoch), len(energy), len(spin_sector), len(ssd_id)))

                cdf_file['epoch'] = epoch
                cdf_file['epoch_delta'] = epoch_delta
                cdf_file['energy'] = energy
                cdf_file['spin_sector'] = spin_sector
                cdf_file['ssd_id'] = ssd_id
                cdf_file['h_intensities'] = h_intensities
                cdf_file['he_intensities'] = he_intensities
                cdf_file['c4_intensities'] = c4_intensities
                cdf_file['c5_intensities'] = c5_intensities
                cdf_file['c6_intensities'] = c6_intensities
                cdf_file['o5_intensities'] = o5_intensities
                cdf_file['o6_intensities'] = o6_intensities
                cdf_file['o7_intensities'] = o7_intensities
                cdf_file['o8_intensities'] = o8_intensities
                cdf_file['mg_intensities'] = mg_intensities
                cdf_file['si_intensities'] = si_intensities
                cdf_file['fe_low_intensities'] = fe_low_intensities
                cdf_file['fe_high_intensities'] = fe_high_intensities

            result: CodiceLoL2Data = CodiceLoL2Data.read_from_cdf(cdf_file_path)
            np.testing.assert_array_equal(result.epoch, epoch)
            np.testing.assert_array_equal(result.epoch_delta, epoch_delta)
            np.testing.assert_array_equal(result.energy, energy)
            np.testing.assert_array_equal(result.spin_sector, spin_sector)
            np.testing.assert_array_equal(result.ssd_id, ssd_id)
            np.testing.assert_array_equal(result.h_intensities, h_intensities)
            np.testing.assert_array_equal(result.he_intensities, he_intensities)
            np.testing.assert_array_equal(result.he_intensities, he_intensities)
            np.testing.assert_array_equal(result.c4_intensities, c4_intensities)
            np.testing.assert_array_equal(result.c5_intensities, c5_intensities)
            np.testing.assert_array_equal(result.c6_intensities, c6_intensities)
            np.testing.assert_array_equal(result.o5_intensities, o5_intensities)
            np.testing.assert_array_equal(result.o6_intensities, o6_intensities)
            np.testing.assert_array_equal(result.o7_intensities, o7_intensities)
            np.testing.assert_array_equal(result.o8_intensities, o8_intensities)
            np.testing.assert_array_equal(result.mg_intensities, mg_intensities)
            np.testing.assert_array_equal(result.si_intensities, si_intensities)
            np.testing.assert_array_equal(result.fe_low_intensities, fe_low_intensities)

    def test_codice_lo_l2_direct_event_read_from_cdf(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cdf_file_path = Path(tmpdir) / "test_cdf.cdf"
            rng = np.random.default_rng()
            with CDF(str(cdf_file_path), readonly=False, masterpath="") as cdf_file:
                epoch = np.array([datetime(2011, 5, 5), datetime(2011, 5, 6)])
                cdf_file['epoch'] = epoch
                expected_event_num = np.arange(77)
                cdf_file['event_num'] = expected_event_num

                expected_values = {}
                for index in range(8):
                    expected_values.update(
                        {
                            f"P{index}_APDEnergy": rng.random((7, 10)),
                            f"P{index}_APDGain": rng.random((7, 10)),
                            f"P{index}_APD_ID": rng.random(7),
                            f"P{index}_DataQuality": rng.random((7, 10)),
                            f"P{index}_EnergyStep": rng.random((7, 10)),
                            f"P{index}_MultiFlag": rng.random(7),
                            f"P{index}_NumEvents": rng.random((7, 10)),
                            f"P{index}_PHAType": rng.random((7, 10)),
                            f"P{index}_SpinAngle": rng.random((7, 10)),
                            f"P{index}_TOF": rng.random((7, 10)),
                        })

                    cdf_file[f"P{index}_APDEnergy"] = expected_values[f"P{index}_APDEnergy"]
                    cdf_file[f"P{index}_APDGain"] = expected_values[f"P{index}_APDGain"]
                    cdf_file[f"P{index}_APD_ID"] = expected_values[f"P{index}_APD_ID"]
                    cdf_file[f"P{index}_DataQuality"] = expected_values[f"P{index}_DataQuality"]
                    cdf_file[f"P{index}_EnergyStep"] = expected_values[f"P{index}_EnergyStep"]
                    cdf_file[f"P{index}_MultiFlag"] = expected_values[f"P{index}_MultiFlag"]
                    cdf_file[f"P{index}_NumEvents"] = expected_values[f"P{index}_NumEvents"]
                    cdf_file[f"P{index}_PHAType"] = expected_values[f"P{index}_PHAType"]
                    cdf_file[f"P{index}_SpinAngle"] = expected_values[f"P{index}_SpinAngle"]
                    cdf_file[f"P{index}_TOF"] = expected_values[f"P{index}_TOF"]

            result: CodiceLoL2DirectEventData = CodiceLoL2DirectEventData.read_from_cdf(cdf_file_path)
            np.testing.assert_array_equal(result.epoch, epoch)
            np.testing.assert_array_equal(result.event_num, expected_event_num)

            # @formatter:off
            for index in range(8):
                priority_event = getattr(result, f"priority_event_{index}")
                np.testing.assert_array_equal(priority_event.apd_energy, expected_values[f"P{index}_APDEnergy"])
                np.testing.assert_array_equal(priority_event.apd_gain, expected_values[f"P{index}_APDGain"])
                np.testing.assert_array_equal(priority_event.apd_id, expected_values[f"P{index}_APD_ID"])
                np.testing.assert_array_equal(priority_event.data_quality,expected_values[f"P{index}_DataQuality"])
                np.testing.assert_array_equal(priority_event.energy_step, expected_values[f"P{index}_EnergyStep"])
                np.testing.assert_array_equal(priority_event.multi_flag, expected_values[f"P{index}_MultiFlag"])
                np.testing.assert_array_equal(priority_event.num_events, expected_values[f"P{index}_NumEvents"])
                np.testing.assert_array_equal(priority_event.pha_type, expected_values[f"P{index}_PHAType"])
                np.testing.assert_array_equal(priority_event.spin_angle, expected_values[f"P{index}_SpinAngle"])
                np.testing.assert_array_equal(priority_event.tof, expected_values[f"P{index}_TOF"])
            # @formatter:on

    def test_codice_lo_l2_direct_event_priority_events(self):
        expected_events = [Mock(), Mock(), Mock(), Mock(), Mock(), Mock(), Mock(), Mock()]
        event_data = CodiceLoL2DirectEventData(Mock(), Mock(), *expected_events)

        self.assertEqual(expected_events, event_data.priority_events)

    def test_calculate_total_number_of_events(self):
        energy_step = np.array([[1, 4, 4, 2, 0],
                                [4, 4, 1, 2, np.nan]])

        spin_angle = np.array([[0, 30, 30, 40, 0],
                               [30, 30, 50, 40, np.nan]])

        priority_event = PriortyEvent(np.array([]), np.array([]), np.array([]), np.array([]), energy_step,
                                      np.array([]), np.array([]), np.array([]), spin_angle, np.array([]))

        expected_total_events_by_energy_step_and_spin_angle = [
            {
                EnergyAndSpinAngle(energy=1, spin_angle=0): 1,
                EnergyAndSpinAngle(energy=4, spin_angle=30): 2,
                EnergyAndSpinAngle(energy=2, spin_angle=40): 1,
                EnergyAndSpinAngle(energy=0, spin_angle=0): 1,
            },
            {
                EnergyAndSpinAngle(energy=4, spin_angle=30): 2,
                EnergyAndSpinAngle(energy=1, spin_angle=50): 1,
                EnergyAndSpinAngle(energy=2, spin_angle=40): 1,
            }
        ]
        # print(np.stack((spin_angle, energy_step), axis=-1))

        self.assertEqual(expected_total_events_by_energy_step_and_spin_angle,
                         priority_event.total_events_binned_by_energy_step_and_spin_angle)

    def test_get_species(self):
        h_intensities = np.array([sentinel.h_intensities])
        he_intensities = np.array([sentinel.he_intensities])
        c4_intensities = np.array([sentinel.c4_intensities])
        c5_intensities = np.array([sentinel.c5_intensities])
        c6_intensities = np.array([sentinel.c6_intensities])
        o5_intensities = np.array([sentinel.o5_intensities])
        o6_intensities = np.array([sentinel.o6_intensities])
        o7_intensities = np.array([sentinel.o7_intensities])
        o8_intensities = np.array([sentinel.o8_intensities])
        mg_intensities = np.array([sentinel.mg_intensities])
        si_intensities = np.array([sentinel.si_intensities])
        fe_low_intensities = np.array([sentinel.fe_low_intensities])
        fe_high_intensities = np.array([sentinel.fe_high_intensities])

        l2_data_product = CodiceLoL2Data(Mock(), Mock(), Mock(), Mock(), Mock(), h_intensities, he_intensities,
                                         c4_intensities, c5_intensities, c6_intensities, o5_intensities, o6_intensities,
                                         o7_intensities, o8_intensities, mg_intensities, si_intensities,
                                         fe_low_intensities, fe_high_intensities)

        species_intensities = l2_data_product.get_species_intensities()

        np.testing.assert_array_equal(species_intensities['H+'], h_intensities)
        np.testing.assert_array_equal(species_intensities['He++'], he_intensities)
        np.testing.assert_array_equal(species_intensities['C+4'], c4_intensities)
        np.testing.assert_array_equal(species_intensities['C+5'], c5_intensities)
        np.testing.assert_array_equal(species_intensities['C+6'], c6_intensities)
        np.testing.assert_array_equal(species_intensities['O+5'], o5_intensities)
        np.testing.assert_array_equal(species_intensities['O+6'], o6_intensities)
        np.testing.assert_array_equal(species_intensities['O+7'], o7_intensities)
        np.testing.assert_array_equal(species_intensities['O+8'], o8_intensities)
        np.testing.assert_array_equal(species_intensities['Mg'], mg_intensities)
        np.testing.assert_array_equal(species_intensities['Si'], si_intensities)
        np.testing.assert_array_equal(species_intensities['Fe (low Q)'], fe_low_intensities)
        np.testing.assert_array_equal(species_intensities['Fe (high Q)'], fe_high_intensities)

    def test_codice_lo_l3a_partial_density_to_data_product(self):
        epoch_data = np.array([datetime.now()])

        input_data_product_kwargs = {
            "epoch": epoch_data,
            "epoch_delta": np.array([10]),
            "h_partial_density": np.array([15]),
            "he_partial_density": np.array([15]),
            "c4_partial_density": np.array([15]),
            "c5_partial_density": np.array([15]),
            "c6_partial_density": np.array([15]),
            "o5_partial_density": np.array([15]),
            "o6_partial_density": np.array([15]),
            "o7_partial_density": np.array([15]),
            "o8_partial_density": np.array([15]),
            "mg_partial_density": np.array([15]),
            "si_partial_density": np.array([15]),
            "fe_low_partial_density": np.array([15]),
            "fe_high_partial_density": np.array([15]),
        }

        data_product = CodiceLoL3aPartialDensityDataProduct(
            **input_data_product_kwargs
        )
        actual_data_product_variables = data_product.to_data_product_variables()

        for input_variable, actual_data_product_variable in zip(input_data_product_kwargs.items(),
                                                                actual_data_product_variables):
            input_name, expected_value = input_variable

            np.testing.assert_array_equal(actual_data_product_variable.value, getattr(data_product, input_name))
            self.assertEqual(input_name, actual_data_product_variable.name)

    def test_codice_lo_l1b_priority_rates_read_from_cdf(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            rng = np.random.default_rng()
            epoch = np.array([datetime(2025, 4, 18), datetime(2025, 4, 18)])
            energy = rng.random((7, 10))
            inst_az = rng.random((7, 10))
            spin_sector = rng.random((7, 10))
            energy_label = rng.random((7, 10))
            acquisition_times = rng.random((7, 10))
            counters = rng.random((7, 10))
            esa_sweep = rng.random((7, 10))
            hi_counters_aggregated_aggregated = rng.random((7, 10))
            hi_counters_singles_tcr = rng.random((7, 10))
            hi_counters_singles_ssdo = rng.random((7, 10))
            hi_counters_singles_stssd = rng.random((7, 10))
            hi_omni_h = rng.random((7, 10))
            hi_omni_he3 = rng.random((7, 10))
            hi_omni_he4 = rng.random((7, 10))
            hi_omni_c = rng.random((7, 10))
            hi_omni_o = rng.random((7, 10))
            hi_omni_ne_mg_si = rng.random((7, 10))
            hi_omni_fe = rng.random((7, 10))
            hi_omni_uh = rng.random((7, 10))
            hi_sectored_h = rng.random((7, 10))
            hi_sectored_he3he4 = rng.random((7, 10))
            hi_sectored_cno = rng.random((7, 10))
            hi_sectored_fe = rng.random((7, 10))
            lo_counters_aggregated_aggregated = rng.random((7, 10))
            lo_counters_singles_apd_singles = rng.random((7, 10))
            lo_sw_angular_hplus = rng.random((7, 10))
            lo_sw_angular_heplusplus = rng.random((7, 10))
            lo_sw_angular_oplus6 = rng.random((7, 10))
            lo_sw_angular_fe_loq = rng.random((7, 10))
            lo_nsw_angular_heplusplus = rng.random((7, 10))
            lo_sw_priority_p0_tcrs = rng.random((7, 10))
            lo_sw_priority_p1_hplus = rng.random((7, 10))
            lo_sw_priority_p2_heplusplus = rng.random((7, 10))
            lo_sw_priority_p3_heavies = rng.random((7, 10))
            lo_sw_priority_p4_dcrs = rng.random((7, 10))
            lo_nsw_priority_p5_heavies = rng.random((7, 10))
            lo_nsw_priority_p6_hplus_heplusplus = rng.random((7, 10))
            lo_sw_species_hplus = rng.random((7, 10))
            lo_sw_species_heplusplus = rng.random((7, 10))
            lo_sw_species_cplus4 = rng.random((7, 10))
            lo_sw_species_cplus5 = rng.random((7, 10))
            lo_sw_species_cplus6 = rng.random((7, 10))
            lo_sw_species_oplus5 = rng.random((7, 10))
            lo_sw_species_oplus6 = rng.random((7, 10))
            lo_sw_species_oplus7 = rng.random((7, 10))
            lo_sw_species_oplus8 = rng.random((7, 10))
            lo_sw_species_ne = rng.random((7, 10))
            lo_sw_species_mg = rng.random((7, 10))
            lo_sw_species_si = rng.random((7, 10))
            lo_sw_species_fe_loq = rng.random((7, 10))
            lo_sw_species_fe_hiq = rng.random((7, 10))
            lo_sw_species_heplus = rng.random((7, 10))
            lo_sw_species_cnoplus = rng.random((7, 10))
            lo_nsw_species_hplus = rng.random((7, 10))
            lo_nsw_species_heplusplus = rng.random((7, 10))
            lo_nsw_species_c = rng.random((7, 10))
            lo_nsw_species_o = rng.random((7, 10))
            lo_nsw_species_ne_si_mg = rng.random((7, 10))
            lo_nsw_species_fe = rng.random((7, 10))
            lo_nsw_species_heplus = rng.random((7, 10))
            lo_nsw_species_cnoplus = rng.random((7, 10))

            cdf_file_path = Path(tmpdir) / "test_cdf.cdf"
            with CDF(str(cdf_file_path), readonly=False, masterpath="") as cdf_file:
                cdf_file["epoch"] = epoch
                cdf_file["energy"] = energy
                cdf_file["inst_az"] = inst_az
                cdf_file["spin_sector"] = spin_sector
                cdf_file["energy_label"] = energy_label
                cdf_file["acquisition_times"] = acquisition_times
                cdf_file["counters"] = counters
                cdf_file["esa_sweep"] = esa_sweep
                cdf_file["hi_counters_aggregated_aggregated"] = hi_counters_aggregated_aggregated
                cdf_file["hi_counters_singles_tcr"] = hi_counters_singles_tcr
                cdf_file["hi_counters_singles_ssdo"] = hi_counters_singles_ssdo
                cdf_file["hi_counters_singles_stssd"] = hi_counters_singles_stssd
                cdf_file["hi_omni_h"] = hi_omni_h
                cdf_file["hi_omni_he3"] = hi_omni_he3
                cdf_file["hi_omni_he4"] = hi_omni_he4
                cdf_file["hi_omni_c"] = hi_omni_c
                cdf_file["hi_omni_o"] = hi_omni_o
                cdf_file["hi_omni_ne_mg_si"] = hi_omni_ne_mg_si
                cdf_file["hi_omni_fe"] = hi_omni_fe
                cdf_file["hi_omni_uh"] = hi_omni_uh
                cdf_file["hi_sectored_h"] = hi_sectored_h
                cdf_file["hi_sectored_he3he4"] = hi_sectored_he3he4
                cdf_file["hi_sectored_cno"] = hi_sectored_cno
                cdf_file["hi_sectored_fe"] = hi_sectored_fe
                cdf_file["lo_counters_aggregated_aggregated"] = lo_counters_aggregated_aggregated
                cdf_file["lo_counters_singles_apd_singles"] = lo_counters_singles_apd_singles
                cdf_file["lo_sw_angular_hplus"] = lo_sw_angular_hplus
                cdf_file["lo_sw_angular_heplusplus"] = lo_sw_angular_heplusplus
                cdf_file["lo_sw_angular_oplus6"] = lo_sw_angular_oplus6
                cdf_file["lo_sw_angular_fe_loq"] = lo_sw_angular_fe_loq
                cdf_file["lo_nsw_angular_heplusplus"] = lo_nsw_angular_heplusplus
                cdf_file["lo_sw_priority_p0_tcrs"] = lo_sw_priority_p0_tcrs
                cdf_file["lo_sw_priority_p1_hplus"] = lo_sw_priority_p1_hplus
                cdf_file["lo_sw_priority_p2_heplusplus"] = lo_sw_priority_p2_heplusplus
                cdf_file["lo_sw_priority_p3_heavies"] = lo_sw_priority_p3_heavies
                cdf_file["lo_sw_priority_p4_dcrs"] = lo_sw_priority_p4_dcrs
                cdf_file["lo_nsw_priority_p5_heavies"] = lo_nsw_priority_p5_heavies
                cdf_file["lo_nsw_priority_p6_hplus_heplusplus"] = lo_nsw_priority_p6_hplus_heplusplus
                cdf_file["lo_sw_species_hplus"] = lo_sw_species_hplus
                cdf_file["lo_sw_species_heplusplus"] = lo_sw_species_heplusplus
                cdf_file["lo_sw_species_cplus4"] = lo_sw_species_cplus4
                cdf_file["lo_sw_species_cplus5"] = lo_sw_species_cplus5
                cdf_file["lo_sw_species_cplus6"] = lo_sw_species_cplus6
                cdf_file["lo_sw_species_oplus5"] = lo_sw_species_oplus5
                cdf_file["lo_sw_species_oplus6"] = lo_sw_species_oplus6
                cdf_file["lo_sw_species_oplus7"] = lo_sw_species_oplus7
                cdf_file["lo_sw_species_oplus8"] = lo_sw_species_oplus8
                cdf_file["lo_sw_species_ne"] = lo_sw_species_ne
                cdf_file["lo_sw_species_mg"] = lo_sw_species_mg
                cdf_file["lo_sw_species_si"] = lo_sw_species_si
                cdf_file["lo_sw_species_fe_loq"] = lo_sw_species_fe_loq
                cdf_file["lo_sw_species_fe_hiq"] = lo_sw_species_fe_hiq
                cdf_file["lo_sw_species_heplus"] = lo_sw_species_heplus
                cdf_file["lo_sw_species_cnoplus"] = lo_sw_species_cnoplus
                cdf_file["lo_nsw_species_hplus"] = lo_nsw_species_hplus
                cdf_file["lo_nsw_species_heplusplus"] = lo_nsw_species_heplusplus
                cdf_file["lo_nsw_species_c"] = lo_nsw_species_c
                cdf_file["lo_nsw_species_o"] = lo_nsw_species_o
                cdf_file["lo_nsw_species_ne_si_mg"] = lo_nsw_species_ne_si_mg
                cdf_file["lo_nsw_species_fe"] = lo_nsw_species_fe
                cdf_file["lo_nsw_species_heplus"] = lo_nsw_species_heplus
                cdf_file["lo_nsw_species_cnoplus"] = lo_nsw_species_cnoplus

            result = CodiceLoL2bPriorityRates.read_from_cdf(cdf_file_path)

            np.testing.assert_array_equal(result.epoch, epoch)
            np.testing.assert_array_equal(result.energy, energy)
            np.testing.assert_array_equal(result.inst_az, inst_az)
            np.testing.assert_array_equal(result.spin_sector, spin_sector)
            np.testing.assert_array_equal(result.energy_label, energy_label)
            np.testing.assert_array_equal(result.acquisition_times, acquisition_times)
            np.testing.assert_array_equal(result.counters, counters)
            np.testing.assert_array_equal(result.esa_sweep, esa_sweep)
            np.testing.assert_array_equal(result.hi_counters_aggregated_aggregated,
                                          hi_counters_aggregated_aggregated)
            np.testing.assert_array_equal(result.hi_counters_singles_tcr, hi_counters_singles_tcr)
            np.testing.assert_array_equal(result.hi_counters_singles_ssdo, hi_counters_singles_ssdo)
            np.testing.assert_array_equal(result.hi_counters_singles_stssd, hi_counters_singles_stssd)
            np.testing.assert_array_equal(result.hi_omni_h, hi_omni_h)
            np.testing.assert_array_equal(result.hi_omni_he3, hi_omni_he3)
            np.testing.assert_array_equal(result.hi_omni_he4, hi_omni_he4)
            np.testing.assert_array_equal(result.hi_omni_c, hi_omni_c)
            np.testing.assert_array_equal(result.hi_omni_o, hi_omni_o)
            np.testing.assert_array_equal(result.hi_omni_ne_mg_si, hi_omni_ne_mg_si)
            np.testing.assert_array_equal(result.hi_omni_fe, hi_omni_fe)
            np.testing.assert_array_equal(result.hi_omni_uh, hi_omni_uh)
            np.testing.assert_array_equal(result.hi_sectored_h, hi_sectored_h)
            np.testing.assert_array_equal(result.hi_sectored_he3he4, hi_sectored_he3he4)
            np.testing.assert_array_equal(result.hi_sectored_cno, hi_sectored_cno)
            np.testing.assert_array_equal(result.hi_sectored_fe, hi_sectored_fe)
            np.testing.assert_array_equal(result.lo_counters_aggregated_aggregated,
                                          lo_counters_aggregated_aggregated)
            np.testing.assert_array_equal(result.lo_counters_singles_apd_singles, lo_counters_singles_apd_singles)
            np.testing.assert_array_equal(result.lo_sw_angular_hplus, lo_sw_angular_hplus)
            np.testing.assert_array_equal(result.lo_sw_angular_heplusplus, lo_sw_angular_heplusplus)
            np.testing.assert_array_equal(result.lo_sw_angular_oplus6, lo_sw_angular_oplus6)
            np.testing.assert_array_equal(result.lo_sw_angular_fe_loq, lo_sw_angular_fe_loq)
            np.testing.assert_array_equal(result.lo_nsw_angular_heplusplus, lo_nsw_angular_heplusplus)
            np.testing.assert_array_equal(result.lo_sw_priority_p0_tcrs, lo_sw_priority_p0_tcrs)
            np.testing.assert_array_equal(result.lo_sw_priority_p1_hplus, lo_sw_priority_p1_hplus)
            np.testing.assert_array_equal(result.lo_sw_priority_p2_heplusplus, lo_sw_priority_p2_heplusplus)
            np.testing.assert_array_equal(result.lo_sw_priority_p3_heavies, lo_sw_priority_p3_heavies)
            np.testing.assert_array_equal(result.lo_sw_priority_p4_dcrs, lo_sw_priority_p4_dcrs)
            np.testing.assert_array_equal(result.lo_nsw_priority_p5_heavies, lo_nsw_priority_p5_heavies)
            np.testing.assert_array_equal(result.lo_nsw_priority_p6_hplus_heplusplus,
                                          lo_nsw_priority_p6_hplus_heplusplus)
            np.testing.assert_array_equal(result.lo_sw_species_hplus, lo_sw_species_hplus)
            np.testing.assert_array_equal(result.lo_sw_species_heplusplus, lo_sw_species_heplusplus)
            np.testing.assert_array_equal(result.lo_sw_species_cplus4, lo_sw_species_cplus4)
            np.testing.assert_array_equal(result.lo_sw_species_cplus5, lo_sw_species_cplus5)
            np.testing.assert_array_equal(result.lo_sw_species_cplus6, lo_sw_species_cplus6)
            np.testing.assert_array_equal(result.lo_sw_species_oplus5, lo_sw_species_oplus5)
            np.testing.assert_array_equal(result.lo_sw_species_oplus6, lo_sw_species_oplus6)
            np.testing.assert_array_equal(result.lo_sw_species_oplus7, lo_sw_species_oplus7)
            np.testing.assert_array_equal(result.lo_sw_species_oplus8, lo_sw_species_oplus8)
            np.testing.assert_array_equal(result.lo_sw_species_ne, lo_sw_species_ne)
            np.testing.assert_array_equal(result.lo_sw_species_mg, lo_sw_species_mg)
            np.testing.assert_array_equal(result.lo_sw_species_si, lo_sw_species_si)
            np.testing.assert_array_equal(result.lo_sw_species_fe_loq, lo_sw_species_fe_loq)
            np.testing.assert_array_equal(result.lo_sw_species_fe_hiq, lo_sw_species_fe_hiq)
            np.testing.assert_array_equal(result.lo_sw_species_heplus, lo_sw_species_heplus)
            np.testing.assert_array_equal(result.lo_sw_species_cnoplus, lo_sw_species_cnoplus)
            np.testing.assert_array_equal(result.lo_nsw_species_hplus, lo_nsw_species_hplus)
            np.testing.assert_array_equal(result.lo_nsw_species_heplusplus, lo_nsw_species_heplusplus)
            np.testing.assert_array_equal(result.lo_nsw_species_c, lo_nsw_species_c)
            np.testing.assert_array_equal(result.lo_nsw_species_o, lo_nsw_species_o)
            np.testing.assert_array_equal(result.lo_nsw_species_ne_si_mg, lo_nsw_species_ne_si_mg)
            np.testing.assert_array_equal(result.lo_nsw_species_fe, lo_nsw_species_fe)
            np.testing.assert_array_equal(result.lo_nsw_species_heplus, lo_nsw_species_heplus)
            np.testing.assert_array_equal(result.lo_nsw_species_cnoplus, lo_nsw_species_cnoplus)
