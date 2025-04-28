import tempfile
import unittest
from dataclasses import fields
from datetime import datetime
from pathlib import Path
from unittest.mock import Mock

import numpy as np
from spacepy.pycdf import CDF

from imap_l3_processing.codice.l3.lo.models import CodiceLoL2SWSpeciesData, CodiceLoL3aPartialDensityDataProduct, \
    CodiceLoL2DirectEventData, \
    CodiceLoL2bPriorityRates, PriorityEvent, EnergyAndSpinAngle, CodiceLoL3aDirectEventDataProduct


class TestModels(unittest.TestCase):
    def test_lo_l2_sw_species_read_from_cdf(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cdf_file_path = Path(tmpdir) / "test_cdf.cdf"
            rng = np.random.default_rng()
            with CDF(str(cdf_file_path), readonly=False, masterpath="") as cdf_file:
                epoch = np.array([datetime(2010, 1, 1), datetime(2010, 1, 2)])
                epoch_delta_minus = rng.random(len(epoch))
                epoch_delta_plus = rng.random(len(epoch))
                energy_table = np.geomspace(2, 1000)
                spin_sector = np.linspace(0, 360, 24)
                hplus = rng.random((len(epoch), len(energy_table), len(spin_sector)))
                heplus = rng.random((len(epoch), len(energy_table), len(spin_sector)))
                heplusplus = rng.random((len(epoch), len(energy_table), len(spin_sector)))
                ne = rng.random((len(epoch), len(energy_table), len(spin_sector)))
                cplus4 = rng.random((len(epoch), len(energy_table), len(spin_sector)))
                cplus5 = rng.random((len(epoch), len(energy_table), len(spin_sector)))
                cplus6 = rng.random((len(epoch), len(energy_table), len(spin_sector)))
                oplus5 = rng.random((len(epoch), len(energy_table), len(spin_sector)))
                oplus6 = rng.random((len(epoch), len(energy_table), len(spin_sector)))
                oplus7 = rng.random((len(epoch), len(energy_table), len(spin_sector)))
                oplus8 = rng.random((len(epoch), len(energy_table), len(spin_sector)))
                mg = rng.random((len(epoch), len(energy_table), len(spin_sector)))
                si = rng.random((len(epoch), len(energy_table), len(spin_sector)))
                cnoplus = rng.random((len(epoch), len(energy_table)))
                fe_loq = rng.random((len(epoch), len(energy_table), len(spin_sector)))
                fe_hiq = rng.random((len(epoch), len(energy_table), len(spin_sector)))
                data_quality = rng.random(len(epoch))
                spin_sector_index = np.array([1])

                cdf_file['epoch'] = epoch
                cdf_file['epoch_delta_minus'] = epoch_delta_minus
                cdf_file['epoch_delta_plus'] = epoch_delta_plus
                cdf_file['energy_table'] = energy_table
                cdf_file['spin_sector'] = spin_sector
                cdf_file['hplus'] = hplus
                cdf_file['heplusplus'] = heplusplus
                cdf_file['heplus'] = heplus
                cdf_file['ne'] = ne
                cdf_file['cplus4'] = cplus4
                cdf_file['cplus5'] = cplus5
                cdf_file['cplus6'] = cplus6
                cdf_file['oplus5'] = oplus5
                cdf_file['oplus6'] = oplus6
                cdf_file['oplus7'] = oplus7
                cdf_file['oplus8'] = oplus8
                cdf_file['cnoplus'] = cnoplus
                cdf_file['mg'] = mg
                cdf_file['si'] = si
                cdf_file['fe_loq'] = fe_loq
                cdf_file['fe_hiq'] = fe_hiq
                cdf_file['data_quality'] = data_quality
                cdf_file['spin_sector_index'] = spin_sector_index

            result: CodiceLoL2SWSpeciesData = CodiceLoL2SWSpeciesData.read_from_cdf(cdf_file_path)
            np.testing.assert_array_equal(result.epoch, epoch)
            np.testing.assert_array_equal(result.epoch_delta_minus, epoch_delta_minus)
            np.testing.assert_array_equal(result.epoch_delta_plus, epoch_delta_plus)
            np.testing.assert_array_equal(result.energy_table, energy_table)
            np.testing.assert_array_equal(result.hplus, hplus)
            np.testing.assert_array_equal(result.heplusplus, heplusplus)
            np.testing.assert_array_equal(result.heplus, heplus)
            np.testing.assert_array_equal(result.ne, ne)
            np.testing.assert_array_equal(result.cplus4, cplus4)
            np.testing.assert_array_equal(result.cplus5, cplus5)
            np.testing.assert_array_equal(result.cplus6, cplus6)
            np.testing.assert_array_equal(result.oplus5, oplus5)
            np.testing.assert_array_equal(result.oplus6, oplus6)
            np.testing.assert_array_equal(result.oplus7, oplus7)
            np.testing.assert_array_equal(result.oplus8, oplus8)
            np.testing.assert_array_equal(result.cnoplus, cnoplus)
            np.testing.assert_array_equal(result.mg, mg)
            np.testing.assert_array_equal(result.si, si)
            np.testing.assert_array_equal(result.fe_loq, fe_loq)
            np.testing.assert_array_equal(result.fe_hiq, fe_hiq)
            np.testing.assert_array_equal(result.data_quality, data_quality)
            np.testing.assert_array_equal(result.spin_sector_index, spin_sector_index)

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
                np.testing.assert_array_equal(priority_event.data_quality, expected_values[f"P{index}_DataQuality"])
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

        priority_event = PriorityEvent(np.array([]), np.array([]), np.array([]), np.array([]), energy_step,
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

        self.assertEqual(expected_total_events_by_energy_step_and_spin_angle,
                         priority_event.total_events_binned_by_energy_step_and_spin_angle())

    def test_codice_lo_l3a_partial_density_to_data_product(self):
        epoch_data = np.array([datetime.now()])

        input_data_product_kwargs = {
            "epoch": epoch_data,
            "epoch_delta_plus": np.array([1]),
            "epoch_delta_minus": np.array([-1]),
            "hplus_partial_density": np.array([15]),
            "heplusplus_partial_density": np.array([15]),
            "cplus4_partial_density": np.array([15]),
            "cplus5_partial_density": np.array([15]),
            "cplus6_partial_density": np.array([15]),
            "oplus5_partial_density": np.array([15]),
            "oplus6_partial_density": np.array([15]),
            "oplus7_partial_density": np.array([15]),
            "oplus8_partial_density": np.array([15]),
            "mg_partial_density": np.array([15]),
            "si_partial_density": np.array([15]),
            "fe_loq_partial_density": np.array([15]),
            "fe_hiq_partial_density": np.array([15]),
        }

        data_product = CodiceLoL3aPartialDensityDataProduct(
            Mock(), **input_data_product_kwargs
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
            lo_nsw_priority_p7_missing = rng.random((7, 10))
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
                cdf_file["lo_nsw_priority_p7_missing"] = lo_nsw_priority_p7_missing
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
            np.testing.assert_array_equal(result.lo_nsw_priority_p7_missing,
                                          lo_nsw_priority_p7_missing)
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

    def test_codice_lo_l3a_direct_event_to_data_product(self):
        rng = np.random.default_rng()

        epoch = np.array([datetime(2022, 3, 5), datetime(2022, 3, 6)])
        event_num = np.array([1, 2, 3, 4])
        spin_angle = np.array([30, 60, 90])
        energy_step = np.array([5.5, 6.6, 7.7])
        priority = np.array([0, 1, 2, 3, 4, 5, 6, 7])

        direct_event = CodiceLoL3aDirectEventDataProduct(
            input_metadata=Mock(),
            epoch=epoch,
            event_num=event_num,
            normalization=rng.random((len(epoch), len(priority), len(spin_angle), len(energy_step))),
            mass_per_charge=rng.random((len(epoch), len(priority), len(event_num))),
            mass=rng.random((len(epoch), len(priority), len(event_num))),
            energy=rng.random((len(epoch), len(priority), len(event_num))),
            gain=rng.random((len(epoch), len(priority), len(event_num))),
            apd_id=rng.random((len(epoch), len(priority), len(event_num))),
            multi_flag=rng.random((len(epoch), len(priority), len(event_num))),
            num_events=rng.random((len(epoch), len(priority), len(event_num))),
            data_quality=rng.random((len(epoch), len(priority))),
            pha_type=rng.random((len(epoch), len(priority), len(event_num))),
            tof=rng.random((len(epoch), len(priority), len(event_num))),
        )

        np.testing.assert_array_equal(direct_event.spin_angle,
                                      np.array([0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330]))
        np.testing.assert_array_equal(direct_event.energy_step, np.arange(128))
        np.testing.assert_array_equal(direct_event.priority, priority)

        data_products = direct_event.to_data_product_variables()

        non_parent_fields = [f for f in fields(CodiceLoL3aDirectEventDataProduct)
                             if f.name in CodiceLoL3aDirectEventDataProduct.__annotations__]

        self.assertEqual(len(data_products), len(non_parent_fields))
        for data_product in data_products:
            np.testing.assert_array_equal(getattr(direct_event, data_product.name), data_product.value)
