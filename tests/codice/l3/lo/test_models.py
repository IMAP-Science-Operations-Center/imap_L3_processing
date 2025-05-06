import tempfile
import unittest
from dataclasses import fields
from datetime import datetime
from pathlib import Path
from unittest.mock import Mock

import numpy as np
from spacepy.pycdf import CDF

from imap_l3_processing.codice.l3.lo.direct_events.science.mass_species_bin_lookup import EventDirection
from imap_l3_processing.codice.l3.lo.models import CodiceLoL2SWSpeciesData, CodiceLoL3aPartialDensityDataProduct, \
    CodiceLoL2DirectEventData, \
    PriorityEvent, EnergyAndSpinAngle, CodiceLoL3aDirectEventDataProduct, \
    CodiceLoL1aSWPriorityRates, CodiceLoL1aNSWPriorityRates, CodiceLo3dData


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

            for index, priority_event in enumerate(result.priority_events):
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
            "epoch_delta": np.array([1]),
            "hplus_partial_density": np.array([15]),
            "heplusplus_partial_density": np.array([15]),
            "cplus4_partial_density": np.array([15]),
            "cplus5_partial_density": np.array([15]),
            "cplus6_partial_density": np.array([15]),
            "oplus5_partial_density": np.array([15]),
            "oplus6_partial_density": np.array([15]),
            "oplus7_partial_density": np.array([15]),
            "oplus8_partial_density": np.array([15]),
            "ne_partial_density": np.array([15]),
            "mg_partial_density": np.array([15]),
            "si_partial_density": np.array([15]),
            "fe_loq_partial_density": np.array([15]),
            "fe_hiq_partial_density": np.array([15]),
            "c_to_o_ratio": np.array([15]),
            "mg_to_o_ratio": np.array([15]),
            "fe_to_o_ratio": np.array([15]),
        }

        data_product = CodiceLoL3aPartialDensityDataProduct(
            Mock(), **input_data_product_kwargs
        )
        actual_data_product_variables = data_product.to_data_product_variables()

        self.assertEqual(19, len(actual_data_product_variables))

        for input_variable, actual_data_product_variable in zip(input_data_product_kwargs.items(),
                                                                actual_data_product_variables):
            input_name, expected_value = input_variable

            np.testing.assert_array_equal(actual_data_product_variable.value, getattr(data_product, input_name))
            self.assertEqual(input_name, actual_data_product_variable.name)

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

    def test_codice_lo_l1a_sw_priority_read_from_cdf(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            epoch = np.array([datetime(2025, 4, 18), datetime(2025, 4, 18)])
            rng = np.random.default_rng()
            energy_table = rng.random(128)
            spin_sector_index = rng.random(12)
            expected_values = {
                "epoch": epoch,
                "epoch_delta_plus": np.repeat(1, len(epoch)),
                "epoch_delta_minus": np.repeat(1, len(epoch)),
                "energy_table": energy_table,
                "acquisition_time_per_step": rng.random((len(epoch), len(energy_table))),
                "spin_sector_index": spin_sector_index,
                "rgfo_half_spin": rng.random(len(epoch)),
                "nso_half_spin": rng.random(len(epoch)),
                "sw_bias_gain_mode": rng.random(len(epoch)),
                "st_bias_gain_mode": rng.random(len(epoch)),
                "data_quality": rng.random(len(epoch)),
                "spin_period": rng.random(len(epoch)),
                "p0_tcrs": rng.random((len(epoch), len(energy_table), len(spin_sector_index))),
                "p1_hplus": rng.random((len(epoch), len(energy_table), len(spin_sector_index))),
                "p2_heplusplus": rng.random((len(epoch), len(energy_table), len(spin_sector_index))),
                "p3_heavies": rng.random((len(epoch), len(energy_table), len(spin_sector_index))),
                "p4_dcrs": rng.random((len(epoch), len(energy_table), len(spin_sector_index))),
            }

            cdf_file_path = Path(tmpdir) / "test_cdf.cdf"
            with CDF(str(cdf_file_path), readonly=False, masterpath="") as cdf_file:
                for k, v in expected_values.items():
                    cdf_file[k] = v

            result = CodiceLoL1aSWPriorityRates.read_from_cdf(cdf_file_path)

            for k, v in expected_values.items():
                np.testing.assert_array_equal(getattr(result, k), v)

    def test_codice_lo_l1a_nsw_priority_read_from_cdf(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            epoch = np.array([datetime(2025, 4, 18), datetime(2025, 4, 18)])
            rng = np.random.default_rng()
            energy_table = rng.random(128)
            spin_sector_index = rng.random(12)
            expected_values = {
                "energy_table": energy_table,
                "acquisition_time_per_step": rng.random(len(energy_table)),
                "epoch": epoch,
                "epoch_delta_plus": np.repeat(1, len(epoch)),
                "epoch_delta_minus": np.repeat(1, len(epoch)),
                "spin_sector_index": spin_sector_index,
                "rgfo_half_spin": rng.random(len(epoch)),
                "data_quality": rng.random(len(epoch)),
                "p5_heavies": rng.random((len(epoch), len(energy_table), len(spin_sector_index))),
                "p6_hplus_heplusplus": rng.random((len(epoch), len(energy_table), len(spin_sector_index))),
                "nso_half_spin": rng.random(len(epoch)),
                "sw_bias_gain_mode": rng.random(len(epoch)),
                "st_bias_gain_mode": rng.random(len(epoch)),
                "spin_period": rng.random(len(epoch))
            }

            cdf_file_path = Path(tmpdir) / "test_cdf.cdf"
            with CDF(str(cdf_file_path), readonly=False, masterpath="") as cdf_file:
                for k, v in expected_values.items():
                    cdf_file[k] = v

            result = CodiceLoL1aNSWPriorityRates.read_from_cdf(cdf_file_path)

            for k, v in expected_values.items():
                np.testing.assert_array_equal(getattr(result, k), v)

    def test_codice_lo_3d_data_get_3d_distribution(self):
        data_in_bins = np.arange(8).reshape((2, 2, 2))
        mass_bin_lookup = Mock()
        mass_bin_lookup.get_species_index.return_value = 1
        codice_lo_3d_data = CodiceLo3dData(data_in_bins, mass_bin_lookup)

        expected_species_data = data_in_bins[:, 1, ...]
        actual_species_data = codice_lo_3d_data.get_3d_distribution("H+", EventDirection.Sunward)

        mass_bin_lookup.get_species_index.assert_called_with("H+", EventDirection.Sunward)

        np.testing.assert_array_equal(actual_species_data, expected_species_data)
