import tempfile
import unittest
from dataclasses import fields
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import Mock

import numpy as np
from spacepy.pycdf import CDF

from imap_l3_processing.codice.l3.lo.direct_events.science.mass_species_bin_lookup import EventDirection
from imap_l3_processing.codice.l3.lo.models import CodiceLoL2SWSpeciesData, CodiceLoL3aPartialDensityDataProduct, \
    CodiceLoL2DirectEventData, \
    CodiceLoL3aDirectEventDataProduct, \
    CodiceLoL1aSWPriorityRates, CodiceLoL1aNSWPriorityRates, CodiceLo3dData, CODICE_LO_L2_NUM_PRIORITIES, \
    CodiceLoL3aRatiosDataProduct, CodiceLoPartialDensityData, CodiceLoL3ChargeStateDistributionsDataProduct, \
    CodiceLoDirectEventData
from imap_l3_processing.models import InputMetadata
from imap_l3_processing.utils import save_data
from tests.test_helpers import get_test_instrument_team_data_path, get_test_data_path


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

    def test_codice_lo_l2_direct_event_read_from_instrument_team_cdf(self):
        path_to_cdf = get_test_instrument_team_data_path(
            'codice/lo/imap_codice_l2_lo-direct-events_20241110_v002.cdf')
        l2_direct_event_data = CodiceLoL2DirectEventData.read_from_cdf(path_to_cdf)
        with CDF(str(path_to_cdf)) as cdf:
            np.testing.assert_array_equal(cdf["epoch"], l2_direct_event_data.epoch)
            np.testing.assert_array_equal(cdf["epoch_delta_plus"], l2_direct_event_data.epoch_delta_plus)
            np.testing.assert_array_equal(cdf["epoch_delta_minus"], l2_direct_event_data.epoch_delta_minus)
            for index in range(CODICE_LO_L2_NUM_PRIORITIES):
                np.testing.assert_array_equal(cdf[f"p{index}_apd_energy"],
                                              l2_direct_event_data.priority_events[index].apd_energy)
                np.testing.assert_array_equal(cdf[f"p{index}_gain"],
                                              l2_direct_event_data.priority_events[index].apd_gain)
                np.testing.assert_array_equal(cdf[f"p{index}_apd_id"],
                                              l2_direct_event_data.priority_events[index].apd_id)
                np.testing.assert_array_equal(cdf[f"p{index}_data_quality"],
                                              l2_direct_event_data.priority_events[index].data_quality)
                np.testing.assert_array_equal(cdf[f"p{index}_energy_step"],
                                              l2_direct_event_data.priority_events[index].energy_step)
                np.testing.assert_array_equal(cdf[f"p{index}_multi_flag"],
                                              l2_direct_event_data.priority_events[index].multi_flag)
                np.testing.assert_array_equal(cdf[f"p{index}_num_events"],
                                              l2_direct_event_data.priority_events[index].num_events)
                np.testing.assert_array_equal(cdf[f"p{index}_spin_sector"],
                                              l2_direct_event_data.priority_events[index].spin_angle)
                np.testing.assert_array_equal(cdf[f"p{index}_position"],
                                              l2_direct_event_data.priority_events[index].elevation)
                np.testing.assert_array_equal(cdf[f"p{index}_tof"], l2_direct_event_data.priority_events[index].tof)

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
        }

        data_product = CodiceLoL3aPartialDensityDataProduct(
            Mock(), CodiceLoPartialDensityData(**input_data_product_kwargs)
        )

        actual_data_product_variables = data_product.to_data_product_variables()

        self.assertEqual(16, len(actual_data_product_variables))

        for input_variable, actual_data_product_variable in zip(input_data_product_kwargs.items(),
                                                                actual_data_product_variables):
            input_name, expected_value = input_variable

            np.testing.assert_array_equal(actual_data_product_variable.value, getattr(data_product.data, input_name))
            self.assertEqual(input_name, actual_data_product_variable.name)

    def test_codice_lo_l3a_ratios_to_data_product(self):
        epoch_data = np.array([datetime.now()])

        input_data_product_kwargs = {
            "epoch": epoch_data,
            "epoch_delta": np.array([1]),
            "c_to_o_ratio": np.array([2]),
            "mg_to_o_ratio": np.array([3]),
            "fe_to_o_ratio": np.array([4]),
            "c6_to_c5_ratio": np.array([5]),
            "c6_to_c4_ratio": np.array([6]),
            "o7_to_o6_ratio": np.array([7]),
            "felo_to_fehi_ratio": np.array([8]),
        }

        data_product = CodiceLoL3aRatiosDataProduct(
            Mock(), **input_data_product_kwargs
        )
        actual_data_product_variables = data_product.to_data_product_variables()

        self.assertEqual(9, len(actual_data_product_variables))

        for input_variable, actual_data_product_variable in zip(input_data_product_kwargs.items(),
                                                                actual_data_product_variables):
            input_name, expected_value = input_variable

            np.testing.assert_array_equal(actual_data_product_variable.value, getattr(data_product, input_name))
            self.assertEqual(input_name, actual_data_product_variable.name)

    def test_codice_lo_l3a_ratios_read_from_cdf(self):
        rng = np.random.default_rng()
        epoch = [datetime.now(), datetime.now() + timedelta(days=1)]
        density_data = CodiceLoPartialDensityData(epoch=np.array(epoch), epoch_delta=np.array([2, 2]),
                                                  hplus_partial_density=rng.random(len(epoch), dtype='float32'),
                                                  heplusplus_partial_density=rng.random(len(epoch), dtype='float32'),
                                                  cplus4_partial_density=rng.random(len(epoch), dtype='float32'),
                                                  cplus5_partial_density=rng.random(len(epoch), dtype='float32'),
                                                  cplus6_partial_density=rng.random(len(epoch), dtype='float32'),
                                                  oplus5_partial_density=rng.random(len(epoch), dtype='float32'),
                                                  oplus6_partial_density=rng.random(len(epoch), dtype='float32'),
                                                  oplus7_partial_density=rng.random(len(epoch), dtype='float32'),
                                                  oplus8_partial_density=rng.random(len(epoch), dtype='float32'),
                                                  ne_partial_density=rng.random(len(epoch), dtype='float32'),
                                                  mg_partial_density=rng.random(len(epoch), dtype='float32'),
                                                  si_partial_density=rng.random(len(epoch), dtype='float32'),
                                                  fe_loq_partial_density=rng.random(len(epoch), dtype='float32'),
                                                  fe_hiq_partial_density=rng.random(len(epoch), dtype='float32'), )

        input_metadata = InputMetadata("codice", "l3a", datetime.now(), datetime.now(), "v000", "lo-partial-densities")

        codice_lo_l3a_partial_densities = CodiceLoL3aPartialDensityDataProduct(input_metadata, density_data)

        with tempfile.TemporaryDirectory() as tmpdir:
            cdf_path = save_data(codice_lo_l3a_partial_densities, folder_path=Path(tmpdir))

            read_in_density_data = CodiceLoPartialDensityData.read_from_cdf(cdf_path)

        for field in fields(CodiceLoPartialDensityData):
            np.testing.assert_array_equal(getattr(read_in_density_data, field.name),
                                          getattr(density_data, field.name),
                                          strict=True)

    def test_codice_lo_l3a_charge_state_distributions_to_data_product(self):
        rng = np.random.default_rng()
        epoch_data = np.array([datetime.now()])
        input_data_product_kwargs = {
            "epoch": epoch_data,
            "epoch_delta": np.array([1]),
            "oxygen_charge_state_distribution": rng.random((len(epoch_data), 4)),
            "carbon_charge_state_distribution": rng.random((len(epoch_data), 3)),
        }

        data_product = CodiceLoL3ChargeStateDistributionsDataProduct(Mock(), **input_data_product_kwargs)
        actual_data_product_variables = data_product.to_data_product_variables()
        self.assertEqual(6, len(actual_data_product_variables))
        for input_variable, actual_data_product_variable in zip(input_data_product_kwargs.items(),
                                                                actual_data_product_variables[:4]):
            input_name, expected_value = input_variable

            np.testing.assert_array_equal(actual_data_product_variable.value, getattr(data_product, input_name))
            self.assertEqual(input_name, actual_data_product_variable.name)

        oxygen_charge_states, carbon_charge_states, = actual_data_product_variables[4:]
        np.testing.assert_array_equal(oxygen_charge_states.value, [5, 6, 7, 8])
        self.assertEqual("oxygen_charge_state", oxygen_charge_states.name)

        np.testing.assert_array_equal(carbon_charge_states.value, [4, 5, 6])
        self.assertEqual("carbon_charge_state", carbon_charge_states.name)

    def test_codice_lo_l3a_direct_event_read_from_cdf(self):
        l3a_cdf_path = get_test_data_path("codice/imap_codice_l3a_lo-direct-events_20241110_v000.cdf")
        actual_event_data = CodiceLoDirectEventData.read_from_cdf(l3a_cdf_path)

        with CDF(str(l3a_cdf_path)) as cdf:
            np.testing.assert_array_equal(actual_event_data.epoch, cdf["epoch"])
            np.testing.assert_array_equal(actual_event_data.epoch_delta, cdf["epoch_delta"])
            np.testing.assert_array_equal(actual_event_data.normalization, cdf["normalization"])
            np.testing.assert_array_equal(actual_event_data.mass_per_charge, cdf["mass_per_charge"])
            np.testing.assert_array_equal(actual_event_data.mass, cdf["mass"])
            np.testing.assert_array_equal(actual_event_data.event_energy, cdf["event_energy"])
            np.testing.assert_array_equal(actual_event_data.gain, cdf["gain"])
            np.testing.assert_array_equal(actual_event_data.apd_id, cdf["apd_id"])
            np.testing.assert_array_equal(actual_event_data.multi_flag, cdf["multi_flag"])
            np.testing.assert_array_equal(actual_event_data.num_events, cdf["num_events"])
            np.testing.assert_array_equal(actual_event_data.data_quality, cdf["data_quality"])
            np.testing.assert_array_equal(actual_event_data.tof, cdf["tof"])
            np.testing.assert_array_equal(actual_event_data.spin_angle, cdf["spin_angle"])
            np.testing.assert_array_equal(actual_event_data.elevation, cdf["elevation"])

    def test_codice_lo_l3a_direct_event_read_from_cdf_handles_fill_value(self):
        all_fill_l3a_cdf_path = get_test_data_path("codice/imap_codice_l3a_lo-direct-events_20241110_v000-all-fill.cdf")
        actual_event_data = CodiceLoDirectEventData.read_from_cdf(all_fill_l3a_cdf_path)

        with CDF(str(all_fill_l3a_cdf_path)) as cdf:
            # @formatter:off
            np.testing.assert_array_equal(actual_event_data.normalization, np.full_like(cdf["normalization"][...], np.nan))
            np.testing.assert_array_equal(actual_event_data.mass_per_charge, np.full_like(cdf["mass_per_charge"][...], np.nan))
            np.testing.assert_array_equal(actual_event_data.mass, np.full_like(cdf["mass"][...], np.nan))
            np.testing.assert_array_equal(actual_event_data.event_energy, np.full_like(cdf["event_energy"][...], np.nan))
            np.testing.assert_array_equal(actual_event_data.spin_angle, np.full_like(cdf["spin_angle"][...], np.nan))
            np.testing.assert_array_equal(actual_event_data.elevation, np.full_like(cdf["elevation"][...], np.nan))

            self.assertIsInstance(actual_event_data.apd_id, np.ma.masked_array)
            np.testing.assert_array_equal(actual_event_data.apd_id.data, cdf["apd_id"])
            self.assertTrue(np.all(actual_event_data.apd_id.mask))

            self.assertIsInstance(actual_event_data.gain, np.ma.masked_array)
            np.testing.assert_array_equal(actual_event_data.gain.data, cdf["gain"])
            self.assertTrue(np.all(actual_event_data.gain.mask))

            self.assertIsInstance(actual_event_data.multi_flag, np.ma.masked_array)
            np.testing.assert_array_equal(actual_event_data.multi_flag.data, cdf["multi_flag"])
            self.assertTrue(np.all(actual_event_data.multi_flag.mask))

            self.assertIsInstance(actual_event_data.num_events, np.ma.masked_array)
            np.testing.assert_array_equal(actual_event_data.num_events.data, cdf["num_events"])
            self.assertTrue(np.all(actual_event_data.num_events.mask))

            self.assertIsInstance(actual_event_data.data_quality, np.ma.masked_array)
            np.testing.assert_array_equal(actual_event_data.data_quality.data, cdf["data_quality"])
            self.assertTrue(np.all(actual_event_data.data_quality.mask))

            self.assertIsInstance(actual_event_data.tof, np.ma.masked_array)
            np.testing.assert_array_equal(actual_event_data.tof.data, cdf["tof"])
            self.assertTrue(np.all(actual_event_data.tof.mask))
            # @formatter:on

    def test_codice_lo_l3a_direct_event_to_data_product(self):
        rng = np.random.default_rng()

        epoch = np.array([datetime(2022, 3, 5), datetime(2022, 3, 6)])
        epoch_delta = np.full(len(epoch), 1)
        event_num = np.array([1, 2, 3, 4])
        spin_angle = np.array([30, 60, 90])
        energy_step = np.array([5.5, 6.6, 7.7])
        priority = np.arange(CODICE_LO_L2_NUM_PRIORITIES)

        direct_event = CodiceLoL3aDirectEventDataProduct(
            input_metadata=Mock(),
            epoch=epoch,
            epoch_delta=epoch_delta,
            normalization=rng.random((len(epoch), len(priority), len(spin_angle), len(energy_step))),
            mass_per_charge=rng.random((len(epoch), len(priority), len(event_num))),
            mass=rng.random((len(epoch), len(priority), len(event_num))),
            event_energy=rng.random((len(epoch), len(priority), len(event_num))),
            gain=rng.random((len(epoch), len(priority), len(event_num))),
            apd_id=rng.random((len(epoch), len(priority), len(event_num))),
            multi_flag=rng.random((len(epoch), len(priority), len(event_num))),
            num_events=rng.random((len(epoch), len(priority), len(event_num))),
            data_quality=rng.random((len(epoch), len(priority))),
            tof=rng.random((len(epoch), len(priority), len(event_num))),
            spin_angle=rng.random((len(epoch), len(priority), len(event_num))),
            elevation=rng.random((len(epoch), len(priority), len(event_num)))
        )

        spin_angle_bins = np.array([0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330])
        np.testing.assert_array_equal(direct_event.spin_angle_bin,
                                      spin_angle_bins)
        np.testing.assert_array_equal(direct_event.energy_bin, np.arange(128))
        np.testing.assert_array_equal(direct_event.priority_index, priority)
        np.testing.assert_array_equal(direct_event.event_index, np.arange(len(event_num)))
        np.testing.assert_array_equal(direct_event.priority_index_label, np.array(["0", "1", "2", "3", "4", "5", "6"]))
        np.testing.assert_array_equal(direct_event.event_index_label, np.array([str(i) for i in range(len(event_num))]))
        np.testing.assert_array_equal(direct_event.energy_bin_label,
                                      np.array([str(e) for e in np.arange(128)]))
        np.testing.assert_array_equal(direct_event.spin_angle_bin_label,
                                      np.array([str(spin_angle) for spin_angle in spin_angle_bins]))

        data_products = direct_event.to_data_product_variables()

        non_parent_fields = [f for f in fields(CodiceLoL3aDirectEventDataProduct)
                             if f.name in CodiceLoL3aDirectEventDataProduct.__annotations__ or
                             f.name in CodiceLoDirectEventData.__annotations__]

        self.assertEqual(len(data_products), len(non_parent_fields))
        for data_product in data_products:
            np.testing.assert_array_equal(getattr(direct_event, data_product.name), data_product.value)

    def test_codice_lo_l1a_sw_priority_read_from_instrument_team_cdf(self):
        instrument_team_cdf_path = get_test_instrument_team_data_path(
            "codice/lo/imap_codice_l1a_lo-sw-priority_20241110_v002.cdf")
        actual_l1a_sw_priority_rates = CodiceLoL1aSWPriorityRates.read_from_cdf(instrument_team_cdf_path)

        with CDF(str(instrument_team_cdf_path)) as cdf:
            np.testing.assert_array_equal(actual_l1a_sw_priority_rates.epoch, cdf["epoch"][...])
            np.testing.assert_array_equal(actual_l1a_sw_priority_rates.epoch_delta_plus, cdf["epoch_delta_plus"][...])
            np.testing.assert_array_equal(actual_l1a_sw_priority_rates.epoch_delta_minus, cdf["epoch_delta_minus"][...])
            np.testing.assert_array_equal(actual_l1a_sw_priority_rates.energy_table, cdf["energy_table"][...])
            np.testing.assert_array_equal(actual_l1a_sw_priority_rates.acquisition_time_per_step,
                                          cdf["acquisition_time_per_step"][...])
            np.testing.assert_array_equal(actual_l1a_sw_priority_rates.spin_sector_index, cdf["spin_sector_index"][...])
            np.testing.assert_array_equal(actual_l1a_sw_priority_rates.rgfo_half_spin, cdf["rgfo_half_spin"][...])
            np.testing.assert_array_equal(actual_l1a_sw_priority_rates.nso_half_spin, cdf["nso_half_spin"][...])
            np.testing.assert_array_equal(actual_l1a_sw_priority_rates.sw_bias_gain_mode, cdf["sw_bias_gain_mode"][...])
            np.testing.assert_array_equal(actual_l1a_sw_priority_rates.st_bias_gain_mode, cdf["st_bias_gain_mode"][...])
            np.testing.assert_array_equal(actual_l1a_sw_priority_rates.data_quality, cdf["data_quality"][...])
            np.testing.assert_array_equal(actual_l1a_sw_priority_rates.spin_period, cdf["spin_period"][...])
            np.testing.assert_array_equal(actual_l1a_sw_priority_rates.p0_tcrs, cdf["p0_tcrs"][...])
            np.testing.assert_array_equal(actual_l1a_sw_priority_rates.p1_hplus, cdf["p1_hplus"][...])
            np.testing.assert_array_equal(actual_l1a_sw_priority_rates.p2_heplusplus, cdf["p2_heplusplus"][...])
            np.testing.assert_array_equal(actual_l1a_sw_priority_rates.p3_heavies, cdf["p3_heavies"][...])
            np.testing.assert_array_equal(actual_l1a_sw_priority_rates.p4_dcrs, cdf["p4_dcrs"][...])

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

    def test_codice_lo_l1a_nsw_priority_read_from_instrument_team_cdf(self):
        instrument_team_cdf_path = get_test_instrument_team_data_path(
            "codice/lo/imap_codice_l1a_lo-nsw-priority_20241110_v002.cdf")
        actual_l1a_nsw_priority_rates = CodiceLoL1aNSWPriorityRates.read_from_cdf(instrument_team_cdf_path)

        with CDF(str(instrument_team_cdf_path)) as cdf:
            np.testing.assert_array_equal(actual_l1a_nsw_priority_rates.epoch, cdf["epoch"][...])
            np.testing.assert_array_equal(actual_l1a_nsw_priority_rates.epoch_delta_plus, cdf["epoch_delta_plus"][...])
            np.testing.assert_array_equal(actual_l1a_nsw_priority_rates.epoch_delta_minus,
                                          cdf["epoch_delta_minus"][...])
            np.testing.assert_array_equal(actual_l1a_nsw_priority_rates.energy_table, cdf["energy_table"][...])
            np.testing.assert_array_equal(actual_l1a_nsw_priority_rates.acquisition_time_per_step,
                                          cdf["acquisition_time_per_step"][...])
            np.testing.assert_array_equal(actual_l1a_nsw_priority_rates.spin_sector_index,
                                          cdf["spin_sector_index"][...])
            np.testing.assert_array_equal(actual_l1a_nsw_priority_rates.rgfo_half_spin, cdf["rgfo_half_spin"][...])
            np.testing.assert_array_equal(actual_l1a_nsw_priority_rates.data_quality, cdf["data_quality"][...])
            np.testing.assert_array_equal(actual_l1a_nsw_priority_rates.p5_heavies, cdf["p5_heavies"][...])
            np.testing.assert_array_equal(actual_l1a_nsw_priority_rates.p6_hplus_heplusplus,
                                          cdf["p6_hplus_heplusplus"][...])
            np.testing.assert_array_equal(actual_l1a_nsw_priority_rates.nso_half_spin, cdf["nso_half_spin"][...])
            np.testing.assert_array_equal(actual_l1a_nsw_priority_rates.sw_bias_gain_mode,
                                          cdf["sw_bias_gain_mode"][...])
            np.testing.assert_array_equal(actual_l1a_nsw_priority_rates.st_bias_gain_mode,
                                          cdf["st_bias_gain_mode"][...])
            np.testing.assert_array_equal(actual_l1a_nsw_priority_rates.spin_period, cdf["spin_period"][...])

    def test_codice_lo_3d_data_get_3d_distribution(self):
        data_in_bins = np.arange(16).reshape((2, 2, 2, 2))
        mass_bin_lookup = Mock()
        mass_bin_lookup.get_species_index.return_value = 1
        codice_lo_3d_data = CodiceLo3dData(data_in_bins, mass_bin_lookup)

        expected_species_data = data_in_bins[1, :, :, ...]
        actual_species_data = codice_lo_3d_data.get_3d_distribution("H+", EventDirection.Sunward)

        mass_bin_lookup.get_species_index.assert_called_with("H+", EventDirection.Sunward)

        np.testing.assert_array_equal(actual_species_data, expected_species_data)
