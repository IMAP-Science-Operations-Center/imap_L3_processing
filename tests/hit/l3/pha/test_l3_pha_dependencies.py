import unittest
from unittest.mock import patch, call, sentinel

from imap_processing.hit.l3.pha.hit_l3_pha_dependencies import HitL3PhaDependencies, HIT_L1A_EVENT_DESCRIPTOR, \
    HIT_L3_RANGE_2_COSINE_LOOKUP_DESCRIPTOR, HIT_L3_RANGE_3_COSINE_LOOKUP_DESCRIPTOR, \
    HIT_L3_RANGE_4_COSINE_LOOKUP_DESCRIPTOR, HIT_L3_LO_GAIN_LOOKUP_DESCRIPTOR, HIT_L3_HI_GAIN_LOOKUP_DESCRIPTOR, \
    HIT_L3_RANGE_2_CHARGE_FIT_LOOKUP_DESCRIPTOR, HIT_L3_RANGE_3_CHARGE_FIT_LOOKUP_DESCRIPTOR, \
    HIT_L3_RANGE_4_CHARGE_FIT_LOOKUP_DESCRIPTOR
from imap_processing.models import UpstreamDataDependency


class TestHitL3PhaDependencies(unittest.TestCase):

    @patch('imap_processing.hit.l3.pha.hit_l3_pha_dependencies.download_dependency')
    @patch('imap_processing.hit.l3.pha.hit_l3_pha_dependencies.HitL1Data.read_from_cdf')
    @patch('imap_processing.hit.l3.pha.hit_l3_pha_dependencies.CosineCorrectionLookupTable')
    @patch('imap_processing.hit.l3.pha.hit_l3_pha_dependencies.GainLookupTable.from_file')
    @patch('imap_processing.hit.l3.pha.hit_l3_pha_dependencies.RangeFitLookup.from_files')
    def test_fetch_dependencies(self, mock_range_fit_from_lookup, mock_gain_lookup_from_file,
                                mock_cosine_correction_lookup_from_file, mock_read_from_cdf, mock_download_dependency):
        upstream_data_dependency = UpstreamDataDependency("hit", "l1a", HIT_L1A_EVENT_DESCRIPTOR, None, None, "v003")

        mock_download_dependency.side_effect = [
            sentinel.l1_data_cdf_path,
            sentinel.range_2_cosine_lookup_cdf_path,
            sentinel.range_3_cosine_lookup_cdf_path,
            sentinel.range_4_cosine_lookup_cdf_path,
            sentinel.lo_gain_lookup_cdf_path,
            sentinel.hi_gain_lookup_cdf_path,
            sentinel.range_2_charge_fit_lookup_cdf_path,
            sentinel.range_3_charge_fit_lookup_cdf_path,
            sentinel.range_4_charge_fit_lookup_cdf_path,
        ]

        hit_L3_pha_dependencies = HitL3PhaDependencies.fetch_dependencies([upstream_data_dependency])

        mock_download_dependency.assert_has_calls([
            call(upstream_data_dependency),

            call(UpstreamDataDependency("hit", "l3", None, None, "latest", HIT_L3_RANGE_2_COSINE_LOOKUP_DESCRIPTOR, )),
            call(UpstreamDataDependency("hit", "l3", None, None, "latest", HIT_L3_RANGE_3_COSINE_LOOKUP_DESCRIPTOR, )),
            call(UpstreamDataDependency("hit", "l3", None, None, "latest", HIT_L3_RANGE_4_COSINE_LOOKUP_DESCRIPTOR, )),

            call(UpstreamDataDependency("hit", "l3", None, None, "latest", HIT_L3_LO_GAIN_LOOKUP_DESCRIPTOR, )),
            call(UpstreamDataDependency("hit", "l3", None, None, "latest", HIT_L3_HI_GAIN_LOOKUP_DESCRIPTOR, )),

            call(
                UpstreamDataDependency("hit", "l3", None, None, "latest",
                                       HIT_L3_RANGE_2_CHARGE_FIT_LOOKUP_DESCRIPTOR, )),
            call(
                UpstreamDataDependency("hit", "l3", None, None, "latest",
                                       HIT_L3_RANGE_3_CHARGE_FIT_LOOKUP_DESCRIPTOR, )),
            call(
                UpstreamDataDependency("hit", "l3", None, None, "latest",
                                       HIT_L3_RANGE_4_CHARGE_FIT_LOOKUP_DESCRIPTOR, )),
        ])
        mock_read_from_cdf.assert_called_with(sentinel.l1_data_cdf_path)

        mock_cosine_correction_lookup_from_file.assert_called_once_with(sentinel.range_2_cosine_lookup_cdf_path,
                                                                        sentinel.range_3_cosine_lookup_cdf_path,
                                                                        sentinel.range_4_cosine_lookup_cdf_path, )
        mock_gain_lookup_from_file.assert_called_once_with(sentinel.hi_gain_lookup_cdf_path,
                                                           sentinel.lo_gain_lookup_cdf_path, )
        mock_range_fit_from_lookup.assert_called_once_with(sentinel.range_2_charge_fit_lookup_cdf_path,
                                                           sentinel.range_3_charge_fit_lookup_cdf_path,
                                                           sentinel.range_4_charge_fit_lookup_cdf_path, )

        self.assertEqual(mock_read_from_cdf.return_value, hit_L3_pha_dependencies.hit_l1_data)
        self.assertEqual(mock_cosine_correction_lookup_from_file.return_value,
                         hit_L3_pha_dependencies.cosine_correction_lookup)
        self.assertEqual(mock_gain_lookup_from_file.return_value, hit_L3_pha_dependencies.gain_lookup)
        self.assertEqual(mock_range_fit_from_lookup.return_value, hit_L3_pha_dependencies.range_fit_lookup)

    @patch('imap_processing.hit.l3.pha.hit_l3_pha_dependencies.download_dependency')
    @patch('imap_processing.hit.l3.pha.hit_l3_pha_dependencies.HitL1Data.read_from_cdf')
    def test_fetch_dependencies_throws_when_no_data_dependency_provided(self, mock_read_from_cdf,
                                                                        mock_download_dependency):
        with self.assertRaises(ValueError) as e:
            _ = HitL3PhaDependencies.fetch_dependencies([])
        self.assertEqual(str(e.exception), f"Missing {HIT_L1A_EVENT_DESCRIPTOR} dependency.")
