from datetime import datetime
from pathlib import Path
from unittest import TestCase
from unittest.mock import patch, call, Mock, sentinel

from imap_data_access.processing_input import ScienceInput, ProcessingInputCollection, \
    AncillaryInput, SPICEInput

from imap_l3_data_processor import imap_l3_processor
from imap_l3_processing.models import InputMetadata


class TestImapL3DataProcessor(TestCase):
    @patch('imap_l3_data_processor.SwapiProcessor')
    @patch('imap_l3_data_processor.argparse')
    @patch('imap_l3_data_processor.ProcessingInputCollection')
    @patch('imap_l3_data_processor.imap_data_access.upload')
    def test_runs_swapi_processor_when_instrument_argument_is_swapi(self, mock_upload, mock_processing_input_collection,
                                                                    mock_argparse, mock_swapi_processor_class):
        instrument_argument = "swapi"
        data_level_argument = "l3a"
        start_date_argument = "20160630"
        version_argument = "v092"
        descriptor_argument = "proton"
        science_input = ScienceInput("imap_swapi_l3a_science_20250101_v112.cdf")
        imap_data_access_dependency = ProcessingInputCollection(science_input)

        cases = [("20170630", datetime(2017, 6, 30)), (None, datetime(2016, 6, 30))]

        mock_processing_input_collection.return_value = imap_data_access_dependency
        mock_processing_input_collection.deserialize = Mock()
        mock_processing_input_collection.get_science_inputs = Mock(return_value=[])

        mock_argument_parser = mock_argparse.ArgumentParser.return_value

        mock_argument_parser.parse_args.return_value.instrument = instrument_argument
        mock_argument_parser.parse_args.return_value.data_level = data_level_argument
        mock_argument_parser.parse_args.return_value.dependency = imap_data_access_dependency.serialize()
        mock_argument_parser.parse_args.return_value.start_date = start_date_argument
        mock_argument_parser.parse_args.return_value.version = version_argument
        mock_argument_parser.parse_args.return_value.version = version_argument
        mock_argument_parser.parse_args.return_value.descriptor = descriptor_argument

        mock_swapi_processor = mock_swapi_processor_class.return_value

        for input_end_date, expected_end_date in cases:
            with self.subTest(input_end_date):
                mock_upload.reset_mock()

                mock_argument_parser.parse_args.return_value.end_date = input_end_date

                mock_swapi_processor.process.return_value = [Mock()]

                imap_l3_processor()

                parser = mock_argparse.ArgumentParser()
                parser.add_argument.assert_has_calls([
                    call("--instrument"),
                    call("--data-level"),
                    call("--descriptor"),
                    call("--start-date"),
                    call("--end-date", required=False),
                    call("--version"),
                    call("--dependency"),
                    call("--upload-to-sdc", action="store_true", required=False,
                         help="Upload completed output files to the IMAP SDC.")
                ])

                expected_input_metadata = InputMetadata("swapi", "l3a", datetime(year=2016, month=6, day=30),
                                                        expected_end_date, "v092", "proton")

                mock_swapi_processor.process.assert_called()

                mock_swapi_processor_class.assert_called_with(imap_data_access_dependency, expected_input_metadata)
                mock_upload.assert_called_once_with(mock_swapi_processor.process.return_value[0])

    @patch('imap_l3_data_processor.GlowsProcessor')
    @patch('imap_l3_data_processor.argparse')
    @patch('imap_l3_data_processor.ProcessingInputCollection')
    @patch('imap_l3_data_processor.imap_data_access.upload')
    def test_runs_glows_processor_when_instrument_argument_is_glows(self, mock_upload, mock_processing_input_collection,
                                                                    mock_argparse, mock_processor_class):
        cases = [("20170630", datetime(2017, 6, 30), "l3a", "lightcurve"),
                 (None, datetime(2016, 6, 30), "l3a", "lightcurve"),
                 ("20170630", datetime(2017, 6, 30), "l3b", "ionization-rates"),
                 (None, datetime(2016, 6, 30), "l3b", "ionization-rates")]

        instrument_argument = "glows"
        start_date_argument = "20160630"
        version_argument = "v092"
        science_input = ScienceInput("imap_glows_l1_science_20250101_v112.cdf")
        imap_data_access_dependency = ProcessingInputCollection(science_input)

        mock_processing_input_collection.return_value = imap_data_access_dependency
        mock_processing_input_collection.deserialize = Mock()
        mock_processing_input_collection.get_science_inputs = Mock(return_value=[])

        mock_argument_parser = mock_argparse.ArgumentParser.return_value

        mock_argument_parser.parse_args.return_value.instrument = instrument_argument
        mock_argument_parser.parse_args.return_value.dependency = imap_data_access_dependency.serialize()
        mock_argument_parser.parse_args.return_value.start_date = start_date_argument
        mock_argument_parser.parse_args.return_value.version = version_argument

        mock_processor = mock_processor_class.return_value

        for input_end_date, expected_end_date, data_level, descriptor in cases:
            with self.subTest(input_end_date):
                mock_upload.reset_mock()
                mock_argument_parser.parse_args.return_value.end_date = input_end_date
                mock_argument_parser.parse_args.return_value.data_level = data_level
                mock_argument_parser.parse_args.return_value.descriptor = descriptor

                mock_processor_class.return_value.process.return_value = [Mock()]

                imap_l3_processor()

                parser = mock_argparse.ArgumentParser()
                parser.add_argument.assert_has_calls([
                    call("--instrument"),
                    call("--data-level"),
                    call("--descriptor"),
                    call("--start-date"),
                    call("--end-date", required=False),
                    call("--version"),
                    call("--dependency"),
                    call("--upload-to-sdc", action="store_true", required=False,
                         help="Upload completed output files to the IMAP SDC.")
                ])

                expected_input_metadata = InputMetadata("glows", data_level, datetime(year=2016, month=6, day=30),
                                                        expected_end_date, "v092", descriptor=descriptor)

                mock_processor_class.assert_called_with(imap_data_access_dependency, expected_input_metadata)

                mock_processor.process.assert_called()
                mock_upload.assert_called_once_with(mock_processor_class.return_value.process.return_value[0])

    @patch('imap_l3_data_processor.SweProcessor')
    @patch('imap_l3_data_processor.argparse')
    @patch('imap_l3_data_processor.ProcessingInputCollection')
    @patch('imap_l3_data_processor.imap_data_access.upload')
    def test_runs_swe_processor_when_instrument_argument_is_swe(self, mock_upload, mock_processing_input_collection,
                                                                mock_argparse,
                                                                mock_processor_class):
        cases = [("20170630", datetime(2017, 6, 30)), (None, datetime(2016, 6, 30))]

        instrument_argument = "swe"
        data_level_argument = "l3"
        start_date_argument = "20160630"
        version_argument = "v092"
        descriptor_argument = "pitch-angle"
        science_input = ScienceInput("imap_swe_l1_science_20250101_v112.cdf")
        imap_data_access_dependency = ProcessingInputCollection(science_input)

        mock_processing_input_collection.return_value = imap_data_access_dependency
        mock_processing_input_collection.deserialize = Mock()
        mock_processing_input_collection.get_science_inputs = Mock(return_value=[])

        mock_argument_parser = mock_argparse.ArgumentParser.return_value

        mock_argument_parser.parse_args.return_value.instrument = instrument_argument
        mock_argument_parser.parse_args.return_value.data_level = data_level_argument
        mock_argument_parser.parse_args.return_value.dependency = imap_data_access_dependency.serialize()
        mock_argument_parser.parse_args.return_value.start_date = start_date_argument
        mock_argument_parser.parse_args.return_value.version = version_argument
        mock_argument_parser.parse_args.return_value.descriptor = descriptor_argument

        mock_processor = mock_processor_class.return_value

        for input_end_date, expected_end_date in cases:
            with self.subTest(input_end_date):
                mock_upload.reset_mock()

                mock_argument_parser.parse_args.return_value.end_date = input_end_date

                mock_processor_class.return_value.process.return_value = [Mock()]

                imap_l3_processor()

                parser = mock_argparse.ArgumentParser()
                parser.add_argument.assert_has_calls([
                    call("--instrument"),
                    call("--data-level"),
                    call("--descriptor"),
                    call("--start-date"),
                    call("--end-date", required=False),
                    call("--version"),
                    call("--dependency"),
                    call("--upload-to-sdc", action="store_true", required=False,
                         help="Upload completed output files to the IMAP SDC.")
                ])

                expected_input_metadata = InputMetadata("swe", "l3", datetime(year=2016, month=6, day=30),
                                                        expected_end_date, "v092", "pitch-angle")

                mock_processor_class.assert_called_with(imap_data_access_dependency, expected_input_metadata)

                mock_processor.process.assert_called()
                mock_upload.assert_called_once_with(mock_processor_class.return_value.process.return_value[0])

    @patch('imap_l3_data_processor.spiceypy')
    @patch('imap_l3_data_processor.argparse')
    @patch('imap_l3_data_processor.imap_data_access.download')
    @patch('imap_l3_data_processor.ProcessingInputCollection')
    @patch('imap_l3_data_processor.imap_data_access.upload')
    @patch('imap_l3_data_processor.SwapiProcessor')
    def test_get_spice_kernels_based_on_input_collection(self, _, __, mock_processing_input_collection, mock_download,
                                                         mock_arg_parser_class, mock_spicepy):
        ancillary_input = AncillaryInput("imap_swe_ancillary_20250101_v112.cdf")
        spice_input_1 = SPICEInput("naif0012.tls")
        spice_input_2 = SPICEInput("imap_sclk_0012.tls")
        imap_data_access_dependency = ProcessingInputCollection(spice_input_1, ancillary_input, spice_input_2)
        imap_data_access_dependency.deserialize = Mock()
        mock_processing_input_collection.return_value = imap_data_access_dependency

        mock_argument_parser = mock_arg_parser_class.ArgumentParser.return_value

        mock_argument_parser.parse_args.return_value.instrument = "swapi"
        mock_argument_parser.parse_args.return_value.data_level = "l3a"
        mock_argument_parser.parse_args.return_value.start_date = "20160630"
        mock_argument_parser.parse_args.return_value.end_date = "20160630"
        mock_argument_parser.parse_args.return_value.version = "v101"
        mock_argument_parser.parse_args.return_value.descriptor = "dont care"
        mock_argument_parser.parse_args.return_value.dependency = "also dont care"

        mock_download.side_effect = [
            Path("naif0012.tls"),
            Path("imap_sclk_0012.tls")
        ]

        imap_l3_processor()
        expected_spice_paths = imap_data_access_dependency.get_file_paths(data_type='spice')
        self.assertEqual(2, mock_download.call_count)
        mock_download.assert_has_calls([
            call(expected_spice_paths[0]),
            call(expected_spice_paths[1])
        ])
        mock_spicepy.furnsh.assert_has_calls(
            [
                call("naif0012.tls"),
                call("imap_sclk_0012.tls")
            ])

    @patch('imap_l3_data_processor.ProcessingInputCollection')
    @patch('imap_l3_data_processor.imap_data_access.upload')
    @patch('imap_l3_data_processor.SweProcessor')
    @patch('imap_l3_data_processor.argparse')
    def test_uses_input_from_processing_input_collection(self, mock_argparse, mock_processor_class, mock_upload,
                                                         mock_processing_input_collection):
        cases = [("20170630", datetime(2017, 6, 30)), (None, datetime(2016, 6, 30))]

        instrument_argument = "swe"
        data_level_argument = "l3"
        start_date_argument = "20160630"
        version_argument = "v092"
        descriptor_argument = "pitch-angle"
        science_input_1 = ScienceInput("imap_swe_l1_sci_20250101_v112.cdf", "imap_swe_l1_sci_20250102_v112.cdf")
        science_input_2 = ScienceInput("imap_mag_l1d_norm-mago_20250101_v112.cdf")
        ancillary_input = AncillaryInput("imap_swe_ancillary_20250101_v112.cdf")
        imap_data_access_dependency = ProcessingInputCollection(science_input_1, science_input_2, ancillary_input)

        mock_processing_input_collection.return_value = imap_data_access_dependency
        mock_processing_input_collection.deserialize = Mock()
        mock_processing_input_collection.get_science_inputs = Mock(return_value=[])

        mock_argument_parser = mock_argparse.ArgumentParser.return_value

        mock_argument_parser.parse_args.return_value.instrument = instrument_argument
        mock_argument_parser.parse_args.return_value.data_level = data_level_argument
        mock_argument_parser.parse_args.return_value.dependency = imap_data_access_dependency.serialize()
        mock_argument_parser.parse_args.return_value.start_date = start_date_argument
        mock_argument_parser.parse_args.return_value.version = version_argument
        mock_argument_parser.parse_args.return_value.descriptor = descriptor_argument

        mock_processor = mock_processor_class.return_value

        for input_end_date, expected_end_date in cases:
            with self.subTest(input_end_date):
                mock_upload.reset_mock()

                mock_argument_parser.parse_args.return_value.end_date = input_end_date

                mock_processor_class.return_value.process.return_value = [Mock()]

                imap_l3_processor()

                parser = mock_argparse.ArgumentParser()
                parser.add_argument.assert_has_calls([
                    call("--instrument"),
                    call("--data-level"),
                    call("--descriptor"),
                    call("--start-date"),
                    call("--end-date", required=False),
                    call("--version"),
                    call("--dependency"),
                    call("--upload-to-sdc", action="store_true", required=False,
                         help="Upload completed output files to the IMAP SDC.")
                ])

                expected_input_metadata = InputMetadata("swe", "l3", datetime(year=2016, month=6, day=30),
                                                        expected_end_date, "v092", "pitch-angle")

                mock_processor_class.assert_called_with(imap_data_access_dependency, expected_input_metadata)

                mock_processor.process.assert_called()
                mock_upload.assert_called_once_with(mock_processor_class.return_value.process.return_value[0])

    @patch('imap_l3_data_processor.ProcessingInputCollection')
    @patch('imap_l3_data_processor.imap_data_access.upload')
    @patch('imap_l3_data_processor.HitProcessor')
    @patch('imap_l3_data_processor.argparse')
    def test_runs_hit_processor_when_instrument_argument_is_hit(self, mock_argparse, mock_processor_class, mock_upload,
                                                                mock_processing_input_collection):
        cases = [("20170630", datetime(2017, 6, 30), 'l3b'),
                 (None, datetime(2016, 6, 30), 'l3a')]

        instrument_argument = "hit"
        start_date_argument = "20160630"
        version_argument = "v092"
        descriptor_argument = "A descriptor"
        science_input = ScienceInput("imap_hit_l1_science_20250101_v112.cdf")
        imap_data_access_dependency = ProcessingInputCollection(science_input)

        mock_processing_input_collection.return_value = imap_data_access_dependency
        mock_processing_input_collection.deserialize = Mock()
        mock_processing_input_collection.get_science_inputs = Mock(return_value=[])

        mock_argument_parser = mock_argparse.ArgumentParser.return_value
        mock_argument_parser.parse_args.return_value.instrument = instrument_argument
        mock_argument_parser.parse_args.return_value.dependency = imap_data_access_dependency.serialize()
        mock_argument_parser.parse_args.return_value.start_date = start_date_argument
        mock_argument_parser.parse_args.return_value.version = version_argument
        mock_argument_parser.parse_args.return_value.descriptor = descriptor_argument

        for input_end_date, expected_end_date, data_level in cases:
            with self.subTest(data_level):
                mock_upload.reset_mock()

                mock_argument_parser.parse_args.return_value.data_level = data_level
                mock_argument_parser.parse_args.return_value.end_date = input_end_date

                mock_processor_class.return_value.process.return_value = [Mock()]

                imap_l3_processor()

                parser = mock_argparse.ArgumentParser()
                parser.add_argument.assert_has_calls([
                    call("--instrument"),
                    call("--data-level"),
                    call("--descriptor"),
                    call("--start-date"),
                    call("--end-date", required=False),
                    call("--version"),
                    call("--dependency"),
                    call("--upload-to-sdc", action="store_true", required=False,
                         help="Upload completed output files to the IMAP SDC.")
                ])

                expected_input_metadata = InputMetadata("hit", data_level, datetime(year=2016, month=6, day=30),
                                                        expected_end_date, "v092", "A descriptor")

                mock_processor_class.assert_called_with(imap_data_access_dependency,
                                                        expected_input_metadata)
                mock_processor_class.return_value.process.assert_called()
                mock_upload.assert_called_once_with(mock_processor_class.return_value.process.return_value[0])

    @patch('imap_l3_data_processor.HiProcessor')
    @patch('imap_l3_data_processor.argparse')
    @patch('imap_l3_data_processor.ProcessingInputCollection')
    @patch('imap_l3_data_processor.imap_data_access.upload')
    def test_runs_hi_processor_when_instrument_argument_is_hi(self, mock_upload, mock_processing_input_collection,
                                                              mock_argparse, mock_processor_class):
        instrument_argument = "hi"
        start_date_argument = "20160630"
        version_argument = "v001"
        descriptor_argument = "A descriptor"
        science_input = ScienceInput("imap_hi_l2_science_20250101_v001.cdf")
        imap_data_access_dependency = ProcessingInputCollection(science_input)
        mock_processing_input_collection.return_value = imap_data_access_dependency
        mock_processing_input_collection.deserialize = Mock()
        mock_processing_input_collection.get_science_inputs = Mock(return_value=[])

        mock_argument_parser = mock_argparse.ArgumentParser.return_value
        mock_argument_parser.parse_args.return_value.instrument = instrument_argument
        mock_argument_parser.parse_args.return_value.dependency = imap_data_access_dependency.serialize()
        mock_argument_parser.parse_args.return_value.start_date = start_date_argument
        mock_argument_parser.parse_args.return_value.version = version_argument
        mock_argument_parser.parse_args.return_value.descriptor = descriptor_argument

        mock_argument_parser.parse_args.return_value.data_level = "l3"
        mock_argument_parser.parse_args.return_value.end_date = None

        mock_processor_class.return_value.process.return_value = [Mock()]

        imap_l3_processor()

        expected_input_metadata = InputMetadata("hi", "l3", datetime(year=2016, month=6, day=30),
                                                datetime(year=2016, month=6, day=30), "v001", "A descriptor")

        mock_processor_class.assert_called_with(imap_data_access_dependency,
                                                expected_input_metadata)
        mock_processor_class.return_value.process.assert_called()
        mock_upload.assert_called_once_with(mock_processor_class.return_value.process.return_value[0])

    @patch('imap_l3_data_processor.LoProcessor')
    @patch('imap_l3_data_processor.argparse')
    @patch('imap_l3_data_processor.ProcessingInputCollection')
    @patch('imap_l3_data_processor.imap_data_access.upload')
    def test_runs_lo_processor_when_instrument_argument_is_lo(self, mock_upload, mock_processing_input_collection,
                                                              mock_argparse, mock_processor_class):
        instrument_argument = "lo"
        start_date_argument = "20160630"
        version_argument = "v001"
        descriptor_argument = "A descriptor"
        science_input = ScienceInput("imap_lo_l2_science_20250101_v001.cdf")
        imap_data_access_dependency = ProcessingInputCollection(science_input)
        mock_processing_input_collection.return_value = imap_data_access_dependency
        mock_processing_input_collection.deserialize = Mock()
        mock_processing_input_collection.get_science_inputs = Mock(return_value=[])

        mock_argument_parser = mock_argparse.ArgumentParser.return_value
        mock_argument_parser.parse_args.return_value.instrument = instrument_argument
        mock_argument_parser.parse_args.return_value.dependency = imap_data_access_dependency.serialize()
        mock_argument_parser.parse_args.return_value.start_date = start_date_argument
        mock_argument_parser.parse_args.return_value.version = version_argument
        mock_argument_parser.parse_args.return_value.descriptor = descriptor_argument

        mock_argument_parser.parse_args.return_value.data_level = "l3"
        mock_argument_parser.parse_args.return_value.end_date = None

        mock_processor_class.return_value.process.return_value = [Mock()]

        imap_l3_processor()

        expected_input_metadata = InputMetadata("lo", "l3", datetime(year=2016, month=6, day=30),
                                                datetime(year=2016, month=6, day=30), "v001", "A descriptor")

        mock_processor_class.assert_called_with(imap_data_access_dependency,
                                                expected_input_metadata)
        mock_processor_class.return_value.process.assert_called()
        mock_upload.assert_called_once_with(mock_processor_class.return_value.process.return_value[0])

    @patch('imap_l3_data_processor.CodiceHiProcessor')
    @patch('imap_l3_data_processor.argparse')
    @patch('imap_l3_data_processor.ProcessingInputCollection')
    @patch('imap_l3_data_processor.imap_data_access.upload')
    def test_runs_codice_hi_processor_when_instrument_argument_is_codice_and_descriptor_starts_with_hi(self,
                                                                                                       mock_upload,
                                                                                                       mock_processing_input_collection,
                                                                                                       mock_argparse,
                                                                                                       mock_processor_class):
        instrument_argument = "codice"
        start_date_argument = "20160630"
        version_argument = "v001"
        descriptor_argument = "hi-descriptor"
        science_input = ScienceInput("imap_codice_l2_science_20250101_v001.cdf")
        imap_data_access_dependency = ProcessingInputCollection(science_input)
        mock_processing_input_collection.return_value = imap_data_access_dependency
        mock_processing_input_collection.deserialize = Mock()
        mock_processing_input_collection.get_science_inputs = Mock(return_value=[])

        mock_argument_parser = mock_argparse.ArgumentParser.return_value
        mock_argument_parser.parse_args.return_value.instrument = instrument_argument
        mock_argument_parser.parse_args.return_value.dependency = imap_data_access_dependency.serialize()
        mock_argument_parser.parse_args.return_value.start_date = start_date_argument
        mock_argument_parser.parse_args.return_value.version = version_argument
        mock_argument_parser.parse_args.return_value.descriptor = descriptor_argument

        mock_argument_parser.parse_args.return_value.data_level = "l3"
        mock_argument_parser.parse_args.return_value.end_date = None

        mock_processor_class.return_value.process.return_value = [Mock()]

        imap_l3_processor()

        expected_input_metadata = InputMetadata("codice", "l3", datetime(year=2016, month=6, day=30),
                                                datetime(year=2016, month=6, day=30), "v001", "hi-descriptor")

        mock_processor_class.assert_called_with(imap_data_access_dependency,
                                                expected_input_metadata)
        mock_processor_class.return_value.process.assert_called()
        mock_upload.assert_called_once_with(mock_processor_class.return_value.process.return_value[0])

    @patch('imap_l3_data_processor.CodiceLoProcessor')
    @patch('imap_l3_data_processor.argparse')
    @patch('imap_l3_data_processor.ProcessingInputCollection')
    @patch('imap_l3_data_processor.imap_data_access.upload')
    def test_runs_codice_lo_processor_when_instrument_argument_is_codice_and_descriptor_starts_with_lo(self,
                                                                                                       mock_upload,
                                                                                                       mock_processing_input_collection,
                                                                                                       mock_argparse,
                                                                                                       mock_processor_class):
        instrument_argument = "codice"
        start_date_argument = "20160630"
        version_argument = "v001"
        descriptor_argument = "lo-descriptor"
        science_input = ScienceInput("imap_codice_l2_science_20250101_v001.cdf")

        imap_data_access_dependency = ProcessingInputCollection(science_input)
        mock_processing_input_collection.return_value = imap_data_access_dependency
        mock_processing_input_collection.deserialize = Mock()
        mock_processing_input_collection.get_science_inputs = Mock(return_value=[])

        mock_argument_parser = mock_argparse.ArgumentParser.return_value
        mock_argument_parser.parse_args.return_value.instrument = instrument_argument
        mock_argument_parser.parse_args.return_value.dependency = imap_data_access_dependency.serialize()
        mock_argument_parser.parse_args.return_value.start_date = start_date_argument
        mock_argument_parser.parse_args.return_value.version = version_argument
        mock_argument_parser.parse_args.return_value.descriptor = descriptor_argument

        mock_argument_parser.parse_args.return_value.data_level = "l3a"
        mock_argument_parser.parse_args.return_value.end_date = None

        mock_processor_class.return_value.process.return_value = [Mock()]

        imap_l3_processor()

        expected_input_metadata = InputMetadata("codice", "l3a", datetime(year=2016, month=6, day=30),
                                                datetime(year=2016, month=6, day=30), "v001", "lo-descriptor")

        mock_processor_class.assert_called_with(imap_data_access_dependency,
                                                expected_input_metadata)
        mock_processor_class.return_value.process.assert_called()
        mock_upload.assert_called_once_with(mock_processor_class.return_value.process.return_value[0])

    @patch('imap_l3_data_processor.UltraProcessor')
    @patch('imap_l3_data_processor.argparse')
    @patch('imap_l3_data_processor.ProcessingInputCollection')
    @patch('imap_l3_data_processor.imap_data_access.upload')
    def test_runs_ultra_processor_when_argument_is_ultra(self, mock_upload, mock_processing_input_collection,
                                                         mock_argparse,
                                                         mock_ultra_processor):
        instrument_arg = "ultra"
        start_date_arg = "20250418"
        version_arg = "v001"
        descriptor_arg = "desc"
        science_input = ScienceInput("imap_ultra_l3_science_20250418_v001.cdf")

        processing_input_collection = ProcessingInputCollection(science_input)
        mock_processing_input_collection.return_value = processing_input_collection
        mock_processing_input_collection.deserialize = Mock()
        mock_processing_input_collection.get_science_inputs = Mock(return_value=[])

        mock_argument_parser = mock_argparse.ArgumentParser.return_value
        mock_argument_parser.parse_args.return_value.instrument = instrument_arg
        mock_argument_parser.parse_args.return_value.dependency = processing_input_collection.serialize()
        mock_argument_parser.parse_args.return_value.start_date = start_date_arg
        mock_argument_parser.parse_args.return_value.version = version_arg
        mock_argument_parser.parse_args.return_value.descriptor = descriptor_arg

        mock_argument_parser.parse_args.return_value.data_level = "l3"
        mock_argument_parser.parse_args.return_value.end_date = None

        mock_ultra_processor.return_value.process.return_value = [Mock()]

        imap_l3_processor()

        expected_input_metadata = InputMetadata("ultra", "l3", datetime(year=2025, month=4, day=18),
                                                datetime(year=2025, month=4, day=18), "v001", descriptor_arg)

        mock_ultra_processor.assert_called_with(processing_input_collection, expected_input_metadata)
        mock_ultra_processor.return_value.process.assert_called()
        mock_upload.assert_called_once_with(mock_ultra_processor.return_value.process.return_value[0])

    @patch('imap_l3_data_processor.GlowsProcessor')
    @patch('imap_l3_data_processor.argparse')
    @patch('imap_l3_data_processor.ProcessingInputCollection')
    @patch('imap_l3_data_processor.imap_data_access.upload')
    def test_uploads_multiple_files(self, mock_upload, mock_processing_input_collection, mock_argparse,
                                    mock_glows_processor):
        instrument_arg = "glows"
        start_date_arg = "20250101"
        version_arg = "v001"
        descriptor_arg = "desc"
        science_input = ScienceInput("imap_glows_l3_science_20250101_v001.cdf")

        processing_input_collection = ProcessingInputCollection(science_input)
        mock_processing_input_collection.return_value = processing_input_collection
        mock_processing_input_collection.deserialize = Mock()
        mock_processing_input_collection.get_science_inputs = Mock(return_value=[])

        mock_argument_parser = mock_argparse.ArgumentParser.return_value
        mock_argument_parser.parse_args.return_value.instrument = instrument_arg
        mock_argument_parser.parse_args.return_value.dependency = processing_input_collection.serialize()
        mock_argument_parser.parse_args.return_value.start_date = start_date_arg
        mock_argument_parser.parse_args.return_value.version = version_arg
        mock_argument_parser.parse_args.return_value.descriptor = descriptor_arg

        mock_argument_parser.parse_args.return_value.data_level = "l3"
        mock_argument_parser.parse_args.return_value.end_date = None

        mock_glows_processor.return_value.process.return_value = [sentinel.one, sentinel.two, sentinel.three]

        imap_l3_processor()

        mock_upload.assert_has_calls([
            call(sentinel.one),
            call(sentinel.two),
            call(sentinel.three)
        ])

    @patch('imap_l3_data_processor.GlowsProcessor')
    @patch('imap_l3_data_processor.argparse')
    @patch('imap_l3_data_processor.ProcessingInputCollection')
    @patch('imap_l3_data_processor.imap_data_access.upload')
    def test_does_not_upload_files_if_upload_to_sdc_flag_is_set_to_false(self, mock_upload,
                                                                         mock_processing_input_collection,
                                                                         mock_argparse,
                                                                         mock_glows_processor):
        processing_input_collection = ProcessingInputCollection(ScienceInput("imap_glows_l3_science_20250101_v001.cdf"))
        mock_argument_parser = mock_argparse.ArgumentParser.return_value
        mock_argument_parser.parse_args.return_value.instrument = "glows"
        mock_argument_parser.parse_args.return_value.dependency = processing_input_collection.serialize()
        mock_argument_parser.parse_args.return_value.start_date = "20250101"
        mock_argument_parser.parse_args.return_value.version = "v001"
        mock_argument_parser.parse_args.return_value.descriptor = "desc"
        mock_argument_parser.parse_args.return_value.upload_to_sdc = False
        mock_argument_parser.parse_args.return_value.data_level = "l3"
        mock_argument_parser.parse_args.return_value.end_date = None

        mock_processing_input_collection.return_value = processing_input_collection
        mock_processing_input_collection.deserialize = Mock()
        mock_processing_input_collection.get_science_inputs = Mock(return_value=[])

        mock_glows_processor.return_value.process.return_value = ["something!"]

        imap_l3_processor()

        mock_glows_processor.return_value.process.assert_called()
        mock_upload.assert_not_called()

    @patch('imap_l3_data_processor.argparse')
    @patch('imap_l3_data_processor.imap_data_access.upload')
    def test_throws_exception_for_unimplemented_instrument(self, mock_upload, mock_argparse):
        instrument_argument = "new_instrument"
        data_level_argument = "l3a"
        start_date_argument = "20160630"
        end_date_argument = None
        version_argument = "v092"
        science_input = ScienceInput("imap_glows_l1_science_20250101_v112.cdf")
        imap_data_access_dependency = ProcessingInputCollection(science_input)

        mock_argument_parser = mock_argparse.ArgumentParser.return_value

        mock_argument_parser.parse_args.return_value.instrument = instrument_argument
        mock_argument_parser.parse_args.return_value.data_level = data_level_argument
        mock_argument_parser.parse_args.return_value.dependency = imap_data_access_dependency.serialize()
        mock_argument_parser.parse_args.return_value.start_date = start_date_argument
        mock_argument_parser.parse_args.return_value.end_date = end_date_argument
        mock_argument_parser.parse_args.return_value.version = version_argument

        with self.assertRaises(NotImplementedError) as exception_manager:
            imap_l3_processor()
        self.assertEqual(str(exception_manager.exception),
                         "Level l3a data processing has not yet been implemented for new_instrument")
        mock_upload.assert_not_called()

    @patch('imap_l3_data_processor.argparse')
    @patch('imap_l3_data_processor.imap_data_access.upload')
    def test_throws_exception_for_codice_descriptor_not_matching_hi_or_lo(self, mock_upload, mock_argparse):
        instrument_argument = "codice"
        data_level_argument = "l3a"
        start_date_argument = "20160630"
        end_date_argument = None
        version_argument = "v092"
        descriptor_argument = "bad"
        science_input = ScienceInput("imap_glows_l1_science_20250101_v112.cdf")
        imap_data_access_dependency = ProcessingInputCollection(science_input)

        mock_argument_parser = mock_argparse.ArgumentParser.return_value

        mock_argument_parser.parse_args.return_value.instrument = instrument_argument
        mock_argument_parser.parse_args.return_value.data_level = data_level_argument
        mock_argument_parser.parse_args.return_value.dependency = imap_data_access_dependency.serialize()
        mock_argument_parser.parse_args.return_value.start_date = start_date_argument
        mock_argument_parser.parse_args.return_value.end_date = end_date_argument
        mock_argument_parser.parse_args.return_value.version = version_argument
        mock_argument_parser.parse_args.return_value.descriptor = descriptor_argument

        with self.assertRaises(NotImplementedError) as exception_manager:
            imap_l3_processor()
        self.assertEqual(str(exception_manager.exception),
                         "Unknown descriptor 'bad' for codice instrument")
        mock_upload.assert_not_called()

    @patch('imap_l3_data_processor.argparse')
    @patch('imap_l3_data_processor.imap_data_access.upload')
    def test_throws_exception_when_attempting_to_process_non_l3_data_levels(self, mock_upload, mock_argparse):
        instrument_argument = "swapi"
        data_level_argument = "l4"
        start_date_argument = "20160630"
        end_date_argument = "20160701"
        version_argument = "v092"
        science_input = ScienceInput("imap_swapi_l3_science_20250101_v112.cdf")
        imap_data_access_dependency = ProcessingInputCollection(science_input)

        mock_argument_parser = mock_argparse.ArgumentParser.return_value

        mock_argument_parser.parse_args.return_value.instrument = instrument_argument
        mock_argument_parser.parse_args.return_value.data_level = data_level_argument
        mock_argument_parser.parse_args.return_value.dependency = imap_data_access_dependency.serialize()
        mock_argument_parser.parse_args.return_value.start_date = start_date_argument
        mock_argument_parser.parse_args.return_value.end_date = end_date_argument
        mock_argument_parser.parse_args.return_value.version = version_argument

        with self.assertRaises(NotImplementedError) as exception_manager:
            imap_l3_processor()
        self.assertEqual(str(exception_manager.exception),
                         "Level l4 data processing has not yet been implemented for swapi")
        mock_upload.assert_not_called()
