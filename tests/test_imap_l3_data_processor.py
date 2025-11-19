from datetime import datetime
from pathlib import Path
from unittest import TestCase
from unittest.mock import patch, call, Mock, sentinel

from imap_data_access.processing_input import ScienceInput, ProcessingInputCollection, \
    AncillaryInput, SPICEInput

from imap_l3_data_processor import imap_l3_processor
from imap_l3_processing.hi.l3.hi_l3_initializer import HI_SP_MAP_DESCRIPTORS
from imap_l3_processing.lo.l3.lo_initializer import LO_SP_MAP_DESCRIPTORS
from imap_l3_processing.maps.map_initializer import PossibleMapToProduce
from imap_l3_processing.models import InputMetadata
from imap_l3_processing.ultra.l3.ultra_initializer import ULTRA_SP_MAP_DESCRIPTORS


class TestImapL3DataProcessor(TestCase):
    @patch('imap_l3_data_processor.ProcessingInputCollection')
    @patch('imap_l3_data_processor.imap_data_access.upload')
    @patch('imap_l3_data_processor.SwapiProcessor')
    @patch('imap_l3_data_processor.SweProcessor')
    @patch('imap_l3_data_processor.HitProcessor')
    @patch('imap_l3_data_processor.HiProcessor')
    @patch('imap_l3_data_processor.LoProcessor')
    @patch('imap_l3_data_processor.UltraProcessor')
    @patch('imap_l3_data_processor.GlowsProcessor')
    @patch('imap_l3_data_processor.CodiceLoProcessor')
    @patch('imap_l3_data_processor.CodiceHiProcessor')
    @patch('imap_l3_data_processor.argparse')
    def test_invokes_correct_processor(self, mock_argparse, mock_codice_hi, mock_codice_lo, mock_glows,
                                       mock_ultra, mock_lo, mock_hi, mock_hit, mock_swe,
                                       mock_swapi, mock_upload,
                                       mock_processing_input):
        cases = [
            ("swapi", "l3a", "proton", mock_swapi),
            ("swapi", "l3b", "combined", mock_swapi),
            ("glows", "l3a", "hist", mock_glows),
            ("glows", "l3b", "ion-rate-profile", mock_glows),
            ("swe", "l3", "sci", mock_swe),
            ("hit", "l3", "macropixel", mock_hit),
            ("hi", "l3", "hi-descriptor", mock_hi),
            ("ultra", "l3", "ultra-descriptor", mock_ultra),
            ("lo", "l3", "lo-descriptor", mock_lo),
            ("codice", "l3a", "hi-direct-events", mock_codice_hi),
            ("codice", "l3b", "hi-pitch-angle", mock_codice_hi),
            ("codice", "l3a", "lo-direct-events", mock_codice_lo),
        ]

        for instrument, data_level, descriptor, expected_processor in cases:
            mock_argparse.reset_mock()
            expected_processor.reset_mock()
            mock_upload.reset_mock()
            mock_processing_input.reset_mock()
            with self.subTest(f'{instrument}{data_level}'):
                mock_argument_parser = mock_argparse.ArgumentParser.return_value
                mock_argument_parser.parse_args.return_value.instrument = instrument
                mock_argument_parser.parse_args.return_value.data_level = data_level
                mock_argument_parser.parse_args.return_value.dependency = "dependency_string"
                mock_argument_parser.parse_args.return_value.start_date = "20250101"
                mock_argument_parser.parse_args.return_value.end_date = None
                mock_argument_parser.parse_args.return_value.repointing = "repoint00022"
                mock_argument_parser.parse_args.return_value.version = sentinel.version
                mock_argument_parser.parse_args.return_value.descriptor = descriptor

                expected_input_metadata = InputMetadata(instrument, data_level, datetime(2025, 1, 1),
                                                        datetime(2025, 1, 1),
                                                        sentinel.version, descriptor=descriptor,
                                                        repointing=22)

                expected_processor.return_value.process.return_value = [sentinel.cdf]

                imap_l3_processor()

                mock_argument_parser.add_argument.assert_has_calls([
                    call("--instrument"),
                    call("--data-level"),
                    call("--descriptor"),
                    call("--start-date"),
                    call("--end-date", required=False),
                    call("--repointing", required=False),
                    call("--version"),
                    call("--dependency"),
                    call("--upload-to-sdc", action="store_true", required=False,
                         help="Upload completed output files to the IMAP SDC.")
                ])

                mock_processing_input.return_value.deserialize.assert_called_once_with("dependency_string")

                expected_processor.assert_called_once_with(mock_processing_input.return_value, expected_input_metadata)
                mock_upload.assert_called_once_with(sentinel.cdf)

    @patch('imap_l3_data_processor.UltraInitializer')
    @patch('imap_l3_data_processor.UltraProcessor')
    @patch('imap_l3_data_processor.HiL3Initializer')
    @patch('imap_l3_data_processor.HiProcessor')
    @patch('imap_l3_data_processor.LoInitializer')
    @patch('imap_l3_data_processor.LoProcessor')
    @patch('imap_l3_data_processor.imap_data_access.upload')
    @patch('imap_l3_data_processor.argparse')
    def test_scheduled_lo_job_invokes_initializer(self, mock_argparse, mock_upload,
                                                  mock_lo_processor_class, mock_lo_initializer_class,
                                                  mock_hi_processor_class, mock_hi_initializer_class,
                                                  mock_ultra_processor_class, mock_ultra_initializer_class,
                                                  ):

        test_cases = [
            ("hi", mock_hi_initializer_class, mock_hi_processor_class, HI_SP_MAP_DESCRIPTORS),
            ("lo", mock_lo_initializer_class, mock_lo_processor_class, LO_SP_MAP_DESCRIPTORS),
            ("ultra", mock_ultra_initializer_class, mock_ultra_processor_class, ULTRA_SP_MAP_DESCRIPTORS),
        ]

        for instrument, mock_initializer_class, mock_processor_class, expected_descriptors in test_cases:
            mock_upload.reset_mock()
            mock_argparse.reset_mock()

            with self.subTest(instrument=instrument):
                data_level = "l3"
                descriptor = "all-maps"
                mock_argument_parser = mock_argparse.ArgumentParser.return_value
                mock_argument_parser.parse_args.return_value.instrument = instrument
                mock_argument_parser.parse_args.return_value.data_level = data_level
                mock_argument_parser.parse_args.return_value.dependency = "[]"
                mock_argument_parser.parse_args.return_value.start_date = "20250101"
                mock_argument_parser.parse_args.return_value.end_date = None
                mock_argument_parser.parse_args.return_value.repointing = "repoint00022"
                mock_argument_parser.parse_args.return_value.version = sentinel.version
                mock_argument_parser.parse_args.return_value.descriptor = descriptor

                expected_input_metadata = InputMetadata(instrument, data_level, datetime(2025, 1, 1),
                                                        datetime(2025, 1, 1),
                                                        sentinel.version, descriptor=descriptor,
                                                        )

                mock_initializer = mock_initializer_class.return_value
                mock_processor = mock_processor_class.return_value

                possible_map_to_produce = PossibleMapToProduce(set(), expected_input_metadata)
                mock_initializer.get_maps_that_should_be_produced.return_value = [possible_map_to_produce]

                mock_processor.process.return_value = [sentinel.cdf]

                imap_l3_processor()

                mock_initializer.get_maps_that_should_be_produced.assert_has_calls([
                    call(descriptor) for descriptor in expected_descriptors
                ])

                self.assertEqual(len(expected_descriptors), mock_processor_class.call_count)

                self.assertEqual(len(expected_descriptors), mock_initializer.furnish_spice_dependencies.call_count)
                mock_initializer.furnish_spice_dependencies.assert_called_with(possible_map_to_produce)

                mock_processor_class.assert_called_with(possible_map_to_produce.processing_input_collection,
                                                        expected_input_metadata)
                self.assertEqual(len(expected_descriptors), mock_processor.process.call_count)

                self.assertEqual(len(expected_descriptors), mock_upload.call_count)
                mock_upload.assert_called_with(sentinel.cdf)

    @patch('imap_l3_data_processor.imap_data_access.upload')
    @patch('imap_l3_data_processor.LoInitializer')
    @patch('imap_l3_data_processor.LoProcessor')
    @patch('imap_l3_data_processor.argparse')
    def test_failing_to_produce_an_sp_map_continues(self, mock_argparse, mock_lo_processor, mock_lo_initializer_class,
                                                    mock_upload):
        mock_lo_initializer = mock_lo_initializer_class.return_value

        instrument = "lo"
        data_level = "l3"
        descriptor = "all-maps"
        mock_argument_parser = mock_argparse.ArgumentParser.return_value
        mock_argument_parser.parse_args.return_value.instrument = instrument
        mock_argument_parser.parse_args.return_value.data_level = data_level
        mock_argument_parser.parse_args.return_value.dependency = "[]"
        mock_argument_parser.parse_args.return_value.start_date = "20250101"
        mock_argument_parser.parse_args.return_value.end_date = None
        mock_argument_parser.parse_args.return_value.repointing = "repoint00022"
        mock_argument_parser.parse_args.return_value.version = sentinel.version
        mock_argument_parser.parse_args.return_value.descriptor = descriptor

        map_to_produce_1 = PossibleMapToProduce(set(), Mock())
        map_to_produce_2 = PossibleMapToProduce(set(), Mock())
        map_to_produce_3 = PossibleMapToProduce(set(), Mock())
        mock_lo_initializer.get_maps_that_should_be_produced.side_effect = [
            [map_to_produce_1, map_to_produce_2, map_to_produce_3],
        ]

        mock_lo_processor.return_value.process.side_effect = [
            [Path("lo_sp_map_1.cdf")],
            ValueError("something went wrong!"),
            [Path("lo_sp_map_2.cdf")]
        ]

        imap_l3_processor()

        self.assertEqual(3, mock_lo_processor.call_count)

        mock_upload.assert_has_calls([
            call(Path("lo_sp_map_1.cdf")),
            call(Path("lo_sp_map_2.cdf")),
        ])

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
        mock_argument_parser.parse_args.return_value.repointing = None

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
        science_input_2 = ScienceInput("imap_mag_l1d_norm-dsrf_20250101_v112.cdf")
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
        mock_argument_parser.parse_args.return_value.repointing = None

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
                    call("--repointing", required=False),
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
        mock_argument_parser.parse_args.return_value.repointing = None

        mock_argument_parser.parse_args.return_value.data_level = "l3b"
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
        mock_argument_parser.parse_args.return_value.data_level = "l3a"
        mock_argument_parser.parse_args.return_value.end_date = None
        mock_argument_parser.parse_args.return_value.repointing = None

        mock_processing_input_collection.return_value = processing_input_collection
        mock_processing_input_collection.deserialize = Mock()
        mock_processing_input_collection.get_science_inputs = Mock(return_value=[])

        mock_glows_processor.return_value.process.return_value = ["something!"]

        imap_l3_processor()

        mock_glows_processor.return_value.process.assert_called()
        mock_upload.assert_not_called()

    @patch('imap_l3_data_processor.GlowsProcessor')
    @patch('imap_l3_data_processor.argparse')
    @patch('imap_l3_data_processor.ProcessingInputCollection')
    @patch('imap_l3_data_processor.imap_data_access.upload')
    def test_upload_fails_tries_all_uploads_and_raises_exception(self, mock_upload,
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
        mock_argument_parser.parse_args.return_value.upload_to_sdc = True
        mock_argument_parser.parse_args.return_value.data_level = "l3a"
        mock_argument_parser.parse_args.return_value.end_date = None
        mock_argument_parser.parse_args.return_value.repointing = None

        mock_processing_input_collection.return_value = processing_input_collection
        mock_processing_input_collection.deserialize = Mock()
        mock_processing_input_collection.get_science_inputs = Mock(return_value=[])

        mock_glows_processor.return_value.process.return_value = ["data_file_1.cdf", "data_file_2.cdf",
                                                                  "data_file_3.cdf", "data_file_4.cdf"]

        error1 = ValueError("Failure uploading!")
        error2 = ValueError("Failure uploading 2")
        mock_upload.side_effect = [None, error1, error2, None]

        with self.assertRaises(IOError) as exception_ctx:
            imap_l3_processor()

        self.assertEqual((f"Failed to upload some files: {[error1, error2]}",), exception_ctx.exception.args)

        mock_glows_processor.return_value.process.assert_called()

        mock_upload.assert_has_calls([
            call("data_file_1.cdf"),
            call("data_file_2.cdf"),
            call("data_file_3.cdf"),
            call("data_file_4.cdf"),
        ])

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
        mock_argument_parser.parse_args.return_value.repointing = None

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
        mock_argument_parser.parse_args.return_value.repointing = None

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
        mock_argument_parser.parse_args.return_value.repointing = None

        with self.assertRaises(NotImplementedError) as exception_manager:
            imap_l3_processor()
        self.assertEqual(str(exception_manager.exception),
                         "Level l4 data processing has not yet been implemented for swapi")
        mock_upload.assert_not_called()
