from datetime import datetime
from unittest import TestCase
from unittest.mock import patch, call

from imap_l3_data_processor import imap_l3_processor
from imap_l3_processing.models import InputMetadata, UpstreamDataDependency


class TestImapL3DataProcessor(TestCase):
    @patch('imap_l3_data_processor.SwapiProcessor')
    @patch('imap_l3_data_processor.argparse')
    def test_runs_swapi_processor_when_instrument_argument_is_swapi(self, mock_argparse, mock_swapi_processor_class):

        cases = [("20170630", datetime(2017, 6, 30)), (None, datetime(2016, 6, 30))]

        instrument_argument = "swapi"
        data_level_argument = "l3a"
        start_date_argument = "20160630"
        version_argument = "v092"
        descriptor_argument = "proton"
        dependencies_argument = (
            "[{'instrument':'not_swapi', 'data_level':'l1000', 'descriptor':'science', 'version':'v112',"
            "'start_date':'20250101'}]")

        mock_argument_parser = mock_argparse.ArgumentParser.return_value

        mock_argument_parser.parse_args.return_value.instrument = instrument_argument
        mock_argument_parser.parse_args.return_value.data_level = data_level_argument
        mock_argument_parser.parse_args.return_value.dependency = dependencies_argument
        mock_argument_parser.parse_args.return_value.start_date = start_date_argument
        mock_argument_parser.parse_args.return_value.version = version_argument
        mock_argument_parser.parse_args.return_value.version = version_argument
        mock_argument_parser.parse_args.return_value.descriptor = descriptor_argument

        mock_swapi_processor = mock_swapi_processor_class.return_value

        for input_end_date, expected_end_date in cases:
            with self.subTest(input_end_date):
                mock_argument_parser.parse_args.return_value.end_date = input_end_date

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

                expected_input_dependencies = [UpstreamDataDependency("not_swapi",
                                                                      "l1000",
                                                                      datetime(2025, 1, 1),
                                                                      None,
                                                                      "v112",
                                                                      "science")]

                expected_input_metadata = InputMetadata("swapi", "l3a", datetime(year=2016, month=6, day=30),
                                                        expected_end_date, "v092", "proton")

                mock_swapi_processor_class.assert_called_with(expected_input_dependencies, expected_input_metadata)

                mock_swapi_processor.process.assert_called()

    @patch('imap_l3_data_processor.GlowsProcessor')
    @patch('imap_l3_data_processor.argparse')
    def test_runs_glows_processor_when_instrument_argument_is_glows(self, mock_argparse, mock_processor_class):
        cases = [("20170630", datetime(2017, 6, 30), "l3a", "lightcurve"),
                 (None, datetime(2016, 6, 30), "l3a", "lightcurve"),
                 ("20170630", datetime(2017, 6, 30), "l3b", "ionization-rates"),
                 (None, datetime(2016, 6, 30), "l3b", "ionization-rates")]

        instrument_argument = "glows"
        start_date_argument = "20160630"
        version_argument = "v092"
        dependencies_argument = (
            "[{'instrument':'not_swapi', 'data_level':'l1000', 'descriptor':'science', 'version':'v112',"
            "'start_date':'20250101','end_date':'20250202'}]")

        mock_argument_parser = mock_argparse.ArgumentParser.return_value

        mock_argument_parser.parse_args.return_value.instrument = instrument_argument
        mock_argument_parser.parse_args.return_value.dependency = dependencies_argument
        mock_argument_parser.parse_args.return_value.start_date = start_date_argument
        mock_argument_parser.parse_args.return_value.version = version_argument

        mock_processor = mock_processor_class.return_value

        for input_end_date, expected_end_date, data_level, descriptor in cases:
            with self.subTest(input_end_date):
                mock_argument_parser.parse_args.return_value.end_date = input_end_date
                mock_argument_parser.parse_args.return_value.data_level = data_level
                mock_argument_parser.parse_args.return_value.descriptor = descriptor

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

                expected_input_dependencies = [UpstreamDataDependency("not_swapi",
                                                                      "l1000",
                                                                      datetime(2025, 1, 1),
                                                                      datetime(2025, 2, 2),
                                                                      "v112",
                                                                      "science")]

                expected_input_metadata = InputMetadata("glows", data_level, datetime(year=2016, month=6, day=30),
                                                        expected_end_date, "v092", descriptor=descriptor)

                mock_processor_class.assert_called_with(expected_input_dependencies, expected_input_metadata)

                mock_processor.process.assert_called()

    @patch('imap_l3_data_processor.SweProcessor')
    @patch('imap_l3_data_processor.argparse')
    def test_runs_swe_processor_when_instrument_argument_is_swe(self, mock_argparse, mock_processor_class):
        cases = [("20170630", datetime(2017, 6, 30)), (None, datetime(2016, 6, 30))]

        instrument_argument = "swe"
        data_level_argument = "l3"
        start_date_argument = "20160630"
        version_argument = "v092"
        descriptor_argument = "pitch-angle"
        dependencies_argument = (
            "[{'instrument':'swe', 'data_level':'l1000', 'descriptor':'science', 'version':'v112',"
            "'start_date':'20250101','end_date':'20250202'}]")

        mock_argument_parser = mock_argparse.ArgumentParser.return_value

        mock_argument_parser.parse_args.return_value.instrument = instrument_argument
        mock_argument_parser.parse_args.return_value.data_level = data_level_argument
        mock_argument_parser.parse_args.return_value.dependency = dependencies_argument
        mock_argument_parser.parse_args.return_value.start_date = start_date_argument
        mock_argument_parser.parse_args.return_value.version = version_argument
        mock_argument_parser.parse_args.return_value.descriptor = descriptor_argument

        mock_processor = mock_processor_class.return_value

        for input_end_date, expected_end_date in cases:
            with self.subTest(input_end_date):
                mock_argument_parser.parse_args.return_value.end_date = input_end_date

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

                expected_input_dependencies = [UpstreamDataDependency("swe",
                                                                      "l1000",
                                                                      datetime(2025, 1, 1),
                                                                      datetime(2025, 2, 2),
                                                                      "v112",
                                                                      "science")]

                expected_input_metadata = InputMetadata("swe", "l3", datetime(year=2016, month=6, day=30),
                                                        expected_end_date, "v092", "pitch-angle")

                mock_processor_class.assert_called_with(expected_input_dependencies, expected_input_metadata)

                mock_processor.process.assert_called()

    @patch('imap_l3_data_processor.HitProcessor')
    @patch('imap_l3_data_processor.argparse')
    def test_runs_hit_processor_when_instrument_argument_is_hit(self, mock_argparse, mock_processor_class):

        cases = [("20170630", datetime(2017, 6, 30), 'l3b'),
                 (None, datetime(2016, 6, 30), 'l3a')]

        instrument_argument = "hit"
        start_date_argument = "20160630"
        version_argument = "v092"
        descriptor_argument = "A descriptor"
        dependencies_argument = (
            "[{'instrument':'dependent_instrument', 'data_level':'l1000', 'descriptor':'science', 'version':'v112',"
            "'start_date':'20250101'}]")

        mock_argument_parser = mock_argparse.ArgumentParser.return_value
        mock_argument_parser.parse_args.return_value.instrument = instrument_argument
        mock_argument_parser.parse_args.return_value.dependency = dependencies_argument
        mock_argument_parser.parse_args.return_value.start_date = start_date_argument
        mock_argument_parser.parse_args.return_value.version = version_argument
        mock_argument_parser.parse_args.return_value.descriptor = descriptor_argument

        for input_end_date, expected_end_date, data_level in cases:
            with self.subTest(data_level):
                mock_argument_parser.parse_args.return_value.data_level = data_level
                mock_argument_parser.parse_args.return_value.end_date = input_end_date

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

                expected_input_dependencies = [UpstreamDataDependency("dependent_instrument",
                                                                      "l1000",
                                                                      datetime(2025, 1, 1),
                                                                      None,
                                                                      "v112",
                                                                      "science")]

                expected_input_metadata = InputMetadata("hit", data_level, datetime(year=2016, month=6, day=30),
                                                        expected_end_date, "v092", "A descriptor")

                mock_processor_class.assert_called_with(expected_input_dependencies,
                                                        expected_input_metadata)
                mock_processor_class.return_value.process.assert_called()

    @patch('imap_l3_data_processor.argparse')
    def test_throws_exception_for_unimplemented_instrument(self, mock_argparse):
        instrument_argument = "new_instrument"
        data_level_argument = "l3a"
        start_date_argument = "20160630"
        end_date_argument = None
        version_argument = "v092"
        dependencies_argument = "[{'instrument':'not_swapi', 'data_level':'l1000', 'descriptor':'science', 'version':'v112', 'start_date': '20250101'}]"

        mock_argument_parser = mock_argparse.ArgumentParser.return_value

        mock_argument_parser.parse_args.return_value.instrument = instrument_argument
        mock_argument_parser.parse_args.return_value.data_level = data_level_argument
        mock_argument_parser.parse_args.return_value.dependency = dependencies_argument
        mock_argument_parser.parse_args.return_value.start_date = start_date_argument
        mock_argument_parser.parse_args.return_value.end_date = end_date_argument
        mock_argument_parser.parse_args.return_value.version = version_argument

        with self.assertRaises(NotImplementedError) as exception_manager:
            imap_l3_processor()
        self.assertEqual(str(exception_manager.exception),
                         "Level l3a data processing has not yet been implemented for new_instrument")

    @patch('imap_l3_data_processor.argparse')
    def test_throws_exception_when_attempting_to_process_non_l3_data_levels(self, mock_argparse):
        instrument_argument = "swapi"
        data_level_argument = "l4"
        start_date_argument = "20160630"
        end_date_argument = "20160701"
        version_argument = "v092"
        dependencies_argument = "[{'instrument':'not_swapi', 'data_level':'l1000', 'descriptor':'science', 'version':'v112', 'start_date':'20250101'}]"

        mock_argument_parser = mock_argparse.ArgumentParser.return_value

        mock_argument_parser.parse_args.return_value.instrument = instrument_argument
        mock_argument_parser.parse_args.return_value.data_level = data_level_argument
        mock_argument_parser.parse_args.return_value.dependency = dependencies_argument
        mock_argument_parser.parse_args.return_value.start_date = start_date_argument
        mock_argument_parser.parse_args.return_value.end_date = end_date_argument
        mock_argument_parser.parse_args.return_value.version = version_argument

        with self.assertRaises(NotImplementedError) as exception_manager:
            imap_l3_processor()
        self.assertEqual(str(exception_manager.exception),
                         "Level l4 data processing has not yet been implemented for swapi")
