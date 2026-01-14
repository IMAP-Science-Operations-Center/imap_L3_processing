import logging
import sys
from datetime import datetime
from pathlib import Path
from unittest.mock import Mock, patch

import spiceypy

import imap_l3_data_processor
from tests.integration.integration_test_helpers import mock_imap_data_access


@patch('imap_l3_data_processor._parse_cli_arguments')
def run_all_maps(mock_parse_cli_arguments, *, output_dir, input_dir,):
    intermediate_output = output_dir.parent / f"{output_dir.name}_intermediate"
    mock_arguments = Mock()
    mock_arguments.instrument = "hi"
    mock_arguments.data_level = "l3"
    mock_arguments.start_date = "20250415"
    mock_arguments.end_date = None
    mock_arguments.repointing = None
    mock_arguments.version = "v001"
    mock_arguments.dependency = "[]"
    mock_arguments.upload_to_sdc = False
    mock_parse_cli_arguments.return_value = mock_arguments
    with mock_imap_data_access(Path(intermediate_output), list(input_dir.rglob('*.*'))):
        mock_arguments.descriptor = "sp-maps"
        logging.basicConfig(force=True, level=logging.INFO,
                            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


        imap_l3_data_processor.imap_l3_processor()
    spiceypy.kclear()
    with mock_imap_data_access(output_dir, list(Path(intermediate_output).rglob('*.*'))):
        mock_arguments.descriptor = "hic-maps"
        imap_l3_data_processor.imap_l3_processor()
    spiceypy.kclear()


if __name__ == "__main__":
    output_dir = Path(f'{datetime.now().strftime('%Y%m%d')}_hi_validation_output')
    try:
        input_dir = Path(sys.argv[1])
    except:
        input_dir = Path(f'hi_validation_input')
        print(f"Could not open input file, using default input path of {input_dir} instead")

    run_all_maps(output_dir=output_dir, input_dir=input_dir)
