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
    mock_arguments = Mock()
    mock_arguments.instrument = "ultra"
    mock_arguments.data_level = "l3"
    mock_arguments.start_date = "20251018"
    mock_arguments.end_date = None
    mock_arguments.repointing = None
    mock_arguments.version = "v002"
    mock_arguments.dependency = "[]"
    mock_arguments.upload_to_sdc = False
    mock_parse_cli_arguments.return_value = mock_arguments
    logging.basicConfig(force=True, level=logging.INFO,
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    with mock_imap_data_access(output_dir, list(input_dir.rglob('*.*'))):
        mock_arguments.descriptor = "u45-maps"
        imap_l3_data_processor.imap_l3_processor()
        spiceypy.kclear()
        mock_arguments.descriptor = "u90-maps"
        imap_l3_data_processor.imap_l3_processor()
        spiceypy.kclear()
        mock_arguments.descriptor = "ulc-nsp-maps"
        imap_l3_data_processor.imap_l3_processor()
        spiceypy.kclear()
        mock_arguments.descriptor = "ulc-sp-maps"
        imap_l3_data_processor.imap_l3_processor()
        spiceypy.kclear()




if __name__ == "__main__":
    output_dir = Path(f'{datetime.now().strftime('%Y%m%d')}_ultra_validation_output')
    try:
        input_dir = Path(sys.argv[1])
    except:
        input_dir = Path(fr'C:\Users\Harrison\AppData\Local\cava\cava\data')
        print(f"Could not open input file, using default input path of {input_dir} instead")

    run_all_maps(output_dir=output_dir, input_dir=input_dir)