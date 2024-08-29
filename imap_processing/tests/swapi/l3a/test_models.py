import os
import shutil
from pathlib import Path
from unittest import TestCase
from unittest.mock import patch, call

import numpy as np
from spacepy import pycdf

import imap_processing
from imap_processing.swapi.l3a.models import SwapiL3ProtonSolarWindData


class TestModels(TestCase):
    def setUp(self) -> None:
        imap_processing_folder = Path(imap_processing.__file__).parent

        print(os.getcwd())
        self.temp_directory = imap_processing_folder / "tests" / "test_files"
        if os.path.exists(self.temp_directory):
            shutil.rmtree(self.temp_directory)
        os.mkdir(self.temp_directory)

    def tearDown(self) -> None:
        shutil.rmtree(self.temp_directory)

    @patch('imap_processing.swapi.l3a.models.ImapAttributeManager')
    def test_proton_sw_write_cdf(self, mock_imap_attribute_manager_constructor):
        mock_imap_attribute_manager = mock_imap_attribute_manager_constructor.return_value
        mock_imap_attribute_manager.get_global_attributes.return_value = {
            'key1': 'value1',
            'key2': 'value2',
            'key3': 'value3',
        }

        epoch = np.arange(10)
        speeds = np.full(10, 450)
        data = SwapiL3ProtonSolarWindData(epoch, speeds)
        version = "v234"
        file_path = f"{self.temp_directory}/proton_cdf_test.cdf"

        data.write_cdf(file_path, version)

        mock_imap_attribute_manager.add_global_attribute.assert_has_calls(
            [call('Logical_file_id', file_path),
             call('Data_version', version)])

        result_cdf = pycdf.CDF(file_path)
        self.assertEqual('value1',result_cdf.attrs['key1'][...][0])
        self.assertEqual('value2',result_cdf.attrs['key2'][...][0])
        self.assertEqual('value3',result_cdf.attrs['key3'][...][0])