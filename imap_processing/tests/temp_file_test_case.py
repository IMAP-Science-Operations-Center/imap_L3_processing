import os
import shutil
from pathlib import Path
from unittest import TestCase

import imap_processing


class TempFileTestCase(TestCase):
    def setUp(self) -> None:
        imap_processing_folder = Path(imap_processing.__file__).parent
        self.temp_directory = imap_processing_folder / "tests" / "temp_test_files"

        if os.path.exists(self.temp_directory):
            shutil.rmtree(self.temp_directory)
        os.mkdir(self.temp_directory)

    def tearDown(self) -> None:
        shutil.rmtree(self.temp_directory)
