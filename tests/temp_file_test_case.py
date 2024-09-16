import os
import shutil
from pathlib import Path
from unittest import TestCase

import tests


class TempFileTestCase(TestCase):
    def setUp(self) -> None:
        tests_folder = Path(tests.__file__).parent
        self.temp_directory = tests_folder / "temp_test_files"

        if os.path.exists(self.temp_directory):
            shutil.rmtree(self.temp_directory)
        os.mkdir(self.temp_directory)

    def tearDown(self) -> None:
        shutil.rmtree(self.temp_directory)
