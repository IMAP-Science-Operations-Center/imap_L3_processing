import os
import shutil
from datetime import datetime
from pathlib import Path
from unittest import TestCase
from unittest.mock import sentinel, patch, call, Mock

import numpy as np

from imap_processing.constants import TEMP_CDF_FOLDER_PATH
from imap_processing.hit.l3.hit_processor import HitProcessor
from imap_processing.hit.l3.models import HitL2Data
from imap_processing.models import UpstreamDataDependency, InputMetadata, MagL1dData
from imap_processing.processor import Processor
from tests.test_helpers import NumpyArrayMatcher


class TestHitProcessor(TestCase):
    def test_is_a_processor(self):
        self.assertIsInstance(
            HitProcessor([], Mock()),
            Processor
        )
