import uuid
from datetime import datetime, date
from pathlib import Path
from typing import List, Optional

import imap_data_access

from imap_processing.cdf.cdf_utils import write_cdf
from imap_processing.cdf.imap_attribute_manager import ImapAttributeManager
from imap_processing.constants import TEMP_CDF_FOLDER_PATH
from imap_processing.models import UpstreamDataDependency, DataProduct, InputMetadata


class Processor:
    def __init__(self, dependencies: List[UpstreamDataDependency], input_metadata: InputMetadata):
        self.input_metadata = input_metadata
        self.dependencies = dependencies
