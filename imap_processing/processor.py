import uuid
from datetime import datetime, date
from pathlib import Path
from typing import List, Optional

import imap_data_access

from imap_processing.cdf.cdf_utils import write_cdf
from imap_processing.cdf.imap_attribute_manager import ImapAttributeManager
from imap_processing.constants import TEMP_CDF_FOLDER_PATH
from imap_processing.models import UpstreamDataDependency, DataProduct


class Processor:
    def __init__(self, dependencies: List[UpstreamDataDependency], instrument: str, level: str, start_date: datetime,
                 end_date: datetime,
                 version: str):
        self.instrument = instrument
        self.level = level
        self.version = version
        self.end_date = end_date
        self.start_date = start_date
        self.dependencies = dependencies

    def upload_data(self, data: DataProduct, descriptor: str):
        formatted_start_date = format_time(self.start_date)
        logical_file_id = f'imap_{self.instrument}_{self.level}_{descriptor}-fake-menlo-{uuid.uuid4()}_{formatted_start_date}_{self.version}'
        file_path = f'{TEMP_CDF_FOLDER_PATH}/{logical_file_id}.cdf'

        attribute_manager = ImapAttributeManager()
        attribute_manager.add_global_attribute("Data_version", self.version)
        attribute_manager.add_instrument_attrs(self.instrument, self.level)
        attribute_manager.add_global_attribute("Generation_date", date.today().strftime("%Y%m%d"))
        attribute_manager.add_global_attribute("Logical_source", f'imap_{self.instrument}_{self.level}_{descriptor}')
        attribute_manager.add_global_attribute("Logical_file_id", logical_file_id)
        write_cdf(file_path, data, attribute_manager)
        imap_data_access.upload(file_path)


def format_time(t: Optional[datetime]) -> Optional[str]:
    if t is not None:
        return t.strftime("%Y%m%d")
    return None


def download_dependency(dependency: UpstreamDataDependency) -> Path:
    print(dependency)
    files_to_download = [result['file_path'] for result in
                         imap_data_access.query(instrument=dependency.instrument,
                                                data_level=dependency.data_level,
                                                descriptor=dependency.descriptor,
                                                start_date=format_time(dependency.start_date),
                                                end_date=format_time(dependency.end_date),
                                                version='latest'
                                                )]
    if len(files_to_download) != 1:
        raise ValueError(f"{files_to_download}. Expected only one file to download.")

    return imap_data_access.download(files_to_download[0])
