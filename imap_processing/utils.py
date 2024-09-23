import os
import uuid
from datetime import datetime, date
from pathlib import Path
from typing import Optional

import imap_data_access
from spacepy.pycdf import CDF
from spiceypy import spiceypy

from imap_processing.cdf.cdf_utils import write_cdf
from imap_processing.cdf.imap_attribute_manager import ImapAttributeManager
from imap_processing.constants import TEMP_CDF_FOLDER_PATH
from imap_processing.models import UpstreamDataDependency, DataProduct, MagL2Data


def load_spice_kernels():
    kernel_dir = "/mnt/spice"
    kernel_paths = [os.path.join(kernel_dir, name) for name in os.listdir(kernel_dir)]
    spiceypy.furnsh(kernel_paths)


def upload_data(data: DataProduct):
    formatted_start_date = format_time(data.input_metadata.start_date)
    logical_file_id = f'imap_{data.input_metadata.instrument}_{data.input_metadata.data_level}_{data.input_metadata.descriptor}-fake-menlo-{uuid.uuid4()}_{formatted_start_date}_{data.input_metadata.version}'
    file_path = f'{TEMP_CDF_FOLDER_PATH}/{logical_file_id}.cdf'

    attribute_manager = ImapAttributeManager()
    attribute_manager.add_global_attribute("Data_version", data.input_metadata.version)
    attribute_manager.add_instrument_attrs(data.input_metadata.instrument, data.input_metadata.data_level)
    attribute_manager.add_global_attribute("Generation_date", date.today().strftime("%Y%m%d"))
    attribute_manager.add_global_attribute("Logical_source",
                                           f'imap_{data.input_metadata.instrument}_{data.input_metadata.data_level}_{data.input_metadata.descriptor}')
    attribute_manager.add_global_attribute("Logical_file_id", logical_file_id)
    write_cdf(file_path, data, attribute_manager)
    imap_data_access.upload(file_path)


def format_time(t: Optional[datetime]) -> Optional[str]:
    if t is not None:
        return t.strftime("%Y%m%d")
    return None


def download_dependency(dependency: UpstreamDataDependency) -> Path:
    files_to_download = [result['file_path'] for result in
                         imap_data_access.query(instrument=dependency.instrument,
                                                data_level=dependency.data_level,
                                                descriptor=dependency.descriptor,
                                                start_date=format_time(dependency.start_date),
                                                end_date=format_time(dependency.end_date),
                                                version='latest'
                                                )]
    if len(files_to_download) != 1:
        raise ValueError(f"{files_to_download}. Expected one file to download, found {len(files_to_download)}.")

    return imap_data_access.download(files_to_download[0])


def read_l2_mag_data(cdf: CDF) -> MagL2Data:
    return MagL2Data(
        epoch=cdf.raw_var("epoch_mag_SC_1min")[...],
        mag_data=cdf["psp_fld_l2_mag_SC_1min"][...])
