import os
from datetime import datetime, date
from pathlib import Path
from typing import Optional, Union
from urllib.error import HTTPError
from urllib.request import urlretrieve

import imap_data_access
from spacepy.pycdf import CDF
from spiceypy import spiceypy

from imap_l3_processing.cdf.cdf_utils import write_cdf, read_variable
from imap_l3_processing.cdf.imap_attribute_manager import ImapAttributeManager
from imap_l3_processing.constants import TEMP_CDF_FOLDER_PATH
from imap_l3_processing.models import UpstreamDataDependency, DataProduct, MagL1dData


def load_spice_kernels():
    kernel_dir = "/mnt/spice"
    kernel_paths = [os.path.join(kernel_dir, name) for name in os.listdir(kernel_dir)]
    spiceypy.furnsh(kernel_paths)


def save_data(data: DataProduct, delete_if_present: bool = False, folder_path: str = TEMP_CDF_FOLDER_PATH) -> str:
    formatted_start_date = format_time(data.input_metadata.start_date)
    logical_source = data.input_metadata.logical_source
    logical_file_id = f'{logical_source}_{formatted_start_date}_{data.input_metadata.version}'
    Path(folder_path).mkdir(exist_ok=True)
    file_path = f'{folder_path}/{logical_file_id}.cdf'

    if delete_if_present:
        Path(file_path).unlink(missing_ok=True)

    attribute_manager = ImapAttributeManager()
    attribute_manager.add_global_attribute("Data_version", data.input_metadata.version)
    attribute_manager.add_instrument_attrs(data.input_metadata.instrument, data.input_metadata.data_level,
                                           data.input_metadata.descriptor)
    attribute_manager.add_global_attribute("Generation_date", date.today().strftime("%Y%m%d"))
    attribute_manager.add_global_attribute("Logical_source", logical_source)
    attribute_manager.add_global_attribute("Logical_file_id", logical_file_id)
    write_cdf(file_path, data, attribute_manager)
    return file_path


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
                                                version=dependency.version
                                                )]
    if len(files_to_download) != 1:
        raise ValueError(f"{files_to_download}. Expected one file to download, found {len(files_to_download)}.")

    return imap_data_access.download(files_to_download[0])


def download_dependency_from_path(path_str: str) -> Path:
    raise NotImplemented


def download_external_dependency(dependency_url: str, filename: str) -> Path | None:
    try:
        saved_path, _ = urlretrieve(dependency_url, filename)
        return Path(saved_path)
    except HTTPError:
        return None


def read_l1d_mag_data(cdf_path: Union[str, Path]) -> MagL1dData:
    with CDF(str(cdf_path)) as cdf:
        return MagL1dData(
            epoch=cdf['epoch'][...],
            mag_data=read_variable(cdf["vectors"])[:, :3])
