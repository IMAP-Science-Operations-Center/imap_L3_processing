import re
from datetime import datetime, date
from pathlib import Path
from typing import Optional, Union, TypeVar
from urllib.error import URLError
from urllib.request import urlretrieve

import imap_data_access
import spiceypy
from imap_data_access import ScienceFilePath
from spacepy.pycdf import CDF

import imap_l3_processing
from imap_l3_processing.cdf.cdf_utils import write_cdf, read_numeric_variable
from imap_l3_processing.cdf.imap_attribute_manager import ImapAttributeManager
from imap_l3_processing.maps.map_models import GlowsL3eRectangularMapInputData, InputRectangularPointingSet
from imap_l3_processing.models import UpstreamDataDependency, DataProduct, MagL1dData
from imap_l3_processing.ultra.l3.models import UltraL1CPSet, UltraGlowsL3eData
from imap_l3_processing.version import VERSION


def save_data(data: DataProduct, delete_if_present: bool = False, folder_path: Path = None,
              cr_number=None) -> Path:
    assert data.input_metadata.repointing is None or cr_number is None, "You cannot call save_data with both a repointing in the metadata while passing in a CR number"
    formatted_start_date = data.input_metadata.start_date.strftime("%Y%m%d")
    science_file_path = ScienceFilePath.generate_from_inputs(
        instrument=data.input_metadata.instrument,
        data_level=data.input_metadata.data_level,
        descriptor=data.input_metadata.descriptor,
        start_time=formatted_start_date,
        repointing=data.input_metadata.repointing,
        cr=cr_number,
        version=data.input_metadata.version,
    )

    file_path = science_file_path.construct_path()
    if folder_path is not None:
        file_path = folder_path / file_path.name

    logical_source = f"imap_{data.input_metadata.instrument}_{data.input_metadata.data_level}_{data.input_metadata.descriptor}"
    logical_file_id = file_path.stem

    file_path.parent.mkdir(parents=True, exist_ok=True)
    if delete_if_present:
        file_path.unlink(missing_ok=True)

    attribute_manager = ImapAttributeManager()
    attribute_manager.add_global_attribute("Data_version", data.input_metadata.version.replace('v', ''))
    attribute_manager.add_instrument_attrs(data.input_metadata.instrument, data.input_metadata.data_level,
                                           data.input_metadata.descriptor)
    attribute_manager.add_global_attribute("Generation_date", date.today().strftime("%Y%m%d"))
    attribute_manager.add_global_attribute("Logical_source", logical_source)
    attribute_manager.add_global_attribute("Logical_file_id", logical_file_id)
    attribute_manager.add_global_attribute("ground_software_version", VERSION)

    map_instruments = ["hi", "lo", "ultra"]
    if data.input_metadata.instrument in map_instruments and attribute_manager.try_load_global_metadata(
            logical_source) is None:
        level = data.input_metadata.data_level.replace('l', '')
        data_type_string = f"Level-{level}"
        if "spx" in data.input_metadata.descriptor:
            data_type_string += f" Spectral Fit Index Map"
        elif "ena" in data.input_metadata.descriptor and "-sp-" in data.input_metadata.descriptor:
            data_type_string += f" Survival Corrected"
        elif "ena" in data.input_metadata.descriptor:
            data_type_string += f" ENA Intensity"

        logical_source_global_attrs = {
            "Data_level": level,
            "Data_type": f"L{level}_{data.input_metadata.descriptor}>{data_type_string}",
            "Logical_source_description": f"IMAP-{data.input_metadata.instrument} {data_type_string}",
        }
        attribute_manager.add_global_attribute(logical_source, logical_source_global_attrs)

    if data.parent_file_names:
        attribute_manager.add_global_attribute("Parents", data.parent_file_names)
    file_path_str = str(file_path)
    write_cdf(file_path_str, data, attribute_manager)
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


def download_dependency_with_repointing(dependency: UpstreamDataDependency) -> (Path, int):
    files_with_repointing_to_download = [(result['file_path'], result['repointing']) for result in
                                         imap_data_access.query(instrument=dependency.instrument,
                                                                data_level=dependency.data_level,
                                                                descriptor=dependency.descriptor,
                                                                start_date=format_time(dependency.start_date),
                                                                end_date=format_time(dependency.end_date),
                                                                version=dependency.version
                                                                )]
    if len(files_with_repointing_to_download) != 1:
        raise ValueError(
            f"{[file[0] for file in files_with_repointing_to_download]}. Expected one file to download, found {len(files_with_repointing_to_download)}.")
    repointing_number = files_with_repointing_to_download[0][1]
    return imap_data_access.download(files_with_repointing_to_download[0][0]), repointing_number


def download_dependency_from_path(path_str: str) -> Path:
    return imap_data_access.download(path_str)


def download_external_dependency(dependency_url: str, filename: str) -> Optional[Path]:
    try:
        saved_path, _ = urlretrieve(dependency_url, filename)
        return Path(saved_path)
    except URLError:
        return None


def read_l1d_mag_data(cdf_path: Union[str, Path]) -> MagL1dData:
    with CDF(str(cdf_path)) as cdf:
        return MagL1dData(
            epoch=cdf['epoch'][...],
            mag_data=read_numeric_variable(cdf["vectors"])[:, :3])


def find_glows_l3e_dependencies(l1c_filenames: list[str], instrument: str) -> list[str]:
    dates = [datetime.strptime(ScienceFilePath(l1c_filename).start_date, "%Y%m%d") for l1c_filename in l1c_filenames]

    start_date = min(dates).strftime("%Y%m%d")
    end_date = max(dates).strftime("%Y%m%d")

    match instrument:
        case 'ultra':
            descriptor = "survival-probability-ul"
        case 'hi':
            initial_descriptor = l1c_filenames[0].split("_")[3]
            sensor = re.search(r"^(\d+)", initial_descriptor)
            descriptor = f"survival-probability-{instrument}-{sensor.group(0)}"
        case _:
            descriptor = f"survival-probability-{instrument}"

    survival_probabilities = [result["file_path"] for result in imap_data_access.query(instrument="glows",
                                                                                       data_level="l3e",
                                                                                       descriptor=descriptor,
                                                                                       start_date=start_date,
                                                                                       end_date=end_date,
                                                                                       version="latest")]

    return survival_probabilities


L1CPointingSet = TypeVar("L1CPointingSet", bound=Union[InputRectangularPointingSet, UltraL1CPSet])
GlowsL3eData = TypeVar("GlowsL3eData", bound=Union[GlowsL3eRectangularMapInputData, UltraGlowsL3eData])


def combine_glows_l3e_with_l1c_pointing(glows_l3e_data: list[GlowsL3eData], l1c_data: list[L1CPointingSet]) -> list[
    tuple[L1CPointingSet, Optional[GlowsL3eData]]]:
    l1c_by_epoch = {l1c.epoch: l1c for l1c in l1c_data}
    glows_by_epoch = {l3e.epoch: l3e for l3e in glows_l3e_data}

    return [(l1c_by_epoch[epoch], glows_by_epoch.get(epoch, None)) for epoch in l1c_by_epoch.keys()]


def furnish_local_spice():
    kernels = Path(imap_l3_processing.__file__).parent.parent.joinpath("spice_kernels")

    total_kernels = spiceypy.ktotal('ALL')
    current_kernels = []
    for i in range(0, total_kernels):
        current_kernels.append(Path(spiceypy.kdata(i, 'ALL')[0]).name)

    for file in kernels.iterdir():
        if file.name not in current_kernels:
            spiceypy.furnsh(str(file))
