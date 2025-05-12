import shutil
from datetime import datetime
from pathlib import Path

import numpy as np
from spacepy.pycdf import CDF

from imap_l3_processing.hi.hi_processor import HiProcessor
from imap_l3_processing.hi.l3.hi_l3_survival_dependencies import HiL3SurvivalDependencies
from imap_l3_processing.map_models import RectangularIntensityDataProduct
from imap_l3_processing.models import InputMetadata
from imap_l3_processing.utils import save_data
from scripts.hi.create_cdf_from_instrument_team_data import create_l2_map_from_instrument_team
from scripts.hi.generate_fake_l1c_cdfs import generate_fake_l1c
from tests.test_helpers import get_test_instrument_team_data_path, get_test_data_path, get_run_local_data_path


def create_hi_full_spin_deps(
        sensor="90",
        output_dir: Path = get_run_local_data_path("hi/full_spin_deps"),
        start_date: datetime = datetime(year=2025, month=4, day=15, hour=12),
        l2_map_dir: Path = None,
        glows_dir: Path = None):
    if l2_map_dir is None:
        l2_map_dir = get_test_instrument_team_data_path(f"hi/hi{sensor}-6months")
    if glows_dir is None:
        glows_dir = get_test_data_path(f"hi/fake_l3e_survival_probabilities/90")
    l1c_folder = output_dir / "l1c"
    l1c_folder.mkdir(parents=True, exist_ok=True)

    l1c_files = [Path(s) for s in generate_fake_l1c(start_date, 185, output_dir=l1c_folder)]
    original_full_spin_map_path = create_l2_map_from_instrument_team(l2_map_dir, output_dir=output_dir)
    parents = [f.name for f in l1c_files]
    glows_files = list(glows_dir.iterdir())

    with CDF(str(original_full_spin_map_path)) as cdf:
        original_intensity = cdf["ena_intensity"][...]

    ramified_map_deps = HiL3SurvivalDependencies.from_file_paths(
        original_full_spin_map_path, l1c_files, glows_files,
        f"h{sensor}-ena-h-sf-nsp-ram-hae-4deg-6mo")

    antiramified_map_deps = HiL3SurvivalDependencies.from_file_paths(
        original_full_spin_map_path, l1c_files,
        glows_files,
        f"h{sensor}-ena-h-sf-nsp-anti-hae-4deg-6mo")

    input_metadata = InputMetadata("hi", "l3", None, None, "v001", f"h90-ena-h-sf-sp-full-hae-4deg-6mo")
    processor = HiProcessor(None,
                            input_metadata)
    ram_data = processor.process_survival_probabilities(ramified_map_deps)
    ram_exposure_is_zero = (np.isnan(ram_data.intensity_map_data.ena_intensity)
                            | (ram_data.intensity_map_data.ena_intensity == 0))

    ram_cdf_path = save_data(
        RectangularIntensityDataProduct(input_metadata, ram_data),
        delete_if_present=True, folder_path=output_dir)
    ram_logical_source = f"imap_hi_l2_h{sensor}-ena-h-sf-nsp-ram-hae-4deg-6mo_20250415_v001"
    with CDF(ram_cdf_path, readonly=False) as ram:
        ram["exposure_factor"] = np.where(ram_exposure_is_zero, 0, ram_data.intensity_map_data.exposure_factor)
        ram["ena_intensity"] = np.where(ram_exposure_is_zero, ram["ena_intensity"].attrs["FILLVAL"],
                                        original_intensity)
        ram.attrs["Parents"] = parents
        ram.attrs["Logical_source"] = ram_logical_source
        ram["ena_intensity"].attrs["CATDESC"] = ram_logical_source

    shutil.move(ram_cdf_path, output_dir / f"{ram_logical_source}.cdf")

    antiram_data = processor.process_survival_probabilities(antiramified_map_deps)
    antiram_exposure_is_zero = (np.isnan(antiram_data.intensity_map_data.ena_intensity)
                                | (antiram_data.intensity_map_data.ena_intensity == 0))

    antiram_cdf_path = save_data(
        RectangularIntensityDataProduct(input_metadata, antiram_data),
        delete_if_present=True, folder_path=output_dir)
    antiram_logical_source = f"imap_hi_l2_h{sensor}-ena-h-sf-nsp-anti-hae-4deg-6mo_20250415_v001"

    with CDF(antiram_cdf_path, readonly=False) as antiram:
        antiram["exposure_factor"] = np.where(antiram_exposure_is_zero, 0,
                                              antiram_data.intensity_map_data.exposure_factor)
        antiram["ena_intensity"] = np.where(antiram_exposure_is_zero, antiram["ena_intensity"].attrs["FILLVAL"],
                                            original_intensity)
        antiram.attrs["Parents"] = parents
        antiram.attrs["Logical_source"] = antiram_logical_source
        antiram["ena_intensity"].attrs["CATDESC"] = antiram_logical_source
    shutil.move(antiram_cdf_path, output_dir / f"{antiram_logical_source}.cdf")


if __name__ == '__main__':
    create_hi_full_spin_deps()
