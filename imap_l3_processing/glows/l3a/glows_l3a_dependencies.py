from dataclasses import dataclass
from pathlib import Path

from imap_data_access import download
from imap_data_access.processing_input import ProcessingInputCollection, ScienceInput
from spacepy.pycdf import CDF

from imap_l3_processing.glows.descriptors import GLOWS_L2_DESCRIPTOR, GLOWS_TIME_DEPENDENT_BACKGROUND_DESCRIPTOR, \
    GLOWS_CALIBRATION_DATA_DESCRIPTOR, \
    GLOWS_EXTRA_HELIOSPHERIC_BACKGROUND_DESCRIPTOR, GLOWS_PIPELINE_SETTINGS_DESCRIPTOR
from imap_l3_processing.glows.l3a.models import GlowsL2Data
from imap_l3_processing.glows.l3a.utils import read_l2_glows_data


@dataclass
class GlowsL3ADependencies:
    data: GlowsL2Data
    ancillary_files: dict[str, Path]

    @classmethod
    def fetch_dependencies(cls, dependencies: ProcessingInputCollection):
        science_inputs: list[ScienceInput] = dependencies.get_science_inputs()
        desired_input = next(science_input for science_input in science_inputs
                             if science_input.descriptor == GLOWS_L2_DESCRIPTOR and science_input.source == 'glows')

        l2_cdf_file_path: Path = desired_input.imap_file_paths[0].construct_path()

        l2_cdf = download(l2_cdf_file_path)

        cdf = CDF(str(l2_cdf))
        l2_glows_data = read_l2_glows_data(cdf)
        calibration_dependency = dependencies.get_file_paths(source='glows',
                                                             descriptor=GLOWS_CALIBRATION_DATA_DESCRIPTOR)
        time_dependent_background_dependency = dependencies.get_file_paths(source='glows',
                                                                           descriptor=GLOWS_TIME_DEPENDENT_BACKGROUND_DESCRIPTOR)
        extra_heliospheric_background_dependency = dependencies.get_file_paths(source='glows',
                                                                               descriptor=GLOWS_EXTRA_HELIOSPHERIC_BACKGROUND_DESCRIPTOR)
        settings_dependency = dependencies.get_file_paths(source='glows', descriptor=GLOWS_PIPELINE_SETTINGS_DESCRIPTOR)

        calibration_path = download(calibration_dependency[0])
        extra_heliospheric_background = download(extra_heliospheric_background_dependency[0])
        time_dependent_background_path = download(time_dependent_background_dependency[0])
        settings_path = download(settings_dependency[0])

        ancillary_files = {
            "calibration_data": calibration_path,
            "settings": settings_path,
            "time_dependent_bckgrd": time_dependent_background_path,
            "extra_heliospheric_bckgrd": extra_heliospheric_background,
        }
        return cls(l2_glows_data, ancillary_files)
