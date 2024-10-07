from spacepy.pycdf import CDF

from imap_processing.models import InputMetadata
from imap_processing.swapi.l3a.utils import read_l2_swapi_data
from imap_processing.swapi.l3b.science.efficiency_calibration_table import EfficiencyCalibrationTable
from imap_processing.swapi.l3b.science.geometric_factor_calibration_table import GeometricFactorCalibrationTable
from imap_processing.swapi.l3b.swapi_l3b_dependencies import SwapiL3BDependencies
from imap_processing.swapi.swapi_processor import SwapiProcessor
from imap_processing.utils import save_data


def create_l3b_cdf(geometric_calibration_file, efficiency_calibration_file, cdf_file):
    geometric_calibration = GeometricFactorCalibrationTable.from_file(geometric_calibration_file)
    efficiency_calibration = EfficiencyCalibrationTable(efficiency_calibration_file)
    cdf_data = CDF(cdf_file)
    swapi_l3_dependencies = SwapiL3BDependencies(cdf_data, geometric_calibration, efficiency_calibration)
    swapi_data = read_l2_swapi_data(swapi_l3_dependencies.data)

    input_metadata = InputMetadata(
        instrument='swapi',
        data_level='l3b',
        start_date=None,
        end_date=None,
        version='v999')
    processor = SwapiProcessor(None, input_metadata)

    l3b_combined_vdf = processor.process_l3b(swapi_data, swapi_l3_dependencies)
    cdf_path = save_data(l3b_combined_vdf)
    return cdf_path


if __name__ == "__main__":
    path = create_l3b_cdf(
        "swapi/test_data/imap_swapi_l2_energy-gf-lut-not-cdf_20240923_v001.cdf",
        "swapi/test_data/imap_swapi_l2_efficiency-lut-text-not-cdf_20241020_v002.cdf",
        "swapi/test_data/imap_swapi_l2_sci_20100101_v001.cdf")
    print(path)
