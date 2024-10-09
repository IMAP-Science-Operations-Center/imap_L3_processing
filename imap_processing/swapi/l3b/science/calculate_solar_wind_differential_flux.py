from numpy import ndarray

from imap_processing.constants import METERS_PER_KILOMETER, CENTIMETERS_PER_METER
from imap_processing.swapi.l3b.science.geometric_factor_calibration_table import GeometricFactorCalibrationTable


def calculate_combined_solar_wind_differential_flux(energies: ndarray, average_count_rates: ndarray,
                                                    efficiency: float,
                                                    geometric_factor_table: GeometricFactorCalibrationTable):
    geometric_factor = geometric_factor_table.lookup_geometric_factor(energies)

    denominator = energies * geometric_factor * efficiency
    result_per_square_km = average_count_rates / denominator
    result_per_square_cm = result_per_square_km / (METERS_PER_KILOMETER * CENTIMETERS_PER_METER) ** 2
    return result_per_square_cm
