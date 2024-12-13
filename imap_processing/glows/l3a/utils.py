from datetime import datetime
from pathlib import Path

import numpy as np
from spacepy.pycdf import CDF

from imap_processing.glows.l3a.models import GlowsL2Data, GlowsL2LightCurve, GlowsLatLon, GlowsL3LightCurve, XYZ, \
    GlowsL2Header
from imap_processing.models import UpstreamDataDependency


def read_l2_glows_data(cdf: CDF) -> GlowsL2Data:
    assert 1 == cdf['photon_flux'].shape[0], "Level 2 file should have only one histogram"

    flags = np.packbits(cdf['histogram_flag_array'][0], bitorder='little', axis=0)[0]
    light_curve = GlowsL2LightCurve(photon_flux=cdf['photon_flux'][0],
                                    spin_angle=cdf['spin_angle'][0],
                                    histogram_flag_array=flags,
                                    exposure_times=cdf['exposure_times'][0],
                                    flux_uncertainties=cdf['flux_uncertainties'][0],
                                    raw_histogram=cdf['raw_histogram'][0],
                                    ecliptic_lon=cdf['ecliptic_lon'][0],
                                    ecliptic_lat=cdf['ecliptic_lat'][0], )
    spin_axis_average = GlowsLatLon(lat=cdf['spin_axis_orientation_average_lat'][0],
                                    lon=cdf['spin_axis_orientation_average_lon'][0])
    spin_axis_std_dev = GlowsLatLon(lat=cdf['spin_axis_orientation_std_dev_lat'][0],
                                    lon=cdf['spin_axis_orientation_std_dev_lon'][0])
    return GlowsL2Data(identifier=cdf['identifier'][0],
                       start_time=str(cdf['start_time'][0]),
                       end_time=str(cdf['end_time'][0]),
                       daily_lightcurve=light_curve,
                       number_of_bins=cdf['number_of_bins'][...],
                       spin_axis_orientation_average=spin_axis_average,
                       spin_axis_orientation_std_dev=spin_axis_std_dev,
                       filter_temperature_average=cdf['filter_temperature_average'][0],
                       filter_temperature_std_dev=cdf['filter_temperature_std_dev'][0],
                       hv_voltage_average=cdf['hv_voltage_average'][0],
                       hv_voltage_std_dev=cdf['hv_voltage_std_dev'][0],
                       spin_period_average=cdf['spin_period_average'][0],
                       spin_period_std_dev=cdf['spin_period_std_dev'][0],
                       spin_period_ground_average=cdf['spin_period_ground_average'][0],
                       spin_period_ground_std_dev=cdf['spin_period_ground_std_dev'][0],
                       pulse_length_average=cdf['pulse_length_average'][0],
                       pulse_length_std_dev=cdf['pulse_length_std_dev'][0],
                       position_angle_offset_average=cdf['position_angle_offset_average'][0],
                       position_angle_offset_std_dev=cdf['position_angle_offset_std_dev'][0],
                       spacecraft_location_average=_read_xyz(cdf, 'spacecraft_location_average'),
                       spacecraft_location_std_dev=_read_xyz(cdf, 'spacecraft_location_std_dev'),
                       spacecraft_velocity_average=_read_xyz(cdf, 'spacecraft_velocity_average'),
                       spacecraft_velocity_std_dev=_read_xyz(cdf, 'spacecraft_velocity_std_dev'),
                       header=GlowsL2Header(
                           flight_software_version=cdf.attrs['flight_software_version'][...][0],
                           pkts_file_name=cdf.attrs['pkts_file_name'][...][0],
                           ancillary_data_files=list(cdf.attrs['ancillary_data_files']),
                       ),
                       l2_file_name=Path(cdf.pathname.decode('utf-8')).name
                       )


def _read_xyz(cdf: CDF, variable_name: str) -> XYZ:
    return {k: cdf[f'{variable_name}_{k}'][0] for k in "xyz"}


def create_glows_l3a_from_dictionary(data: dict, input_metadata: UpstreamDataDependency) -> GlowsL3LightCurve:
    start_time = datetime.fromisoformat(data["start_time"])
    end_time = datetime.fromisoformat(data["end_time"])
    total_time = end_time - start_time
    epoch = start_time + total_time / 2
    return GlowsL3LightCurve(
        input_metadata=input_metadata,
        epoch=np.array([epoch]),
        epoch_delta=np.array([total_time.total_seconds() / 2 * 1e9]),
        photon_flux=np.array(data["daily_lightcurve"]["photon_flux"]).reshape(1, -1),
        photon_flux_uncertainty=np.array(data["daily_lightcurve"]["flux_uncertainties"]).reshape(1, -1),
        raw_histogram=np.array(data["daily_lightcurve"]["raw_histogram"]).reshape(1, -1),
        exposure_times=np.array(data["daily_lightcurve"]["exposure_times"]).reshape(1, -1),
        spin_angle=np.array(data["daily_lightcurve"]["spin_angle"]).reshape(1, -1),
        latitude=np.array(data["daily_lightcurve"]["ecliptic_lat"]).reshape(1, -1),
        longitude=np.array(data["daily_lightcurve"]["ecliptic_lon"]).reshape(1, -1),
        extra_heliospheric_background=np.array(data["daily_lightcurve"]["extra_heliospheric_bckgrd"]).reshape(1, -1),
        time_dependent_background=np.array(data["daily_lightcurve"]["time_dependent_bckgrd"]).reshape(1, -1),
        filter_temperature_average=np.array([data["filter_temperature_average"]]),
        filter_temperature_std_dev=np.array([data["filter_temperature_std_dev"]]),
        hv_voltage_average=np.array([data["hv_voltage_average"]]),
        hv_voltage_std_dev=np.array([data["hv_voltage_std_dev"]]),
        spin_period_average=np.array([data["spin_period_average"]]),
        spin_period_std_dev=np.array([data["spin_period_std_dev"]]),
        spin_period_ground_average=np.array([data["spin_period_ground_average"]]),
        spin_period_ground_std_dev=np.array([data["spin_period_ground_std_dev"]]),
        pulse_length_average=np.array([data["pulse_length_average"]]),
        pulse_length_std_dev=np.array([data["pulse_length_std_dev"]]),
        position_angle_offset_average=np.array([data["position_angle_offset_average"]]),
        position_angle_offset_std_dev=np.array([data["position_angle_offset_std_dev"]]),
        spin_axis_orientation_average=get_lon_lat(data["spin_axis_orientation_average"]),
        spin_axis_orientation_std_dev=get_lon_lat(data["spin_axis_orientation_std_dev"]),
        spacecraft_location_average=get_xyz(data["spacecraft_location_average"]),
        spacecraft_location_std_dev=get_xyz(data["spacecraft_location_std_dev"]),
        spacecraft_velocity_average=get_xyz(data["spacecraft_velocity_average"]),
        spacecraft_velocity_std_dev=get_xyz(data["spacecraft_velocity_std_dev"]),
    )


def get_lon_lat(data: dict) -> np.ndarray:
    return np.array([[data["lon"], data["lat"]]])


def get_xyz(data: dict) -> np.ndarray:
    return np.array([[data["x"], data["y"], data["z"]]])
