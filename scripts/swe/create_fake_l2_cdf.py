import traceback
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pyhdf
import pyhdf.SD
from pyhdf.VS import *
from pyhdf.HDF import *
from spacepy.pycdf import CDF


def create_fake_l2_cdf(l1_hdf_path: str, l2_hdf_path: str, l2_swe_cdf_file_path: str, mag_file_path: str,
                       swapi_file_path: str):
    l1_hdf = pyhdf.SD.SD(l1_hdf_path)

    l2_hdf_path = HDF(l2_hdf_path)
    vs = l2_hdf_path.vstart()
    vd = vs.attach("swepam_e")

    b_rtn_index = vd.field("b_rtn")._index
    b_rtn_data = np.array([x[b_rtn_index] for x in vd[:]])
    mag_r, mag_t, mag_n = b_rtn_data.T
    mag_despun_vectors = np.stack((mag_n, mag_t, -mag_r), axis=1)

    sw_velocity_index = vd.field(
        "v_rtn_i")._index  # this is wrong, eventually use swepam I file for proton sw velocity instead
    sw_velocity_data = np.array([x[sw_velocity_index] for x in vd[:]])
    sw_speed = np.linalg.norm(sw_velocity_data, axis=1)
    sw_r, sw_t, sw_n = sw_velocity_data.T
    sw_x, sw_y, sw_z = sw_n, sw_t, -sw_r
    deflection = np.rad2deg(np.arctan2(np.sqrt(sw_x ** 2 + sw_y ** 2), -sw_z))
    clock = np.rad2deg(np.arctan2(sw_x, -sw_y))
    counts = l1_hdf.select("DNSWE_COUNT")[:]
    total_number_of_energies = 20
    spin_spectors = 30
    apertures = 7
    counts_in_swe_shape = np.moveaxis(counts, 3, 2).reshape((-1, total_number_of_energies, spin_spectors, apertures))

    starting_date_time = datetime(year=1999, month=9, day=6)
    epochs = starting_date_time + (timedelta(minutes=1) * np.arange(len(counts_in_swe_shape)))

    with open(path.parent.parent.parent / 'temp_cdf_data/swepam.txt', 'w') as file:
        dist_fun_1d_sw_index = vd.field('dist_fun_1d_sw')._index
        dist_fun_1d_sw_data = np.array([x[dist_fun_1d_sw_index] for x in vd[:]])
        np.savetxt(file, dist_fun_1d_sw_data)

    with CDF(l2_swe_cdf_file_path, readonly=False) as cdf:
        cdf["epoch"] = epochs

        evenly_spaced_energies = np.geomspace(1, 1350, 20, endpoint=True)
        replace_variable(cdf, "energy", evenly_spaced_energies)

        energies_reshaped = evenly_spaced_energies[:, np.newaxis, np.newaxis]
        psd = counts_in_swe_shape / np.square(energies_reshaped)
        replace_variable(cdf, "phase_space_density_spin_sector", psd)

        flux = psd * energies_reshaped
        replace_variable(cdf, "flux_spin_sector", flux)

        shape_without_aperture_axis = psd.shape[:-1]
        placeholder_insta_az_spin_sector = np.random.random(shape_without_aperture_axis) * 360
        replace_variable(cdf, "inst_az_spin_sector", placeholder_insta_az_spin_sector)

        minutes = len(counts_in_swe_shape)
        measurement_count = placeholder_insta_az_spin_sector.size
        start_time = (datetime(1999, 9, 6) - datetime(2010, 1, 1)).total_seconds()
        measurement_times = start_time + np.linspace(0, minutes * 60, measurement_count)
        placeholder_acquisition_time = measurement_times.reshape(shape_without_aperture_axis)
        replace_variable(cdf, "acquisition_time", placeholder_acquisition_time)
    with CDF(mag_file_path, readonly=False, create=True) as cdf:
        cdf["epoch"] = epochs
        cdf["vectors"] = mag_despun_vectors
        cdf["vectors"].attrs["FILLVAL"] = -1e31
    with CDF(swapi_file_path, readonly=False, create=True) as cdf:
        cdf['epoch'] = epochs
        cdf['epoch_delta'] = np.array([])
        cdf['proton_sw_speed'] = sw_speed
        cdf['proton_sw_speed'].attrs["FILLVAL"] = -1e31
        cdf['proton_sw_clock_angle'] = clock
        cdf['proton_sw_clock_angle'].attrs["FILLVAL"] = -1e31
        cdf['proton_sw_deflection_angle'] = deflection
        cdf['proton_sw_deflection_angle'].attrs["FILLVAL"] = -1e31


def replace_variable(cdf: CDF, variable_name: str, new_values: np.ndarray):
    del cdf[variable_name]
    cdf[variable_name] = new_values
    cdf[variable_name].attrs["FILLVAL"] = -1e31


if __name__ == "__main__":
    path = Path(__file__)
    l1_hdf_path = path.parent.parent.parent / "instrument_team_data/swe/ACE_LV1_1999-159.swepam.hdf"
    l2_hdf_path = path.parent.parent.parent / "instrument_team_data/swe/swepam-nswe-1999-159.v1-02.hdf"

    l2_swe_cdf_file_path = path.parent.parent.parent / "tests" / "test_data" / "swe" / "imap_swe_l2_sci-with-ace-data_19990906_v002.cdf"
    mag_file_path = path.parent.parent.parent / "tests" / "test_data" / "swe" / "imap_mag_l1d_mago-normal_19990906_v001.cdf"
    swapi_file_path = path.parent.parent.parent / "tests" / "test_data" / "swe" / "imap_swapi_l3a_proton-sw_19990906_v001.cdf"

    mag_file_path.unlink(missing_ok=True)
    swapi_file_path.unlink(missing_ok=True)
    try:
        create_fake_l2_cdf(str(l1_hdf_path),
                           str(l2_hdf_path),
                           str(l2_swe_cdf_file_path),
                           str(mag_file_path),
                           str(swapi_file_path))
    except Exception as e:
        traceback.print_exc()
