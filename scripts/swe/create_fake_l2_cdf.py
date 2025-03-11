import traceback
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pyhdf
import pyhdf.SD
from pyhdf.VS import *
from pyhdf.HDF import *
from spacepy import pycdf
from spacepy.pycdf import CDF


def create_fake_swe_l2_cdf(l1_hdf_path: str, l2_hdf_path: str, output_l2_swe_cdf_file_path: str):
    l1_hdf = pyhdf.SD.SD(l1_hdf_path)
    counts = l1_hdf.select("DNSWE_COUNT")[:]
    total_number_of_energies = 20
    spin_spectors = 30
    apertures = 7
    counts_in_swe_shape = np.moveaxis(counts, 3, 2).reshape((-1, total_number_of_energies, spin_spectors, apertures))

    l2_swepam_electron_file = HDF(l2_hdf_path)
    vs_electron = l2_swepam_electron_file.vstart()
    vd_electron = vs_electron.attach("swepam_e")
    epochs = get_epochs_from_output_file(vd_electron)

    with CDF(output_l2_swe_cdf_file_path, readonly=False) as cdf:
        cdf.compress(pycdf.const.GZIP_COMPRESSION)
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
        measurement_times = np.linspace(0, minutes * 60, measurement_count)
        placeholder_acquisition_time = measurement_times.reshape(shape_without_aperture_axis)
        replace_variable(cdf, "acquisition_time", placeholder_acquisition_time)


def create_fake_swapi_l3a_cdf(l2_swepam_ion_file_path: str,
                              output_swapi_file_path: str):
    l2_swepam_ion_file = HDF(l2_swepam_ion_file_path)
    vs_ion = l2_swepam_ion_file.vstart()
    vd_ion = vs_ion.attach("swepam_i")

    sw_velocity_index = vd_ion.field(
        "vel_p_rtn")._index
    sw_velocity_data = np.array([x[sw_velocity_index] for x in vd_ion[:]])
    sw_speed = np.linalg.norm(sw_velocity_data, axis=1)
    sw_r, sw_t, sw_n = sw_velocity_data.T
    sw_x, sw_y, sw_z = sw_n, sw_t, -sw_r
    deflection = np.rad2deg(np.arctan2(np.sqrt(sw_x ** 2 + sw_y ** 2), -sw_z))
    clock = np.rad2deg(np.arctan2(sw_x, -sw_y))

    epochs = get_epochs_from_output_file(vd_ion)

    epoch_deltas = np.full(epochs.shape, 32_000_000_000)

    with CDF(output_swapi_file_path, readonly=False, create=True) as cdf:
        cdf.compress(pycdf.const.GZIP_COMPRESSION)
        cdf['epoch'] = epochs
        cdf['epoch_delta'] = epoch_deltas
        cdf['proton_sw_speed'] = sw_speed
        cdf['proton_sw_speed'].attrs["FILLVAL"] = -1e31
        cdf['proton_sw_clock_angle'] = clock
        cdf['proton_sw_clock_angle'].attrs["FILLVAL"] = -1e31
        cdf['proton_sw_deflection_angle'] = deflection
        cdf['proton_sw_deflection_angle'].attrs["FILLVAL"] = -1e31


def create_fake_mag_l1d_cdf(l2_swepam_electron_file_path: str,
                            output_mag_file_path: str):
    l2_swepam_electron_file_path = HDF(l2_swepam_electron_file_path)
    vs_electron = l2_swepam_electron_file_path.vstart()
    vd_electron = vs_electron.attach("swepam_e")

    b_rtn_index = vd_electron.field("b_rtn")._index
    b_rtn_data = np.array([x[b_rtn_index] for x in vd_electron[:]])
    mag_r, mag_t, mag_n = b_rtn_data.T
    mag_despun_vectors = np.stack((mag_n, mag_t, -mag_r), axis=1)

    with CDF(output_mag_file_path, readonly=False, create=True) as cdf:
        cdf.compress(pycdf.const.GZIP_COMPRESSION)
        cdf["epoch"] = get_epochs_from_output_file(vd_electron)
        cdf["vectors"] = mag_despun_vectors
        cdf["vectors"].attrs["FILLVAL"] = -1e31

    # if we want to write out expected 1d output
    # alternative: make CDF for visual comparison?
    # with open(path.parent.parent.parent / 'temp_cdf_data/swepam.txt', 'w') as file:
    #     dist_fun_1d_sw_index = vd.field('dist_fun_1d_sw')._index
    #     dist_fun_1d_sw_data = np.array([x[dist_fun_1d_sw_index] for x in vd[:]])
    #     np.savetxt(file, dist_fun_1d_sw_data)


def replace_variable(cdf: CDF, variable_name: str, new_values: np.ndarray):
    del cdf[variable_name]
    cdf[variable_name] = new_values
    cdf[variable_name].attrs["FILLVAL"] = -1e31


def get_epochs_from_output_file(dataset: VD) -> np.array:
    years_index = dataset.field("year")._index
    month_index = dataset.field("mon")._index
    day_index = dataset.field("mday")._index
    hour_index = dataset.field("hour")._index
    min_index = dataset.field("min")._index
    sec_index = dataset.field("sec")._index

    correction_factor = (datetime(2010, 1, 1) - datetime(1999, 6, 8))

    return np.array([datetime(year=x[years_index], month=x[month_index], day=x[day_index], hour=x[hour_index],
                              minute=x[min_index], second=x[sec_index]) + correction_factor for x in dataset[:]])


if __name__ == "__main__":
    path = Path(__file__)
    l1_hdf_path = path.parent.parent.parent / "instrument_team_data/swe/ACE_LV1_1999-159.swepam.hdf"
    l2_electron_hdf_path = path.parent.parent.parent / "instrument_team_data/swe/swepam-nswe-1999-159.v1-02.hdf"
    l2_ion_hdf_path = path.parent.parent.parent / "instrument_team_data/swe/swepam-swi-1999-159.v2-01.hdf"

    l2_swe_cdf_file_path = path.parent.parent.parent / "tests" / "test_data" / "swe" / "imap_swe_l2_sci-with-ace-data_20100101_v002.cdf"
    mag_file_path = path.parent.parent.parent / "tests" / "test_data" / "swe" / "imap_mag_l1d_mago-normal_20100101_v001.cdf"
    swapi_file_path = path.parent.parent.parent / "tests" / "test_data" / "swe" / "imap_swapi_l3a_proton-sw_20100101_v001.cdf"

    mag_file_path.unlink(missing_ok=True)
    swapi_file_path.unlink(missing_ok=True)
    create_fake_swe_l2_cdf(str(l1_hdf_path), str(l2_electron_hdf_path), str(l2_swe_cdf_file_path))
    create_fake_mag_l1d_cdf(str(l2_electron_hdf_path), str(mag_file_path))
    create_fake_swapi_l3a_cdf(str(l2_ion_hdf_path), str(swapi_file_path))
