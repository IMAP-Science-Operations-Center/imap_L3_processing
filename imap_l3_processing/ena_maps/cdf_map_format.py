import enum


class MapVars(enum.StrEnum):
    EPOCH = "epoch"
    EPOCH_DELTA = "epoch_delta"
    ENERGY = "energy"
    ENERGY_DELTA_PLUS = "energy_delta_plus"
    ENERGY_DELTA_MINUS = "energy_delta_minus"
    ENERGY_LABEL = "energy_label"
    LATITUDE = "latitude"
    LATITUDE_LABEL = "latitude_label"
    LONGITUDE = "longitude"
    LONGITUDE_LABEL = "longitude_label"
    EXPOSURE_FACTOR = "exposure_factor"
    OBS_DATE = "obs_date"
    OBS_DATE_RANGE = "obs_date_range"
    SOLID_ANGLE = "solid_angle"


class IntensityMapVars(MapVars):
    ENA_INTENSITY = "ena_intensity",
    ENA_INTENSITY_STAT_UNC = "ena_intensity_stat_unc",
    ENA_INTENSITY_SYS_ERR = "ena_intensity_sys_err"


class RectangularMapVars(MapVars):
    LATITUDE_DELTA = "latitude_delta"
    LONGITUDE_DELTA = "longitude_delta"


class HealpixMapVars(MapVars):
    HEALPIX_INDEX = "healpix_index"
