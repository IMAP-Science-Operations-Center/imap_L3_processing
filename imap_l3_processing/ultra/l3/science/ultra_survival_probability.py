import xarray as xr
from imap_processing.ena_maps.ena_maps import UltraPointingSet
from imap_processing.ena_maps.utils.coordinates import CoordNames
from imap_processing.spice import geometry


class UltraSurvivalProbability(UltraPointingSet):
    def __init__(self, l1cdataset: xr.Dataset):
        super().__init__(l1cdataset, geometry.SpiceFrame.ECLIPJ2000)

        self.data["survival_probability_times_exposure"] = (
            [
                CoordNames.TIME.value,
                CoordNames.ENERGY.value,
                CoordNames.HEALPIX_INDEX.value
            ],
            self.data["counts"].values
        )
