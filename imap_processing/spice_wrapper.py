from contextlib import contextmanager
from pathlib import Path

import numpy as np
from spiceypy import spiceypy

import imap_processing

FAKE_SPICE_DATA_FROM_PSP = np.array([-2.79573704e+07, 2.06980031e+07, -1.03669775e+06, -7.05092444e+01,
                                     -3.22099099e+01, 3.27852669e+00])

FAKE_ROTATION_MATRIX_FROM_PSP = np.array(
    [[-8.03319036e-01, -5.95067395e-01, -2.39441182e-02, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00],
     [5.94803234e-01, -8.03675802e-01, 1.77289947e-02, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00],
     [-2.97932551e-02, 0.00000000e+00, 9.99556082e-01, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00],
     [-1.16314295e-06, 1.56750981e-06, 6.68593934e-08, -8.03319036e-01, -5.95067395e-01, -2.39441182e-02],
     [-1.56457525e-06, -1.16063465e-06, -1.21809529e-07, 5.94803234e-01, -8.03675802e-01, 1.77289947e-02],
     [1.26218156e-07, 5.29395592e-23, 3.76211978e-09, -2.97932551e-02, 0.00000000e+00, 9.99556082e-01]])


@contextmanager
def fake_spice_context():
    yield FakeSpiceContext()


class FakeSpiceContext:
    def spkezr(self, target, observer_epoch, reference_frame, abcorr, observer):
        if target == "IMAP" and reference_frame == "HCI" and observer == "SUN":
            return FAKE_SPICE_DATA_FROM_PSP
        else:
            raise ValueError("Do not have fake spice data for those parameters")

    def sxform(self, from_frame: str, to_frame: str, ephemeris_time: float) -> np.ndarray:
        if from_frame == "IMAP-SWAPI" and to_frame == "HCI":
            return FAKE_ROTATION_MATRIX_FROM_PSP
        else:
            raise ValueError("Do not have fake space data for those parameters")

    def reclat(self, ):
        pass


def furnish():
    if Path("/mnt/spice").is_dir():
        kernels = Path("/mnt/spice")
    else:
        kernels = Path(imap_processing.__file__).parent.parent.joinpath("spice_kernels")
    for file in kernels.iterdir():
        spiceypy.furnsh(str(file))
