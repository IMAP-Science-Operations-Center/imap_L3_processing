import logging
from pathlib import Path

import numpy as np
from spiceypy import spiceypy

import imap_processing

FAKE_ROTATION_MATRIX_FROM_PSP = np.array(
    [[-8.03319036e-01, -5.95067395e-01, -2.39441182e-02, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00],
     [5.94803234e-01, -8.03675802e-01, 1.77289947e-02, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00],
     [-2.97932551e-02, 0.00000000e+00, 9.99556082e-01, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00],
     [-1.16314295e-06, 1.56750981e-06, 6.68593934e-08, -8.03319036e-01, -5.95067395e-01, -2.39441182e-02],
     [-1.56457525e-06, -1.16063465e-06, -1.21809529e-07, 5.94803234e-01, -8.03675802e-01, 1.77289947e-02],
     [1.26218156e-07, 5.29395592e-23, 3.76211978e-09, -2.97932551e-02, 0.00000000e+00, 9.99556082e-01]])


def furnish():
    logger = logging.getLogger(__name__)

    kernels = Path(imap_processing.__file__).parent.parent.joinpath("spice_kernels")
    for file in kernels.iterdir():
        logger.log(logging.INFO, f"loading packaged kernel: {file}")
        spiceypy.furnsh(str(file))

    kernels = Path("/mnt/spice")
    if kernels.is_dir():
        for file in kernels.iterdir():
            logger.log(logging.INFO, f"Spice file: {str(file)}")
            try:
                if file.is_symlink():
                    logger.log(logging.INFO, f"Spice file symlink: {file.readlink()} {file.exists()}")
                spiceypy.furnsh(str(file))
            except:
                logger.exception("Error while trying to load spice kernel: ")
