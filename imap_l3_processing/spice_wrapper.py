import logging
from pathlib import Path

import spiceypy as spice

spiceypy = spice

import imap_l3_processing

furnished = False


def furnish():
    global furnished
    if furnished:
        return
    logger = logging.getLogger(__name__)

    kernels = Path(imap_l3_processing.__file__).parent.parent.joinpath("spice_kernels")
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
    furnished = True


def ensure_furnished():
    if not furnished:
        furnish()


furnish()
