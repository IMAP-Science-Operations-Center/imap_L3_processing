from contextlib import contextmanager

import numpy as np

FAKE_SPICE_DATA_FROM_PSP = np.array([-2.79573704e+07, 2.06980031e+07, -1.03669775e+06, -7.05092444e+01,
                                     -3.22099099e+01, 3.27852669e+00])


@contextmanager
def fake_spice_context():
    yield FakeSpiceContext()


class FakeSpiceContext:
    def spkezr(self, target, observer_epoch, reference_frame, abcorr, observer):
        if target == "IMAP" and reference_frame == "HCI" and observer == "SUN":
            return FAKE_SPICE_DATA_FROM_PSP
        else:
            raise ValueError("Do not have fake spice data for those parameters")
