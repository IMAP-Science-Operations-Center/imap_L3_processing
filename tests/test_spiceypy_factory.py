import unittest
from pathlib import Path

import tests
from imap_l3_processing.spicepy_factory import SpiceypyFactory


class TestSpiceypyFactory(unittest.TestCase):

    def setUp(self) -> None:
        SpiceypyFactory.get_spiceypy().kclear()

    def test_get_spiceypy_has_zero_kernels(self):
        spiceypy = SpiceypyFactory.get_spiceypy()
        self.assertEqual(0, spiceypy.ktotal('ALL'))

    def test_furnish_will_add_kernels_to_spiceypy(self):
        kernel_parent_path = Path(tests.__file__).parent.parent / 'spice_kernels'
        naif_kernel_path = kernel_parent_path / 'naif0012.tls'
        sclk_kernel_path = kernel_parent_path / 'imap_sclk_0000.tsc'

        expected_kernels = [naif_kernel_path, sclk_kernel_path]

        SpiceypyFactory.furnish(expected_kernels)

        spiceypy = SpiceypyFactory.get_spiceypy()
        self.assertEqual(2, spiceypy.ktotal('ALL'))
        self.assertEqual(str(expected_kernels[0]), spiceypy.kdata(0, 'ALL')[0])
        self.assertEqual(str(expected_kernels[1]), spiceypy.kdata(1, 'ALL')[0])
