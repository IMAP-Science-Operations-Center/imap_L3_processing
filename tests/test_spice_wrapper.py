import unittest
from pathlib import Path
from unittest.mock import patch, call, Mock

import imap_processing
from imap_processing.spice_wrapper import furnish


class TestSpiceWrapper(unittest.TestCase):
    @patch('imap_processing.spice_wrapper.logging')
    @patch('imap_processing.spice_wrapper.Path')
    @patch('imap_processing.spice_wrapper.spiceypy.furnsh')
    def test_furnish_with_spice_dir_mounted(self, mock_furnsh, mock_path_constructor, _mock_logging):
        mock_mnt_spice_path = Mock()
        mock_imap_processing_path = Mock()
        mock_path_constructor.side_effect = [mock_imap_processing_path, mock_mnt_spice_path]
        mock_mnt_spice_path.is_dir.return_value = True

        kernel_1_path = Path("mock_kernel_path/kernel1")
        kernel_2_path = Path("mock_kernel_path/kernel2")
        mock_builtin_kernel_path = mock_imap_processing_path.parent.parent.joinpath.return_value

        mock_builtin_kernel_path.iterdir.return_value = [kernel_1_path, kernel_2_path]

        kernel_3_path = Path("/mnt/spice/kernel1")
        kernel_4_path = Path("/mnt/spice/kernel2")
        mock_mnt_spice_path.iterdir.return_value = [kernel_3_path, kernel_4_path]

        mock_furnsh.side_effect = [None, None, Exception, None]

        furnish()

        mock_imap_processing_path.parent.parent.joinpath.assert_called_with("spice_kernels")
        mock_path_constructor.assert_has_calls([
            call(imap_processing.__file__),
            call("/mnt/spice")
        ])

        mock_furnsh.assert_has_calls([
            call(str(kernel_1_path)),
            call(str(kernel_2_path)),
            call(str(kernel_3_path)),
            call(str(kernel_4_path)),
        ])

    @patch('imap_processing.spice_wrapper.Path')
    @patch('imap_processing.spice_wrapper.spiceypy.furnsh')
    def test_furnish_with_spice_dir_unmounted(self, mock_furnsh, mock_path_constructor):
        mock_mnt_spice_path = Mock()
        mock_imap_processing_path = Mock()
        mock_path_constructor.side_effect = [mock_imap_processing_path, mock_mnt_spice_path]
        mock_mnt_spice_path.is_dir.return_value = False
        mock_kernel_path = mock_imap_processing_path.parent.parent.joinpath.return_value

        kernel_1_path = Path("mock_kernel_path/kernel1")
        kernel_2_path = Path("mock_kernel_path/kernel2")
        mock_kernel_path.iterdir.return_value = [
            kernel_1_path,
            kernel_2_path,
        ]

        furnish()

        mock_imap_processing_path.parent.parent.joinpath.assert_called_with("spice_kernels")
        mock_path_constructor.assert_has_calls([
            call(imap_processing.__file__),
            call("/mnt/spice")
        ])
        mock_furnsh.assert_has_calls([
            call(str(kernel_1_path)),
            call(str(kernel_2_path)),
        ])


if __name__ == '__main__':
    unittest.main()
