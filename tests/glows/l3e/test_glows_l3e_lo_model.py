import unittest
from datetime import datetime, timedelta
from unittest.mock import MagicMock, Mock, patch, sentinel

import numpy as np

from imap_l3_processing.glows.l3e.glows_l3e_call_arguments import (
    GlowsL3eCallArguments,
    GlowsL3eSpacecraftInfo,
)
from imap_l3_processing.glows.l3e.glows_l3e_lo_model import (
    ELONGATION_VAR_NAME,
    ENERGY_DELTA_MINUS_VAR_NAME,
    ENERGY_DELTA_PLUS_VAR_NAME,
    ENERGY_LABEL_VAR_NAME,
    ENERGY_VAR_NAME,
    EPOCH_CDF_VAR_NAME,
    EPOCH_DELTA_CDF_VAR_NAME,
    GLOWS_FLAGS_VAR_NAME,
    PROBABILITY_OF_SURVIVAL_VAR_NAME,
    PROGRAM_VERSION_VAR_NAME,
    SPACECRAFT_LATITUDE_VAR_NAME,
    SPACECRAFT_LONGITUDE_VAR_NAME,
    SPACECRAFT_RADIUS_VAR_NAME,
    SPACECRAFT_VELOCITY_X_VAR_NAME,
    SPACECRAFT_VELOCITY_Y_VAR_NAME,
    SPACECRAFT_VELOCITY_Z_VAR_NAME,
    SPIN_ANGLE_LABEL_VAR_NAME,
    SPIN_ANGLE_VAR_NAME,
    SPIN_AXIS_LATITUDE_VAR_NAME,
    SPIN_AXIS_LONGITUDE_VAR_NAME,
    GlowsL3ELoData,
)
from imap_l3_processing.models import DataProductVariable
from tests.test_helpers import NumpyArrayMatcher, get_test_instrument_team_data_path


class TestL3eLoModel(unittest.TestCase):
    def test_l3e_lo_model_to_data_product_variables(self):
        test_cases = {
            "case 1": (
                [1.2234, 2, 4.51, 66.7666],
                [1, 41650233.0, 4.22, 9.5],
                ["1.22", "2.00", "4.51", "66.77"],
                ["1", "41650233", "4", "10"],
            ),
            "case 2": (
                [4.536, 48.193, 4253.1],
                [13, 14.0, 34.2, 19.5],
                ["4.54", "48.19", "4253.10"],
                ["13", "14", "34", "20"],
            ),
        }

        for name, (
            energy_array,
            spin_angle_array,
            expected_energy_labels,
            expected_spin_angle_labels,
        ) in test_cases.items():
            with self.subTest(name):
                l3e_lo: GlowsL3ELoData = GlowsL3ELoData(
                    Mock(),
                    sentinel.epoch,
                    sentinel.epoch_delta,
                    energy_array,
                    sentinel.energy_delta_plus,
                    sentinel.energy_delta_minus,
                    spin_angle_array,
                    sentinel.probability_of_survival,
                    sentinel.elongation,
                    sentinel.spin_axis_latitude,
                    sentinel.spin_axis_longitude,
                    sentinel.program_version,
                    sentinel.spacecraft_radius,
                    sentinel.spacecraft_latitude,
                    sentinel.spacecraft_longitude,
                    sentinel.spacecraft_velocity_x,
                    sentinel.spacecraft_velocity_y,
                    sentinel.spacecraft_velocity_z,
                    sentinel.glows_flags,
                )
                data_products = l3e_lo.to_data_product_variables()

                expected_data_products = [
                    DataProductVariable(EPOCH_CDF_VAR_NAME, sentinel.epoch),
                    DataProductVariable(EPOCH_DELTA_CDF_VAR_NAME, sentinel.epoch_delta),
                    DataProductVariable(ENERGY_VAR_NAME, energy_array),
                    DataProductVariable(
                        ENERGY_DELTA_PLUS_VAR_NAME, sentinel.energy_delta_plus
                    ),
                    DataProductVariable(
                        ENERGY_DELTA_MINUS_VAR_NAME, sentinel.energy_delta_minus
                    ),
                    DataProductVariable(SPIN_ANGLE_VAR_NAME, spin_angle_array),
                    DataProductVariable(
                        PROBABILITY_OF_SURVIVAL_VAR_NAME,
                        sentinel.probability_of_survival,
                    ),
                    DataProductVariable(ENERGY_LABEL_VAR_NAME, expected_energy_labels),
                    DataProductVariable(
                        SPIN_ANGLE_LABEL_VAR_NAME, expected_spin_angle_labels
                    ),
                    DataProductVariable(ELONGATION_VAR_NAME, sentinel.elongation),
                    DataProductVariable(
                        SPIN_AXIS_LATITUDE_VAR_NAME,
                        np.array([sentinel.spin_axis_latitude]),
                    ),
                    DataProductVariable(
                        SPIN_AXIS_LONGITUDE_VAR_NAME,
                        np.array([sentinel.spin_axis_longitude]),
                    ),
                    DataProductVariable(
                        PROGRAM_VERSION_VAR_NAME, np.array([sentinel.program_version])
                    ),
                    DataProductVariable(
                        SPACECRAFT_RADIUS_VAR_NAME,
                        np.array([sentinel.spacecraft_radius]),
                    ),
                    DataProductVariable(
                        SPACECRAFT_LATITUDE_VAR_NAME,
                        np.array([sentinel.spacecraft_latitude]),
                    ),
                    DataProductVariable(
                        SPACECRAFT_LONGITUDE_VAR_NAME,
                        np.array([sentinel.spacecraft_longitude]),
                    ),
                    DataProductVariable(
                        SPACECRAFT_VELOCITY_X_VAR_NAME,
                        np.array([sentinel.spacecraft_velocity_x]),
                    ),
                    DataProductVariable(
                        SPACECRAFT_VELOCITY_Y_VAR_NAME,
                        np.array([sentinel.spacecraft_velocity_y]),
                    ),
                    DataProductVariable(
                        SPACECRAFT_VELOCITY_Z_VAR_NAME,
                        np.array([sentinel.spacecraft_velocity_z]),
                    ),
                    DataProductVariable(GLOWS_FLAGS_VAR_NAME, sentinel.glows_flags),
                ]
                self.assertEqual(expected_data_products, data_products)

    @patch("imap_l3_processing.glows.l3e.glows_l3e_lo_model.calculate_energy_deltas")
    def test_convert_dat_to_glows_l3e_lo_product(self, mock_calculate_energy_deltas):
        lo_file_path = get_test_instrument_team_data_path(
            "glows/probSur.Imap.Lo_20090101_010101_2009.000_60.00.txt"
        )
        epoch = datetime(year=2009, month=1, day=1)
        epoch_delta = timedelta(hours=3)
        expected_epoch_delta_in_nanoseconds = 3 * 3600 * 1e9
        expected_energy = [
            0.1700000,
            0.2212954,
            0.2880685,
            0.3749897,
            0.4881381,
            0.6354278,
            0.8271602,
            1.0767456,
            1.4016403,
            1.8245678,
            2.3751086,
            3.0917682,
            4.0246710,
        ]

        mock_calculate_energy_deltas.return_value = (
            sentinel.energy_delta_plus,
            sentinel.energy_delta_minus,
        )

        expected_prob_of_survival_first_col_1 = [
            0.48543928e00,
            0.48415770e00,
            0.48286930e00,
            0.48155365e00,
            0.48022886e00,
            0.47885535e00,
            0.47745419e00,
            0.47602527e00,
            0.47456880e00,
            0.47308723e00,
            0.47161934e00,
            0.47007995e00,
            0.46855039e00,
            0.46696817e00,
            0.46535270e00,
            0.46368022e00,
            0.46201160e00,
            0.46026144e00,
            0.45852743e00,
            0.45670696e00,
            0.45486823e00,
            0.45294031e00,
            0.45103086e00,
            0.44905075e00,
            0.44699322e00,
            0.44493658e00,
            0.44274056e00,
            0.44049151e00,
            0.43813752e00,
            0.43578356e00,
            0.43334287e00,
            0.43079981e00,
            0.42816069e00,
            0.42550222e00,
            0.42269926e00,
            0.41984889e00,
            0.41688223e00,
            0.41387416e00,
            0.41075224e00,
            0.40761023e00,
            0.40432881e00,
            0.40096472e00,
            0.39747450e00,
            0.39382089e00,
            0.39011010e00,
            0.38626253e00,
            0.38238974e00,
            0.37847099e00,
            0.37447377e00,
            0.37034872e00,
            0.36611061e00,
            0.36168805e00,
            0.35708535e00,
            0.35234758e00,
            0.34755946e00,
            0.34276461e00,
            0.33796280e00,
            0.33305954e00,
            0.32807347e00,
            0.32294017e00,
            0.31786557e00,
            0.31276043e00,
            0.30773144e00,
            0.30281685e00,
            0.29797800e00,
            0.29321404e00,
            0.28846079e00,
            0.28371704e00,
            0.27912920e00,
            0.27468323e00,
            0.27045568e00,
            0.26630037e00,
            0.26237361e00,
            0.25859967e00,
            0.25499865e00,
            0.25157148e00,
            0.24843419e00,
            0.24562329e00,
            0.24332850e00,
            0.24163423e00,
            0.24043665e00,
            0.23972507e00,
            0.23958627e00,
            0.23984571e00,
            0.24038628e00,
            0.24120584e00,
            0.24238245e00,
            0.24409800e00,
            0.24644643e00,
            0.24933851e00,
            0.25273734e00,
            0.25673467e00,
            0.26113737e00,
            0.26578085e00,
            0.27063217e00,
            0.27563385e00,
            0.28065683e00,
            0.28568105e00,
            0.29082592e00,
            0.29609185e00,
            0.30147878e00,
            0.30707066e00,
            0.31284106e00,
            0.31876025e00,
            0.32458878e00,
            0.32999671e00,
            0.33517199e00,
            0.34027166e00,
            0.34522320e00,
            0.35023574e00,
            0.35525591e00,
            0.36033213e00,
            0.36532967e00,
            0.37024320e00,
            0.37475598e00,
            0.37896325e00,
            0.38299882e00,
            0.38699896e00,
            0.39085028e00,
            0.39468049e00,
            0.39842597e00,
            0.40207629e00,
            0.40552207e00,
            0.40883026e00,
            0.41207903e00,
            0.41518970e00,
            0.41827498e00,
            0.42130951e00,
            0.42430284e00,
            0.42722657e00,
            0.43005367e00,
            0.43275769e00,
            0.43536314e00,
            0.43791211e00,
            0.44040868e00,
            0.44286096e00,
            0.44523738e00,
            0.44760685e00,
            0.44986863e00,
            0.45215006e00,
            0.45433928e00,
            0.45643657e00,
            0.45854771e00,
            0.46059383e00,
            0.46257694e00,
            0.46460808e00,
            0.46651408e00,
            0.46842937e00,
            0.47031930e00,
            0.47214870e00,
            0.47394193e00,
            0.47575376e00,
            0.47747515e00,
            0.47919249e00,
            0.48087054e00,
            0.48248460e00,
            0.48411980e00,
            0.48572180e00,
            0.48729084e00,
            0.48882613e00,
            0.49029764e00,
            0.49173477e00,
            0.49316337e00,
            0.49455725e00,
            0.49591591e00,
            0.49723950e00,
            0.49855476e00,
            0.49983531e00,
            0.50105632e00,
            0.50224193e00,
            0.50338975e00,
            0.50452839e00,
            0.50568552e00,
            0.50673081e00,
            0.50774070e00,
            0.50873875e00,
            0.50974595e00,
            0.51063701e00,
            0.51151521e00,
            0.51238110e00,
            0.51321059e00,
            0.51397275e00,
            0.51472495e00,
            0.51540664e00,
            0.51610368e00,
            0.51667264e00,
            0.51730154e00,
            0.51776070e00,
            0.51827571e00,
            0.51868642e00,
            0.51905088e00,
            0.51938042e00,
            0.51966850e00,
            0.51988677e00,
            0.52010423e00,
            0.52020368e00,
            0.52028969e00,
            0.52027278e00,
            0.52016485e00,
            0.52001351e00,
            0.51981105e00,
            0.51959358e00,
            0.51926552e00,
            0.51884032e00,
            0.51847414e00,
            0.51792884e00,
            0.51731622e00,
            0.51669638e00,
            0.51593604e00,
            0.51513940e00,
            0.51427166e00,
            0.51335442e00,
            0.51238909e00,
            0.51129305e00,
            0.51014602e00,
            0.50890185e00,
            0.50758409e00,
            0.50615044e00,
            0.50464683e00,
            0.50296036e00,
            0.50118289e00,
            0.49930168e00,
            0.49730277e00,
            0.49530416e00,
            0.49314260e00,
            0.49083236e00,
            0.48844397e00,
            0.48583154e00,
            0.48323435e00,
            0.48058649e00,
            0.47780153e00,
            0.47506716e00,
            0.47209482e00,
            0.46907585e00,
            0.46597907e00,
            0.46291516e00,
            0.45981653e00,
            0.45665363e00,
            0.45351121e00,
            0.45044243e00,
            0.44730392e00,
            0.44413627e00,
            0.44106236e00,
            0.43794686e00,
            0.43485646e00,
            0.43175941e00,
            0.42864794e00,
            0.42564675e00,
            0.42266427e00,
            0.41976102e00,
            0.41687658e00,
            0.41406969e00,
            0.41130938e00,
            0.40859441e00,
            0.40588626e00,
            0.40343902e00,
            0.40125003e00,
            0.39935862e00,
            0.39784753e00,
            0.39670361e00,
            0.39585956e00,
            0.39523582e00,
            0.39476103e00,
            0.39439905e00,
            0.39423968e00,
            0.39435100e00,
            0.39473203e00,
            0.39554310e00,
            0.39666286e00,
            0.39808096e00,
            0.39972118e00,
            0.40153794e00,
            0.40352131e00,
            0.40548763e00,
            0.40751478e00,
            0.40963578e00,
            0.41174409e00,
            0.41388333e00,
            0.41612846e00,
            0.41834220e00,
            0.42056149e00,
            0.42288866e00,
            0.42519516e00,
            0.42763343e00,
            0.43005765e00,
            0.43251298e00,
            0.43502684e00,
            0.43754215e00,
            0.44011421e00,
            0.44269314e00,
            0.44537654e00,
            0.44802922e00,
            0.45079502e00,
            0.45340240e00,
            0.45589495e00,
            0.45836848e00,
            0.46074029e00,
            0.46320577e00,
            0.46557171e00,
            0.46804433e00,
            0.47046801e00,
            0.47291016e00,
            0.47527772e00,
            0.47750468e00,
            0.47952019e00,
            0.48141970e00,
            0.48317952e00,
            0.48487642e00,
            0.48650926e00,
            0.48807554e00,
            0.48963920e00,
            0.49111387e00,
            0.49259059e00,
            0.49402593e00,
            0.49539224e00,
            0.49660220e00,
            0.49767488e00,
            0.49868284e00,
            0.49962355e00,
            0.50044291e00,
            0.50121762e00,
            0.50196273e00,
            0.50261243e00,
            0.50326871e00,
            0.50387423e00,
            0.50434609e00,
            0.50480509e00,
            0.50515270e00,
            0.50543081e00,
            0.50566424e00,
            0.50580062e00,
            0.50590890e00,
            0.50589201e00,
            0.50588645e00,
            0.50578122e00,
            0.50561852e00,
            0.50539304e00,
            0.50510737e00,
            0.50474719e00,
            0.50435977e00,
            0.50384726e00,
            0.50336489e00,
            0.50279180e00,
            0.50216504e00,
            0.50148265e00,
            0.50078978e00,
            0.50000804e00,
            0.49918459e00,
            0.49836367e00,
            0.49744147e00,
            0.49651714e00,
            0.49557195e00,
            0.49456744e00,
            0.49353034e00,
            0.49245761e00,
            0.49135232e00,
            0.49023631e00,
            0.48909069e00,
            0.48791635e00,
            0.48669295e00,
        ]

        expected_spin_angle = np.arange(1, 361, 1, dtype=np.float64)
        elongation_value = 75
        expected_survival_probability_shape = (1, 13, 360)

        mock_metadata = Mock()

        spin_axis_lat = 45.0
        spin_axis_lon = 90.0

        args = MagicMock(spec=GlowsL3eCallArguments)
        expected_program_version = "Lo.v00.01"

        spacecraft_info = MagicMock(spec=GlowsL3eSpacecraftInfo)
        spacecraft_info.spin_axis_latitude = spin_axis_lat
        spacecraft_info.spin_axis_longitude = spin_axis_lon
        spacecraft_info.spacecraft_radius = 0.5
        spacecraft_info.spacecraft_longitude = 85.4
        spacecraft_info.spacecraft_latitude = 45.1

        spacecraft_info.spacecraft_velocity_x = 2.1
        spacecraft_info.spacecraft_velocity_y = 2.2
        spacecraft_info.spacecraft_velocity_z = 2.3

        args.spacecraft_info = spacecraft_info

        l3e_lo_product: GlowsL3ELoData = (
            GlowsL3ELoData.convert_dat_to_glows_l3e_lo_product(
                mock_metadata, lo_file_path, epoch, epoch_delta, elongation_value, args
            )
        )

        mock_calculate_energy_deltas.assert_called_once_with(
            NumpyArrayMatcher(l3e_lo_product.energy)
        )

        np.testing.assert_equal([epoch], l3e_lo_product.epoch, strict=True)
        np.testing.assert_equal(
            [expected_epoch_delta_in_nanoseconds],
            l3e_lo_product.epoch_delta,
            strict=True,
        )
        np.testing.assert_equal(l3e_lo_product.energy, expected_energy, strict=True)
        np.testing.assert_equal(
            l3e_lo_product.energy_delta_plus, sentinel.energy_delta_plus
        )
        np.testing.assert_equal(
            l3e_lo_product.energy_delta_minus,
            sentinel.energy_delta_minus,
        )

        np.testing.assert_equal(
            l3e_lo_product.spin_angle, expected_spin_angle, strict=True
        )
        np.testing.assert_equal(
            l3e_lo_product.probability_of_survival.shape,
            expected_survival_probability_shape,
            strict=True,
        )
        np.testing.assert_equal(
            l3e_lo_product.probability_of_survival[0][0],
            expected_prob_of_survival_first_col_1,
            strict=True,
        )
        np.testing.assert_equal(
            l3e_lo_product.elongation, np.array([elongation_value]), strict=True
        )

        np.testing.assert_equal(
            np.array([spin_axis_lat]), l3e_lo_product.spin_axis_lat, strict=True
        )
        np.testing.assert_equal(
            np.array([spin_axis_lon]), l3e_lo_product.spin_axis_lon, strict=True
        )

        np.testing.assert_equal(
            [expected_program_version], l3e_lo_product.program_version, strict=True
        )

        np.testing.assert_equal(
            l3e_lo_product.spacecraft_radius, np.array([0.5]), strict=True
        )
        np.testing.assert_equal(
            l3e_lo_product.spacecraft_longitude, np.array([85.4]), strict=True
        )
        np.testing.assert_equal(
            l3e_lo_product.spacecraft_latitude, np.array([45.1]), strict=True
        )

        np.testing.assert_equal(
            l3e_lo_product.spacecraft_velocity_x, np.array([2.1]), strict=True
        )
        np.testing.assert_equal(
            l3e_lo_product.spacecraft_velocity_y, np.array([2.2]), strict=True
        )
        np.testing.assert_equal(
            l3e_lo_product.spacecraft_velocity_z, np.array([2.3]), strict=True
        )
