import unittest

import numpy as np

from bpp.params import BppParams


class TestBppParams(unittest.TestCase):
    def setUp(self) -> None:
        item_weight = [1, 2, 3, 4, 5]
        bin_capacity = [6, 7, 8, 9]
        self.params = BppParams(item_weight=item_weight, bin_capacity=bin_capacity)

    def test_n_bins(self) -> None:
        self.assertEqual(self.params.n_bins, 4)

    def test_n_items(self) -> None:
        self.assertEqual(self.params.n_items, 5)

    def test_N(self) -> None:
        N = [3, 3, 4, 4]
        self.assertListEqual(list(self.params.N), N)

    def test_sum_N(self) -> None:
        self.assertEqual(self.params.sum_N, 14)

    def test_shape_Q(self) -> None:
        self.assertTupleEqual(self.params.shape_Q, (38, 38))

    def test_shape_Qxx(self) -> None:
        self.assertTupleEqual(self.params.shape_Qxx, (20, 20))

    def test_shape_Qxy(self) -> None:
        self.assertTupleEqual(self.params.shape_Qxy, (20, 4))

    def test_shape_Qxs(self) -> None:
        self.assertTupleEqual(self.params.shape_Qxs, (20, 14))

    def test_shape_Qyx(self) -> None:
        self.assertTupleEqual(self.params.shape_Qyx, (4, 20))

    def test_shape_Qyy(self) -> None:
        self.assertTupleEqual(self.params.shape_Qyy, (4, 4))

    def test_shape_Qys(self) -> None:
        self.assertTupleEqual(self.params.shape_Qys, (4, 14))

    def test_shape_Qsx(self) -> None:
        self.assertTupleEqual(self.params.shape_Qsx, (14, 20))

    def test_shape_Qsy(self) -> None:
        self.assertTupleEqual(self.params.shape_Qsy, (14, 4))

    def test_shape_Qss(self) -> None:
        self.assertTupleEqual(self.params.shape_Qss, (14, 14))

    def test_total_weight_exceeds_total_capacity(self) -> None:
        with self.assertRaises(AssertionError):
            BppParams(item_weight=[2, 2], bin_capacity=[3])

    def test_max_weight_exceeds_max_capacity(self) -> None:
        with self.assertRaises(AssertionError):
            BppParams(item_weight=[3], bin_capacity=[2, 2])

    def test_convert_float_weights_to_integers(self) -> None:
        params = BppParams(item_weight=[1.0, 2.0, 3.0], bin_capacity=[6])
        self.assertListEqual(list(params.item_weight), [1, 2, 3])

    def test_convert_float_capacity_to_integers(self) -> None:
        params = BppParams(item_weight=[1], bin_capacity=[1.0, 2.0, 3.0])
        self.assertListEqual(list(params.bin_capacity), [1, 2, 3])

    def test_non_positive_weight_raises(self) -> None:
        with self.assertRaises(AssertionError):
            BppParams(item_weight=[0], bin_capacity=[1])

    def test_non_positive_capacity_raises(self) -> None:
        with self.assertRaises(AssertionError):
            BppParams(item_weight=[1], bin_capacity=[-1])

    def test_float_weight_raises(self) -> None:
        with self.assertRaises(AssertionError):
            BppParams(item_weight=[1.1], bin_capacity=[2])

    def test_float_capacity_raises(self) -> None:
        with self.assertRaises(AssertionError):
            BppParams(item_weight=[1], bin_capacity=[1.1])


if __name__ == "__main__":
    unittest.main()
