import unittest

import numpy as np

from bpp.qubo import Qubo


class TestQubo(unittest.TestCase):
    def setUp(self) -> None:
        # abstract example
        items = [1, 2, 3, 4]
        bins = [5, 6, 7]
        lambda_ec = [1.0, 1.0, 1.0, 1.0]
        lambda_ic = [2.0, 2.0, 2.0]
        self.qubo = Qubo(items, bins, lambda_ec, lambda_ic)

        # concrete example
        bins = [5, 5]
        items = [5, 5]
        lambda_ec = [1, 1]
        lambda_ic = [1, 1]
        self.qubo_concrete = Qubo(items, bins, lambda_ec, lambda_ic)

    def test_n_bins(self) -> None:
        self.assertEqual(self.qubo.n_bins, 3)

    def test_n_items(self) -> None:
        self.assertEqual(self.qubo.n_items, 4)

    def test_N(self) -> None:
        self.assertListEqual(list(self.qubo.N), [3, 3, 3])

    def test_offset(self) -> None:
        self.assertEqual(self.qubo.offset, 4)

    def test_x_labels(self) -> None:
        labels = [
            "x[0,0]",
            "x[0,1]",
            "x[0,2]",
            "x[1,0]",
            "x[1,1]",
            "x[1,2]",
            "x[2,0]",
            "x[2,1]",
            "x[2,2]",
            "x[3,0]",
            "x[3,1]",
            "x[3,2]",
        ]
        self.assertListEqual(self.qubo.x_labels, labels)

    def test_y_labels(self) -> None:
        labels = ["y[0]", "y[1]", "y[2]"]
        self.assertListEqual(self.qubo.y_labels, labels)

    def test_s_labels(self) -> None:
        labels = [
            "s[0,0]",
            "s[0,1]",
            "s[0,2]",
            "s[1,0]",
            "s[1,1]",
            "s[1,2]",
            "s[2,0]",
            "s[2,1]",
            "s[2,2]",
        ]
        self.assertListEqual(self.qubo.s_labels, labels)

    def test_x_hat_labels(self) -> None:
        labels = [
            "x[0,0]",
            "x[0,1]",
            "x[0,2]",
            "x[1,0]",
            "x[1,1]",
            "x[1,2]",
            "x[2,0]",
            "x[2,1]",
            "x[2,2]",
            "x[3,0]",
            "x[3,1]",
            "x[3,2]",
            "y[0]",
            "y[1]",
            "y[2]",
            "s[0,0]",
            "s[0,1]",
            "s[0,2]",
            "s[1,0]",
            "s[1,1]",
            "s[1,2]",
            "s[2,0]",
            "s[2,1]",
            "s[2,2]",
        ]
        self.assertListEqual(self.qubo.x_hat_labels, labels)

    def test_check_vadility_with_valid_solution(self) -> None:
        x = np.array(
            [
                [1, 0, 0],
                [1, 0, 0],
                [0, 1, 0],
                [0, 0, 1],
            ]
        ).flatten()
        y = np.array([1, 1, 1])
        solution = np.concatenate((x, y, np.zeros(9)))
        self.assertTrue(self.qubo.check_vadility(solution))

    def test_check_vadility_with_double_bin_assignment(self) -> None:
        x = np.array(
            [
                [1, 1, 0],  # double bin assignment
                [1, 0, 0],
                [0, 1, 0],
                [0, 0, 1],
            ]
        ).flatten()
        y = np.array([1, 1, 1])
        solution = np.concatenate((x, y, np.zeros(9)))
        self.assertFalse(self.qubo.check_vadility(solution))

    def test_check_vadility_with_missing_bin_assignment(self) -> None:
        x = np.array(
            [
                [0, 0, 0],  # missing bin assignment
                [1, 0, 0],
                [0, 1, 0],
                [0, 0, 1],
            ]
        ).flatten()
        y = np.array([1, 1, 1])
        solution = np.concatenate((x, y, np.zeros(9)))
        self.assertFalse(self.qubo.check_vadility(solution))

    def test_check_vadility_with_capacity_exceed(self) -> None:
        x = np.array(
            [
                [1, 0, 0],
                [1, 0, 0],
                [1, 0, 0],
                [1, 0, 0],
            ]
        ).flatten()
        y = np.array([1, 0, 0])
        solution = np.concatenate((x, y, np.zeros(9)))
        self.assertFalse(self.qubo.check_vadility(solution))

    def test_check_vadility_with_bin_selection_mismatch(self) -> None:
        x = np.array(
            [
                [1, 0, 0],
                [1, 0, 0],
                [0, 1, 0],
                [0, 0, 1],  # not selected in y
            ]
        ).flatten()
        y = np.array([1, 1, 0])
        solution = np.concatenate((x, y, np.zeros(9)))
        self.assertFalse(self.qubo.check_vadility(solution))

    def test_Q_size(self) -> None:
        self.assertEqual(self.qubo.Q_size, len(self.qubo.x_hat_labels))

    def test_x2global(self) -> None:
        self.assertEqual(self.qubo.x2global(0, 0), 0)
        self.assertEqual(self.qubo.x2global(0, 2), 2)
        self.assertEqual(self.qubo.x2global(3, 0), 9)
        self.assertEqual(self.qubo.x2global(3, 2), 11)

    def test_y2global(self) -> None:
        self.assertEqual(self.qubo.y2global(0), 12)
        self.assertEqual(self.qubo.y2global(2), 14)

    def test_s2global(self) -> None:
        self.assertEqual(self.qubo.s2global(0, 0), 15)
        self.assertEqual(self.qubo.s2global(0, 2), 17)
        self.assertEqual(self.qubo.s2global(1, 0), 18)
        self.assertEqual(self.qubo.s2global(2, 2), 23)

    def test_Q_shape(self) -> None:
        self.assertEqual(self.qubo.Q.shape, (24, 24))

    def test_Qxx_shape(self) -> None:
        self.assertEqual(self.qubo.Qxx.shape, (12, 12))

    def test_Qxy_shape(self) -> None:
        self.assertEqual(self.qubo.Qxy.shape, (12, 3))

    def test_Qxs_shape(self) -> None:
        self.assertEqual(self.qubo.Qxs.shape, (12, 9))

    def test_Qyy_shape(self) -> None:
        self.assertEqual(self.qubo.Qyy.shape, (3, 3))

    def test_Qys_shape(self) -> None:
        self.assertEqual(self.qubo.Qys.shape, (3, 9))

    def test_Qss_shape(self) -> None:
        self.assertEqual(self.qubo.Qss.shape, (9, 9))

    def test_Q(self) -> None:
        Q = np.array(
            [
                [24.0, 2.0, 50.0, 0.0, -50.0, 0.0, 10.0, 20.0, 40.0, 0.0, 0.0, 0.0],
                [0.0, 24.0, 0.0, 50.0, 0.0, -50.0, 0.0, 0.0, 0.0, 10.0, 20.0, 40.0],
                [0.0, 0.0, 24.0, 2.0, -50.0, 0.0, 10.0, 20.0, 40.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 24.0, 0.0, -50.0, 0.0, 0.0, 0.0, 10.0, 20.0, 40.0],
                [0.0, 0.0, 0.0, 0.0, 26.0, 0.0, -10.0, -20.0, -40.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 26.0, 0.0, 0.0, 0.0, -10.0, -20.0, -40.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 4.0, 8.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 4.0, 16.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 16.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 4.0, 8.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 4.0, 16.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 16.0],
            ]
        )

        self.assertEqual(self.qubo_concrete.Q.shape, (12, 12))
        self.assertTrue(np.array_equal(self.qubo_concrete.Q, Q))

    def test_Qxx(self) -> None:
        Qxx = np.array(
            [
                [24.0, 2.0, 50.0, 0.0],
                [0.0, 24.0, 0.0, 50.0],
                [0.0, 0.0, 24.0, 2.0],
                [0.0, 0.0, 0.0, 24.0],
            ]
        )

        self.assertEqual(self.qubo_concrete.Qxx.shape, (4, 4))
        self.assertTrue(np.array_equal(self.qubo_concrete.Qxx, Qxx))

    def test_Qxy(self) -> None:
        Qxy = np.array(
            [
                [-50.0, 0.0],
                [0.0, -50.0],
                [-50.0, 0.0],
                [0.0, -50.0],
            ]
        )

        self.assertEqual(self.qubo_concrete.Qxy.shape, (4, 2))
        self.assertTrue(np.array_equal(self.qubo_concrete.Qxy, Qxy))

    def test_Qxs(self) -> None:
        Qxs = np.array(
            [
                [10.0, 20.0, 40.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 10.0, 20.0, 40.0],
                [10.0, 20.0, 40.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 10.0, 20.0, 40.0],
            ]
        )

        self.assertEqual(self.qubo_concrete.Qxs.shape, (4, 6))
        self.assertTrue(np.array_equal(self.qubo_concrete.Qxs, Qxs))

    def test_Qyy(self) -> None:
        Qyy = np.array(
            [
                [26.0, 0.0],
                [0.0, 26.0],
            ]
        )

        self.assertEqual(self.qubo_concrete.Qyy.shape, (2, 2))
        self.assertTrue(np.array_equal(self.qubo_concrete.Qyy, Qyy))

    def test_Qys(self) -> None:
        Qys = np.array(
            [
                [-10.0, -20.0, -40.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, -10.0, -20.0, -40.0],
            ]
        )

        self.assertEqual(self.qubo_concrete.Qys.shape, (2, 6))
        self.assertTrue(np.array_equal(self.qubo_concrete.Qys, Qys))

    def test_Qss(self) -> None:
        Qss = np.array(
            [
                [1.0, 4.0, 8.0, 0.0, 0.0, 0.0],
                [0.0, 4.0, 16.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 16.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0, 4.0, 8.0],
                [0.0, 0.0, 0.0, 0.0, 4.0, 16.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 16.0],
            ]
        )

        self.assertEqual(self.qubo_concrete.Qss.shape, (6, 6))
        self.assertTrue(np.array_equal(self.qubo_concrete.Qss, Qss))


if __name__ == "__main__":
    unittest.main()
