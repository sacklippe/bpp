import itertools
from typing import Any

import numpy as np
from numpy.typing import ArrayLike


class Qubo:
    def __init__(
        self,
        weights: ArrayLike,
        bin_capacities: ArrayLike,
        lambda_EC: ArrayLike,
        lambda_IC: ArrayLike,
    ) -> None:
        weights = np.array(weights)
        bin_capacities = np.array(bin_capacities)
        lambda_EC = np.array(lambda_EC)
        lambda_IC = np.array(lambda_IC)

        assert len(bin_capacities.shape) == 1, "Bin capacities must be a vector"
        assert len(weights.shape) == 1, "Item weights must be a vector"
        assert len(lambda_EC.shape) == 1, "lambda_EC must be a vector"
        assert len(lambda_IC.shape) == 1, "lambda_IC must be a vector"
        assert np.all(np.array(bin_capacities) > 0), "Bin capacities must be positive"
        assert np.all(np.array(weights) > 0), "Item weights must be positive"
        is_integer = np.floor(bin_capacities) == bin_capacities
        assert np.all(is_integer), "Bin capacities must be integers"
        is_integer = np.floor(weights) == weights
        assert np.all(is_integer), "Item weights must be integers"
        max_weight_fits = np.max(bin_capacities) >= np.max(weights)
        assert max_weight_fits, "Heaviest item exceeds max. bin capacity"
        total_weights_fit = np.sum(bin_capacities) >= np.sum(weights)
        assert total_weights_fit, "Total weights exceed total bin capacity"
        match_size = lambda_EC.shape == weights.shape
        assert match_size, "lambda_EC must be a vector of length n_bins"
        match_size = lambda_IC.shape == bin_capacities.shape
        assert match_size, "lambda_IC must be a vector of length n_items"
        assert np.all(lambda_EC > 0), "lambda_EC must be positive"
        assert np.all(lambda_IC > 0), "lambda_IC must be positive"

        self.weights = weights
        self.bin_capacities = bin_capacities
        self.lambda_EC = lambda_EC
        self.lambda_IC = lambda_IC

    @property
    def n_bins(self) -> int:
        return self.bin_capacities.shape[0]

    @property
    def n_items(self) -> int:
        return self.weights.shape[0]

    @property
    def N(self) -> np.ndarray[Any, np.dtype[np.int64]]:
        return (np.floor(np.log2(self.bin_capacities)) + 1).astype(int)

    @property
    def offset(self) -> float:
        return np.sum(self.lambda_EC)

    @property
    def x_labels(self) -> list[str]:
        permutations = itertools.product(range(self.n_items), range(self.n_bins))
        return [f"x[{i},{j}]" for i, j in permutations]

    @property
    def y_labels(self) -> list[str]:
        return [f"y[{j}]" for j in range(self.n_bins)]

    @property
    def s_labels(self) -> list[str]:
        labels = []
        for j in range(self.n_bins):
            labels += [f"s[{j},{k}]" for k in range(self.N[j])]
        return labels

    @property
    def x_hat_labels(self) -> list[str]:
        return self.x_labels + self.y_labels + self.s_labels

    def x2global(self, i: int, j: int) -> int:
        """
        Map x-index to global x_hat-index.
        """
        assert isinstance(i, int), f"{i=} must be an integer"
        assert isinstance(j, int), f"{j=} must be an integer"
        assert 0 <= i < self.n_items, f"{i=} must be in [0, n_items)"
        assert 0 <= j < self.n_bins, f"{j=} must be in [0, n_bins)"

        return i * self.n_bins + j

    def y2global(self, j: int) -> int:
        """
        Map y-index to global x_hat-index.
        """
        assert isinstance(j, int), f"{j=} must be an integer"
        assert 0 <= j < self.n_bins, f"{j=} must be in [0, n_bins)"

        return self.n_bins * self.n_items + j

    def s2global(self, j: int, k: int) -> int:
        """
        Map s-index to global x_hat-index.
        """
        assert isinstance(j, int), f"{j=} must be an integer"
        assert isinstance(k, int), f"{k=} must be an integer"
        assert 0 <= j < self.n_bins, f"{j=} must be in [0, n_bins)"
        assert 0 <= k < self.N[j], f"{k=} must be in [0, N[j])"

        return self.n_bins * self.n_items + self.n_bins + sum(self.N[:j]) + k

    @property
    def Q_size(self) -> int:
        return (self.n_items + 1) * self.n_bins + sum(self.N)

    def _f_OF(self) -> np.ndarray[Any, np.dtype[np.float64]]:
        """
        Compute contribution of the objective function (OF).
        """
        Q1_OF = np.zeros((self.Q_size, self.Q_size))

        for j in range(self.n_bins):
            j_glob = self.y2global(j)
            Q1_OF[j_glob, j_glob] += 1

        return Q1_OF

    def _f1_IC(self) -> np.ndarray[Any, np.dtype[np.float64]]:
        """
        Compute contribution of the 1st inequality contrains (IC).
        """
        Q1_IC = np.zeros((self.Q_size, self.Q_size))

        for j in range(self.n_bins):
            for i in range(self.n_items):
                i_glob = self.x2global(i, j)
                Q1_IC[i_glob, i_glob] += self.weights[i] ** 2
                for i2 in range(i + 1, self.n_items):
                    j_glob = self.x2global(i2, j)
                    Q1_IC[i_glob, j_glob] += (
                        2 * self.lambda_IC[j] * self.weights[i] * self.weights[i2]
                    )

        return Q1_IC

    def _f2_IC(self) -> np.ndarray[Any, np.dtype[np.float64]]:
        """
        Compute contribution of the 2nd inequality contrains (IC).
        """
        Q2_IC = np.zeros((self.Q_size, self.Q_size))

        for j in range(self.n_bins):
            j_glob = self.y2global(j)
            Q2_IC[j_glob, j_glob] += self.lambda_IC[j] * self.bin_capacities[j] ** 2

        return Q2_IC

    def _f3_IC(self) -> np.ndarray[Any, np.dtype[np.float64]]:
        """
        Compute contribution of the 3rd inequality contrains (IC).
        """
        Q3_IC = np.zeros((self.Q_size, self.Q_size))

        for j in range(self.n_bins):
            for k in range(self.N[j]):
                i_glob = self.s2global(j, k)
                Q3_IC[i_glob, i_glob] += self.lambda_IC[j] * 2 ** (2 * k)
                for k2 in range(k + 1, self.N[j]):
                    j_glob = self.s2global(j, k2)
                    Q3_IC[i_glob, j_glob] += 2 * self.lambda_IC[j] * 2 ** (k + k2)

        return Q3_IC

    def _f4_IC(self) -> np.ndarray[Any, np.dtype[np.float64]]:
        """
        Compute contribution of the 4th inequality contrains (IC).
        """
        Q4_IC = np.zeros((self.Q_size, self.Q_size))

        for j in range(self.n_bins):
            for i in range(self.n_items):
                for k in range(self.N[j]):
                    i_glob = self.x2global(i, j)
                    j_glob = self.s2global(j, k)
                    Q4_IC[i_glob, j_glob] += (
                        2 * self.lambda_IC[j] * self.weights[i] * 2**k
                    )

        return Q4_IC

    def _f5_IC(self) -> np.ndarray[Any, np.dtype[np.float64]]:
        """
        Compute contribution of the 5th inequality contrains (IC).
        """
        Q5_IC = np.zeros((self.Q_size, self.Q_size))

        for j in range(self.n_bins):
            for i in range(self.n_items):
                i_glob = self.x2global(i, j)
                j_glob = self.y2global(j)
                Q5_IC[i_glob, j_glob] -= (
                    2 * self.lambda_IC[j] * self.bin_capacities[j] * self.weights[i]
                )

        return Q5_IC

    def _f6_IC(self) -> np.ndarray[Any, np.dtype[np.float64]]:
        """
        Compute contribution of the 6th inequality contrains (IC).
        """
        Q6_IC = np.zeros((self.Q_size, self.Q_size))

        for j in range(self.n_bins):
            for k in range(self.N[j]):
                i_glob = self.y2global(j)
                j_glob = self.s2global(j, k)
                Q6_IC[i_glob, j_glob] -= (
                    2 * self.lambda_IC[j] * self.bin_capacities[j] * 2**k
                )

        return Q6_IC

    def _f_EC(self) -> np.ndarray[Any, np.dtype[np.float64]]:
        """
        Compute contribution of the equality contrains (EC).
        """
        Q_EC = np.zeros((self.Q_size, self.Q_size))
        for i in range(self.n_items):
            for j in range(self.n_bins):
                i_glob = self.x2global(i, j)
                Q_EC[i_glob, i_glob] -= self.lambda_EC[i]
                for j2 in range(j + 1, self.n_bins):
                    j_glob = self.x2global(i, j2)
                    Q_EC[i_glob, j_glob] += 2 * self.lambda_EC[i]

        return Q_EC

    @property
    def Q(self) -> np.ndarray[Any, np.dtype[np.float64]]:
        """
        Compute Q-matrix.
        """
        Q = (
            self._f_OF()
            + self._f1_IC()
            + self._f2_IC()
            + self._f3_IC()
            + self._f4_IC()
            + self._f5_IC()
            + self._f6_IC()
            + self._f_EC()
        )

        return Q

    @property
    def Qxx(self) -> np.ndarray[Any, np.dtype[np.float64]]:
        """
        Compute Qxx-matrix.
        """
        x_size = self.n_items * self.n_bins
        Qxx = self.Q[:x_size, :x_size]

        return Qxx

    @property
    def Qxy(self) -> np.ndarray[Any, np.dtype[np.float64]]:
        """
        Compute Qxy-matrix.
        """
        x_size = self.n_items * self.n_bins
        s_size = sum(self.N)
        Qxy = self.Q[:x_size, x_size:-s_size]

        return Qxy

    @property
    def Qxs(self) -> np.ndarray[Any, np.dtype[np.float64]]:
        """
        Compute Qxs-matrix.
        """
        x_size = self.n_items * self.n_bins
        s_size = sum(self.N)
        Qxs = self.Q[:x_size, -s_size:]

        return Qxs

    @property
    def Qyy(self) -> np.ndarray[Any, np.dtype[np.float64]]:
        """
        Compute Qyy-matrix.
        """
        x_size = self.n_items * self.n_bins
        s_size = sum(self.N)
        Qyy = self.Q[x_size:-s_size, x_size:-s_size]

        return Qyy

    @property
    def Qys(self) -> np.ndarray[Any, np.dtype[np.float64]]:
        """
        Compute Qys-matrix.
        """
        x_size = self.n_items * self.n_bins
        s_size = sum(self.N)
        Qys = self.Q[x_size:-s_size, -s_size:]

        return Qys

    @property
    def Qss(self) -> np.ndarray[Any, np.dtype[np.float64]]:
        """
        Compute Qss-matrix.
        """
        s_size = sum(self.N)
        Qss = self.Q[-s_size:, -s_size:]

        return Qss
