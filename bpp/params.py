from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True, slots=True)
class BppParams:
    bin_capacity: np.ndarray
    item_weight: np.ndarray
    n_items: int = field(init=False)
    n_bins: int = field(init=False)
    N: np.ndarray = field(init=False)
    sum_N: int = field(init=False)
    shape_Q: tuple[int, int] = field(init=False)
    shape_Qxx: tuple[int, int] = field(init=False)
    shape_Qxy: tuple[int, int] = field(init=False)
    shape_Qxs: tuple[int, int] = field(init=False)
    shape_Qyx: tuple[int, int] = field(init=False)
    shape_Qyy: tuple[int, int] = field(init=False)
    shape_Qys: tuple[int, int] = field(init=False)
    shape_Qsx: tuple[int, int] = field(init=False)
    shape_Qsy: tuple[int, int] = field(init=False)
    shape_Qss: tuple[int, int] = field(init=False)

    def __post_init__(self):
        assert np.all(
            np.array(self.bin_capacity) > 0
        ), "Bin capacities should be positive"
        assert np.all(np.array(self.item_weight) > 0), "Item weights should be positive"
        assert np.all(
            np.floor(self.bin_capacity) == self.bin_capacity
        ), "Bin capacities should be integers"
        assert np.all(
            np.floor(self.item_weight) == self.item_weight
        ), "Item weights should be integers"

        object.__setattr__(
            self, "bin_capacity", np.array(self.bin_capacity).astype(int)
        )
        object.__setattr__(self, "item_weight", np.array(self.item_weight).astype(int))

        assert np.max(self.bin_capacity) >= np.max(
            self.item_weight
        ), "Maximum bin capacity should be greater than maximum item weight"
        assert np.sum(self.bin_capacity) >= np.sum(
            self.item_weight
        ), "Sum of bin capacities should be greater than sum of item weights"
        # check all capacities and weights are positive integers

        object.__setattr__(self, "n_items", len(self.item_weight))
        object.__setattr__(self, "n_bins", len(self.bin_capacity))

        N = (np.floor(np.log2(self.bin_capacity)) + 1).astype(int)
        object.__setattr__(self, "N", N)
        object.__setattr__(self, "sum_N", np.sum(self.N))

        size_Q = (self.n_items + 1) * self.n_bins + self.sum_N
        size_Qxx = self.n_items * self.n_bins
        size_Qyy = self.n_bins
        size_Qss = self.sum_N
        object.__setattr__(self, "shape_Q", (size_Q, size_Q))
        object.__setattr__(self, "shape_Qxx", (size_Qxx, size_Qxx))
        object.__setattr__(self, "shape_Qxy", (size_Qxx, size_Qyy))
        object.__setattr__(self, "shape_Qxs", (size_Qxx, size_Qss))
        object.__setattr__(self, "shape_Qyx", (size_Qyy, size_Qxx))
        object.__setattr__(self, "shape_Qyy", (size_Qyy, size_Qyy))
        object.__setattr__(self, "shape_Qys", (size_Qyy, size_Qss))
        object.__setattr__(self, "shape_Qsx", (size_Qss, size_Qxx))
        object.__setattr__(self, "shape_Qsy", (size_Qss, size_Qyy))
        object.__setattr__(self, "shape_Qss", (size_Qss, size_Qss))
