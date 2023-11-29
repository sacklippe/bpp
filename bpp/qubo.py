import numpy as np

from bpp.params import BppParams


def formulate_Qxx(
    params: BppParams, lambda_eq: float, lambda_ineq: float
) -> np.ndarray:
    """Return the Qxx matrix."""
    assert lambda_eq > 0, "lamda_eq should be positive"
    assert lambda_ineq > 0, "lambda_ineq should be positive"
    Qxx = np.zeros(params.shape_Qxx)
    return Qxx


def formulate_Qyy(params: BppParams, lambda_ineq) -> np.ndarray:
    """Return the Qyy matrix."""
    assert lambda_ineq > 0, "lambda_ineq should be positive"
    Qyy = np.identity(params.shape_Qyy[0])
    return Qyy


def formulate_Qss(params: BppParams, lambda_ineq: float) -> np.ndarray:
    """Return the Qss matrix."""
    assert lambda_ineq > 0, "lambda_ineq should be positive"
    Qss = np.zeros(params.shape_Qss)
    return Qss


def formulate_Qxy(params: BppParams, lambda_ineq: float) -> np.ndarray:
    """Return the Qxy matrix."""
    assert lambda_ineq > 0, "lambda_ineq should be positive"
    Qxy = np.zeros(params.shape_Qxy)
    return Qxy


def formulate_Qxs(params: BppParams, lambda_ineq: float) -> np.ndarray:
    """Return the Qxs matrix."""
    assert lambda_ineq > 0, "lambda_ineq should be positive"
    Qxs = np.zeros(params.shape_Qxs)
    return Qxs


def formulate_Qys(params: BppParams, lambda_ineq: float) -> np.ndarray:
    """Return the Qys matrix."""
    assert lambda_ineq > 0, "lambda_ineq should be positive"
    Qys = np.zeros(params.shape_Qys)
    return Qys


def formulate_Q(params: BppParams, lambda_eq: float, lambda_ineq: float) -> np.ndarray:
    """Return the Q matrix."""
    Qxx = formulate_Qxx(params, lambda_eq, lambda_ineq)
    Qxy = formulate_Qxy(params, lambda_ineq)
    Qxs = formulate_Qxs(params, lambda_ineq)

    Qyx = np.zeros(params.shape_Qyx)
    Qyy = formulate_Qyy(params, lambda_ineq)
    Qys = formulate_Qys(params, lambda_ineq)

    Qsx = np.zeros(params.shape_Qsx)
    Qsy = np.zeros(params.shape_Qsy)
    Qss = formulate_Qss(params, lambda_ineq)

    Q = np.block([[Qxx, Qxy, Qxs], [Qyx, Qyy, Qys], [Qsx, Qsy, Qss]])
    return Q
