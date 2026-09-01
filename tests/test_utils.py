import lgdo
import numpy as np

from dspeed.utils import contains_nan, numba_defaults


def test_numba_defaults_loading():
    numba_defaults.cache = False
    numba_defaults.boundscheck = True


def test_contains_nan():
    # lengths around the 64-sample block boundary of the vectorized scan
    for dtype in (np.float32, np.float64):
        for n in (1, 63, 64, 65, 200):
            w = np.zeros(n, dtype=dtype)
            assert not contains_nan(w)
            for pos in (0, n // 2, n - 1):
                w = np.zeros(n, dtype=dtype)
                w[pos] = np.nan
                assert contains_nan(w)
    assert not contains_nan(np.zeros(0, dtype=np.float64))
    assert not contains_nan(np.array([np.inf, -np.inf]))


def isclose(lhs, rhs, rtol=1e-5, atol=1e-8, equal_nan=True):
    # an is close comparison for LGDO structures

    if isinstance(lhs, lgdo.Struct) and isinstance(rhs, lgdo.Struct):
        if set(lhs) != set(rhs) or lhs.attrs != rhs.attrs:
            return False

        for k in lhs:
            if not isclose(lhs[k], rhs[k], rtol=rtol, atol=atol, equal_nan=equal_nan):
                return False
        return True

    elif isinstance(lhs, lgdo.Array) and isinstance(rhs, lgdo.Array):
        if len(lhs) != len(rhs) or lhs.attrs != rhs.attrs:
            return False
        return np.all(np.isclose(lhs, rhs, rtol=rtol, atol=atol, equal_nan=equal_nan))

    elif isinstance(lhs, lgdo.VectorOfVectors) and isinstance(
        rhs, lgdo.VectorOfVectors
    ):
        if len(lhs) != len(rhs) or lhs.attrs != rhs.attrs:
            return False
        return lhs.cumulative_length == rhs.cumulative_length and np.all(
            np.isclose(lhs, rhs, rtol=rtol, atol=atol, equal_nan=equal_nan)
        )

    return False
