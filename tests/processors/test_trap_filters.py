import numpy as np
import pytest

from dspeed.errors import DSPFatal
from dspeed.processors import asym_trap_filter, trap_filter, trap_norm, trap_pickoff


def test_trap_filter(compare_numba_vs_python):
    w_in = np.concatenate([np.zeros(100), np.ones(100)])
    rise = 10
    flat = 20
    exp_out = np.concatenate(
        [
            np.zeros(99),
            np.arange(10),
            np.full(20, 10),
            np.arange(10, 0, -1),
            np.zeros(61),
        ]
    )

    result = compare_numba_vs_python(trap_filter, w_in, rise, flat)
    assert np.allclose(result, exp_out)

    w_in[3] = np.nan
    assert np.isnan(compare_numba_vs_python(trap_filter, w_in, rise, flat)).all()

    w_in = np.concatenate([np.zeros(100), np.ones(100)])
    w_out = np.zeros_like(w_in)
    with pytest.raises(DSPFatal):
        compare_numba_vs_python(trap_filter, w_in, -1, 10, w_out)
    with pytest.raises(DSPFatal):
        compare_numba_vs_python(trap_filter, w_in, 10, -1, w_out)
    with pytest.raises(DSPFatal):
        compare_numba_vs_python(trap_filter, w_in, 100, 50, w_out)


def test_trap_norm(compare_numba_vs_python):
    w_in = np.concatenate([np.zeros(100), np.ones(100)])
    rise = 10
    flat = 20
    exp_out = np.concatenate(
        [
            np.zeros(99),
            np.arange(10) / 10,
            np.ones(20),
            np.arange(10, 0, -1) / 10,
            np.zeros(61),
        ]
    )

    result = compare_numba_vs_python(trap_norm, w_in, rise, flat)
    assert np.allclose(result, exp_out)

    w_in[3] = np.nan
    assert np.isnan(compare_numba_vs_python(trap_norm, w_in, rise, flat)).all()

    w_in = np.concatenate([np.zeros(100), np.ones(100)])
    w_out = np.zeros_like(w_in)
    with pytest.raises(DSPFatal):
        compare_numba_vs_python(trap_norm, w_in, -1, 10, w_out)
    with pytest.raises(DSPFatal):
        compare_numba_vs_python(trap_norm, w_in, 10, -1, w_out)
    with pytest.raises(DSPFatal):
        compare_numba_vs_python(trap_norm, w_in, 100, 50, w_out)


def test_asym_trap_filter(compare_numba_vs_python):
    w_in = np.concatenate([np.zeros(100), np.ones(100)])
    rise = 10
    flat = 20
    fall = 30
    exp_out = np.concatenate(
        [
            np.zeros(99),
            np.arange(10) / 10,
            np.ones(20),
            np.arange(30, 0, -1) / 30,
            np.zeros(41),
        ]
    )

    result = compare_numba_vs_python(asym_trap_filter, w_in, rise, flat, fall)
    assert np.allclose(result, exp_out)

    w_in[3] = np.nan
    assert np.isnan(
        compare_numba_vs_python(asym_trap_filter, w_in, rise, flat, fall)
    ).all()

    w_in = np.concatenate([np.zeros(100), np.ones(100)])
    w_out = np.zeros_like(w_in)
    with pytest.raises(DSPFatal):
        compare_numba_vs_python(asym_trap_filter, w_in, -1, 10, 5, w_out)
    with pytest.raises(DSPFatal):
        compare_numba_vs_python(asym_trap_filter, w_in, 10, -1, 5, w_out)
    with pytest.raises(DSPFatal):
        compare_numba_vs_python(asym_trap_filter, w_in, 10, 10, -1, w_out)
    with pytest.raises(DSPFatal):
        compare_numba_vs_python(asym_trap_filter, w_in, 200, 50, 10, w_out)


def test_trap_pickoff(compare_numba_vs_python):
    w_in = np.concatenate([np.zeros(100), np.ones(100)])
    rise = 10
    flat = 20
    filtered = compare_numba_vs_python(trap_norm, w_in, rise, flat)

    # spot check baseline, rise, flat top, fall, and baseline
    # also spot check fractional interpolated pickoff times
    for i in [80, 100, 105, 110, 120, 130, 135, 140, 150]:
        assert np.isclose(
            compare_numba_vs_python(trap_pickoff, w_in, rise, flat, i), filtered[i]
        )
        assert np.isclose(
            compare_numba_vs_python(trap_pickoff, w_in, rise, flat, i + 0.3),
            filtered[i] * 0.7 + filtered[i + 1] * 0.3,
        )

    assert np.isnan(compare_numba_vs_python(trap_pickoff, w_in, rise, flat, 10))
    assert np.isnan(compare_numba_vs_python(trap_pickoff, w_in, rise, flat, 220))

    w_in[3] = np.nan
    assert np.isnan(compare_numba_vs_python(trap_pickoff, w_in, rise, flat, 125))

    w_in = np.concatenate([np.zeros(100), np.ones(100)])
    w_out = np.zeros((1), dtype=w_in.dtype)
    with pytest.raises(DSPFatal):
        compare_numba_vs_python(trap_pickoff, w_in, -1, 10, 100, w_out)
    with pytest.raises(DSPFatal):
        compare_numba_vs_python(trap_pickoff, w_in, 10, -1, 100, w_out)
    with pytest.raises(DSPFatal):
        compare_numba_vs_python(trap_pickoff, w_in, 100, 50, 100, w_out)
