from __future__ import annotations

import numpy as np
from numpy.testing import assert_allclose

from questvar._correction import p_adjust


def _fdr_bh_ref(p, n):
    order = np.argsort(p)[::-1]
    inv_order = np.argsort(order)
    steps = n / np.arange(n, 0, -1)
    q = np.minimum(1.0, np.minimum.accumulate(steps * p[order]))
    return q[inv_order]


def _holm_ref(p, n):
    order = np.argsort(p)
    ranked = np.sort(p)
    adjusted = np.minimum(1.0, ranked * (n - np.arange(n)))
    for i in range(1, n):
        adjusted[i] = max(adjusted[i], adjusted[i - 1])
    result = np.empty_like(p)
    result[order] = adjusted
    return result


class TestMultipleTestingCorrection:
    def test_none_passthrough(self):
        p = np.array([0.001, 0.01, 0.1])
        assert_allclose(p_adjust(p, None), p)

    def test_bonferroni(self):
        p = np.array([0.001, 0.01, 0.03, 0.05, 0.5, 0.8])
        expected = np.minimum(p * len(p), 1.0)
        assert_allclose(p_adjust(p, "bonferroni"), expected)

    def test_holm(self):
        p = np.array([0.001, 0.01, 0.03, 0.05, 0.1, 0.2, 0.5, 0.8])
        expected = _holm_ref(p, len(p))
        assert_allclose(p_adjust(p, "holm"), expected)

    def test_fdr_bh(self):
        p = np.array([0.001, 0.01, 0.03, 0.05, 0.1, 0.2, 0.5, 0.8])
        expected = _fdr_bh_ref(p, len(p))
        assert_allclose(p_adjust(p, "fdr"), expected)
        assert_allclose(p_adjust(p, "fdr_bh"), expected)

    def test_bonferroni_r_reference(self):
        # R: p.adjust(c(0.001, 0.01, 0.03, 0.05, 0.1, 0.2, 0.5, 0.8), "bonferroni")
        # 0.008 0.080 0.240 0.400 0.800 1.000 1.000 1.000
        p = np.array([0.001, 0.01, 0.03, 0.05, 0.1, 0.2, 0.5, 0.8])
        r_ref = np.array([0.008, 0.080, 0.240, 0.400, 0.800, 1.000, 1.000, 1.000])
        assert_allclose(p_adjust(p, "bonferroni"), r_ref, atol=1e-6)

    def test_holm_r_reference(self):
        # R: p.adjust(c(0.001, 0.01, 0.03, 0.05, 0.1, 0.2, 0.5, 0.8), "holm")
        # 0.008 0.070 0.180 0.250 0.400 0.600 1.000 1.000
        p = np.array([0.001, 0.01, 0.03, 0.05, 0.1, 0.2, 0.5, 0.8])
        r_ref = np.array([0.008, 0.070, 0.180, 0.250, 0.400, 0.600, 1.000, 1.000])
        assert_allclose(p_adjust(p, "holm"), r_ref, atol=1e-6)

    def test_fdr_bh_r_reference(self):
        # R: p.adjust(c(0.001, 0.01, 0.03, 0.05, 0.1, 0.2, 0.5, 0.8), "BH")
        # 0.008, 0.040, 0.080, 0.100, 0.160, 0.267, 0.571, 0.800
        p = np.array([0.001, 0.01, 0.03, 0.05, 0.1, 0.2, 0.5, 0.8])
        r_ref = np.array([0.008, 0.04, 0.08, 0.1, 0.16, 0.266666666666667, 0.571428571428571, 0.8])
        assert_allclose(p_adjust(p, "fdr"), r_ref)
        assert_allclose(p_adjust(p, "fdr_bh"), r_ref)

    def test_hochberg_r_reference(self):
        # R: p.adjust(c(0.001, 0.01, 0.03, 0.05, 0.1, 0.2, 0.5, 0.8), "hochberg")
        # 0.008, 0.070, 0.180, 0.250, 0.400, 0.600, 0.800, 0.800
        p = np.array([0.001, 0.01, 0.03, 0.05, 0.1, 0.2, 0.5, 0.8])
        r_ref = np.array([0.008, 0.07, 0.18, 0.25, 0.4, 0.6, 0.8, 0.8])
        assert_allclose(p_adjust(p, "hochberg"), r_ref)

    def test_by_r_reference(self):
        # R: p.adjust(c(0.001, 0.01, 0.03, 0.05, 0.1, 0.2, 0.5, 0.8), "BY")
        # 0.02174286, 0.10871429, 0.21742857, 0.27178571, 0.43485714, 0.72476190, 1.0, 1.0
        p = np.array([0.001, 0.01, 0.03, 0.05, 0.1, 0.2, 0.5, 0.8])
        r_ref = np.array(
            [
                0.02174286,
                0.10871429,
                0.21742857,
                0.27178571,
                0.43485714,
                0.72476190,
                1.0,
                1.0,
            ]
        )
        assert_allclose(p_adjust(p, "BY"), r_ref, atol=1e-6)

    def test_bh_monotonicity(self):
        rng = np.random.default_rng(42)
        for _ in range(20):
            p = rng.uniform(0, 1, 100)
            adj = p_adjust(p, "fdr")
            order = np.argsort(p)
            assert np.all(np.diff(adj[order]) >= -1e-15)

    def test_bonferroni_monotonicity(self):
        p = np.array([0.001, 0.01, 0.03, 0.05, 0.1, 0.2, 0.5, 0.8])
        adj = p_adjust(p, "bonferroni")
        order = np.argsort(p)
        assert np.all(np.diff(adj[order]) >= -1e-15)

    def test_empty(self):
        p = np.array([])
        assert len(p_adjust(p, "bonferroni")) == 0

    def test_single(self):
        p = np.array([0.01])
        assert_allclose(p_adjust(p, "bonferroni"), [0.01])
        assert_allclose(p_adjust(p, "fdr"), [0.01])

    def test_all_one(self):
        p = np.ones(10)
        assert_allclose(p_adjust(p, "bonferroni"), np.ones(10))
        assert_allclose(p_adjust(p, "holm"), np.ones(10))
        assert_allclose(p_adjust(p, "fdr"), np.ones(10))

    def test_qvalue_all_one(self):
        p = np.ones(100)
        q = p_adjust(p, "qvalue")
        assert_allclose(q, np.ones(100))

    def test_qvalue_all_zero(self):
        p = np.full(100, 1e-15)
        q = p_adjust(p, "qvalue")
        assert np.all(q >= 0) and np.all(q <= 1)

    def test_qvalue_small_n(self):
        p = np.random.default_rng(42).uniform(0, 1, 50)
        q = p_adjust(p, "qvalue")
        assert len(q) == 50
        assert np.all(q >= 0) and np.all(q <= 1)

    def test_n_tests_override_all_methods(self):
        p = np.array([0.001, 0.01, 0.1])
        for method in ["bonferroni", "holm", "hochberg", "fdr", "BY", "qvalue"]:
            adj = p_adjust(p, method, n_tests=100)
            assert adj.shape == p.shape
            assert np.all(adj >= 0) and np.all(adj <= 1)

    def test_unknown_method(self):
        import pytest

        with pytest.raises(ValueError, match="Unknown correction"):
            p_adjust(np.array([0.01]), "invalid")
