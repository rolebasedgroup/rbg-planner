"""Tests for perf_interpolation PrefillInterpolator with varying data point counts."""

import numpy as np
import pytest

from rbg_planner.utils.perf_interpolation import PrefillInterpolator


class TestPrefillInterpolatorFewPoints:
    """interp1d with kind='cubic' requires 4+ points; verify graceful fallback."""

    @pytest.mark.parametrize("n_points", [1, 2, 3, 4, 5])
    def test_constructs_without_error(self, n_points):
        isl = np.linspace(128, 512, n_points)
        ttft = isl * 0.2
        thpt = np.full(n_points, 5.0)

        interpolator = PrefillInterpolator(
            raw_data={
                "prefill_isl": isl,
                "prefill_ttft": ttft,
                "prefill_thpt_per_gpu": thpt,
            }
        )

        # Should not raise
        result = interpolator.interpolate_ttft(256.0)
        assert isinstance(result, float)

    def test_three_points_matches_configmap_scenario(self):
        """Reproduce the exact CrashLoopBackOff scenario from the cluster."""
        raw_data = {
            "prefill_isl": [128, 256, 512],
            "prefill_ttft": [25611.24, 51206.99, 102407.44],
            "prefill_thpt_per_gpu": [5.0, 5.0, 5.0],
        }

        interpolator = PrefillInterpolator(raw_data=raw_data)

        assert interpolator.min_isl == 128.0
        assert interpolator.max_isl == 512.0

        ttft = interpolator.interpolate_ttft(256.0)
        assert abs(ttft - 51206.99) < 1.0

        thpt = interpolator.interpolate_thpt_per_gpu(300.0)
        assert thpt == 5.0

    def test_clamps_to_range(self):
        raw_data = {
            "prefill_isl": [128, 256, 512],
            "prefill_ttft": [25611.24, 51206.99, 102407.44],
            "prefill_thpt_per_gpu": [5.0, 5.0, 5.0],
        }

        interpolator = PrefillInterpolator(raw_data=raw_data)

        below = interpolator.interpolate_ttft(0.0)
        above = interpolator.interpolate_ttft(9999.0)
        assert abs(below - 25611.24) < 1.0
        assert abs(above - 102407.44) < 1.0
