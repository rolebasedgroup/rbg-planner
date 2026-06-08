# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Ported from dynamo/planner/utils/perf_interpolation.py for RoleBasedGroup.

"""Performance interpolation for profiling data.

Ported from dynamo/components/src/dynamo/planner/utils/perf_interpolation.py
Supports JSON format for ConfigMap-mounted profiling results.
"""

import json
import logging
import os
from typing import Optional

import numpy as np

from rbg_planner.utils.exceptions import ProfilingDataNotFoundError

logger = logging.getLogger(__name__)


class PrefillInterpolator:
    """Interpolates throughput/gpu and TTFT for a given ISL using profiling data."""

    def __init__(self, profile_results_dir: Optional[str] = None, raw_data: Optional[dict] = None):
        if profile_results_dir:
            # Try NPZ first, then JSON (ConfigMap mount)
            npz_fn = os.path.join(
                profile_results_dir, "selected_prefill_interpolation", "raw_data.npz"
            )
            json_fn = os.path.join(profile_results_dir, "prefill_raw_data.json")

            if os.path.exists(npz_fn):
                with np.load(npz_fn) as data:
                    self.prefill_isl = data["prefill_isl"]
                    self.prefill_ttft = data["prefill_ttft"]
                    self.prefill_thpt_per_gpu = data["prefill_thpt_per_gpu"]
            elif os.path.exists(json_fn):
                with open(json_fn, "r") as f:
                    data = json.load(f)
                    self.prefill_isl = np.array(data["prefill_isl"])
                    self.prefill_ttft = np.array(data["prefill_ttft"])
                    self.prefill_thpt_per_gpu = np.array(data["prefill_thpt_per_gpu"])
            else:
                raise ProfilingDataNotFoundError(profile_results_dir)
        elif raw_data:
            self.prefill_isl = np.array(raw_data["prefill_isl"])
            self.prefill_ttft = np.array(raw_data["prefill_ttft"])
            self.prefill_thpt_per_gpu = np.array(raw_data["prefill_thpt_per_gpu"])
        else:
            raise ValueError("Either profile_results_dir or raw_data must be provided")

        self.min_isl = float(min(self.prefill_isl))
        self.max_isl = float(max(self.prefill_isl))

        import scipy.interpolate

        self.ttft_interpolator = scipy.interpolate.interp1d(
            self.prefill_isl, self.prefill_ttft, kind="cubic"
        )
        self.thpt_interpolator = scipy.interpolate.interp1d(
            self.prefill_isl, self.prefill_thpt_per_gpu, kind="cubic"
        )

    def interpolate_ttft(self, isl: float) -> float:
        isl = max(self.min_isl, min(isl, self.max_isl))
        return float(self.ttft_interpolator(isl))

    def interpolate_thpt_per_gpu(self, isl: float) -> float:
        isl = max(self.min_isl, min(isl, self.max_isl))
        return float(self.thpt_interpolator(isl))


class DecodeInterpolator:
    """Interpolates throughput/gpu and ITL for a given decode context length."""

    def __init__(
        self,
        profile_results_dir: Optional[str] = None,
        resolution: int = 100,
        raw_data: Optional[dict] = None,
    ):
        if profile_results_dir:
            npz_fn = os.path.join(
                profile_results_dir, "selected_decode_interpolation", "raw_data.npz"
            )
            json_fn = os.path.join(profile_results_dir, "decode_raw_data.json")

            if os.path.exists(npz_fn):
                with np.load(npz_fn) as data:
                    self.x_kv_usage = data["x_kv_usage"]
                    self.y_context_length = data["y_context_length"]
                    self.z_itl = data["z_itl"]
                    self.z_thpt_per_gpu = data["z_thpt_per_gpu"]
                    self.max_kv_tokens = data["max_kv_tokens"][0]
            elif os.path.exists(json_fn):
                with open(json_fn, "r") as f:
                    data = json.load(f)
                    self.x_kv_usage = np.array(data["x_kv_usage"])
                    self.y_context_length = np.array(data["y_context_length"])
                    self.z_itl = np.array(data["z_itl"])
                    self.z_thpt_per_gpu = np.array(data["z_thpt_per_gpu"])
                    self.max_kv_tokens = int(data["max_kv_tokens"])
            else:
                raise ProfilingDataNotFoundError(profile_results_dir)
        elif raw_data:
            self.x_kv_usage = np.array(raw_data["x_kv_usage"])
            self.y_context_length = np.array(raw_data["y_context_length"])
            self.z_itl = np.array(raw_data["z_itl"])
            self.z_thpt_per_gpu = np.array(raw_data["z_thpt_per_gpu"])
            self.max_kv_tokens = int(raw_data["max_kv_tokens"])
        else:
            raise ValueError("Either profile_results_dir or raw_data must be provided")

        self.resolution = resolution
        self.xi = np.linspace(0, 1, resolution)
        self.yi = np.linspace(0, float(max(self.y_context_length)), resolution)
        self.X, self.Y = np.meshgrid(self.xi, self.yi)

        import scipy.interpolate

        self.itl_interpolator = scipy.interpolate.griddata(
            (self.x_kv_usage, self.y_context_length),
            self.z_itl,
            (self.X, self.Y),
            method="cubic",
        )
        nan_mask = np.isnan(self.itl_interpolator)
        if np.any(nan_mask):
            itl_nearest = scipy.interpolate.griddata(
                (self.x_kv_usage, self.y_context_length),
                self.z_itl,
                (self.X, self.Y),
                method="nearest",
            )
            self.itl_interpolator[nan_mask] = itl_nearest[nan_mask]

        self.thpt_interpolator = scipy.interpolate.griddata(
            (self.x_kv_usage, self.y_context_length),
            self.z_thpt_per_gpu,
            (self.X, self.Y),
            method="cubic",
        )
        nan_mask = np.isnan(self.thpt_interpolator)
        if np.any(nan_mask):
            thpt_nearest = scipy.interpolate.griddata(
                (self.x_kv_usage, self.y_context_length),
                self.z_thpt_per_gpu,
                (self.X, self.Y),
                method="nearest",
            )
            self.thpt_interpolator[nan_mask] = thpt_nearest[nan_mask]

    def compute_idx(self, concurrency: float, context_length: float) -> tuple[int, int]:
        kv_usage = concurrency * context_length / self.max_kv_tokens
        ix = int(
            np.clip(
                np.round((kv_usage - self.xi[0]) / (self.xi[1] - self.xi[0])),
                0,
                self.resolution - 1,
            )
        )
        iy = int(
            np.clip(
                np.round((context_length - self.yi[0]) / (self.yi[1] - self.yi[0])),
                0,
                self.resolution - 1,
            )
        )
        return ix, iy

    def interpolate_itl(self, concurrency: float, context_length: float) -> float:
        ix, iy = self.compute_idx(concurrency, context_length)
        return float(self.itl_interpolator[iy, ix])

    def interpolate_thpt_per_gpu(self, concurrency: float, context_length: float) -> float:
        ix, iy = self.compute_idx(concurrency, context_length)
        return float(self.thpt_interpolator[iy, ix])

    def find_best_throughput_per_gpu(
        self, itl: float, context_length: float
    ) -> tuple[float, float, float]:
        """Find max throughput/gpu that achieves ITL <= target."""
        iy = int(
            np.clip(
                np.round((context_length - self.yi[0]) / (self.yi[1] - self.yi[0])),
                0,
                self.resolution - 1,
            )
        )
        iy = max(0, min(iy, self.resolution - 1))

        for ix in range(self.resolution - 1, -1, -1):
            if self.itl_interpolator[iy, ix] <= itl:
                return (
                    float(self.thpt_interpolator[iy, ix]),
                    float(self.itl_interpolator[iy, ix]),
                    float(self.xi[ix]),
                )
        return (
            float(self.thpt_interpolator[iy, 0]),
            float(self.itl_interpolator[iy, 0]),
            float(self.xi[0]),
        )
