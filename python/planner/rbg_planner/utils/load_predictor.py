"""Load predictors for forecasting request patterns.

Ported from dynamo/components/src/dynamo/planner/utils/load_predictor.py
Supports Constant, ARIMA, and optionally Prophet predictors.
"""

import logging
import math
import warnings
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)

# Suppress noisy warnings from statistical libraries
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


class BasePredictor(ABC):
    """Base class for all load predictors."""

    def __init__(self, minimum_data_points: int = 5):
        self.minimum_data_points = minimum_data_points
        self.data_buffer: list[float] = []

    def add_data_point(self, value: float):
        if math.isnan(value):
            value = 0.0
        if len(self.data_buffer) == 0 and value == 0:
            return
        self.data_buffer.append(value)

    def get_last_value(self) -> float:
        if not self.data_buffer:
            return 0.0
        return self.data_buffer[-1]

    @abstractmethod
    def predict_next(self) -> float:
        pass


class ConstantPredictor(BasePredictor):
    """Assumes next load equals current load."""

    def __init__(self, **kwargs):
        super().__init__(minimum_data_points=1)

    def predict_next(self) -> float:
        return self.get_last_value()


class ARIMAPredictor(BasePredictor):
    """Auto ARIMA model from pmdarima."""

    def __init__(self, window_size: int = 100, minimum_data_points: int = 5, **kwargs):
        super().__init__(minimum_data_points=minimum_data_points)
        self.window_size = window_size
        self.model = None

    def add_data_point(self, value: float):
        super().add_data_point(value)
        if len(self.data_buffer) > self.window_size:
            self.data_buffer = self.data_buffer[-self.window_size:]

    def predict_next(self) -> float:
        if len(self.data_buffer) < self.minimum_data_points:
            return self.get_last_value()

        if len(set(self.data_buffer)) == 1:
            return self.data_buffer[0]

        try:
            import pmdarima

            self.model = pmdarima.auto_arima(
                self.data_buffer,
                suppress_warnings=True,
                error_action="ignore",
            )
            forecast = self.model.predict(n_periods=1)
            return float(forecast[0])
        except Exception as e:
            logger.warning(f"ARIMA prediction failed: {e}, using last value")
            return self.get_last_value()


class ProphetPredictor(BasePredictor):
    """Facebook Prophet time-series forecasting (optional dependency)."""

    def __init__(self, window_size: int = 100, step_size: int = 3600, minimum_data_points: int = 5, **kwargs):
        super().__init__(minimum_data_points=minimum_data_points)
        self.window_size = window_size
        self.curr_step = 0
        self.data_buffer: list[dict] = []  # type: ignore

        from datetime import datetime
        self.start_date = datetime(2024, 1, 1)

    def add_data_point(self, value: float):
        from datetime import timedelta

        value = 0.0 if math.isnan(value) else value
        if len(self.data_buffer) == 0 and value == 0:
            return

        timestamp = self.start_date + timedelta(seconds=self.curr_step)
        self.data_buffer.append({"ds": timestamp, "y": value})
        self.curr_step += 1

        if len(self.data_buffer) > self.window_size:
            self.data_buffer = self.data_buffer[-self.window_size:]

    def get_last_value(self) -> float:
        if not self.data_buffer:
            return 0.0
        return self.data_buffer[-1]["y"]

    def predict_next(self) -> float:
        if len(self.data_buffer) < self.minimum_data_points:
            return self.get_last_value()

        try:
            from datetime import timedelta

            import pandas as pd
            from prophet import Prophet

            df = pd.DataFrame(self.data_buffer)
            model = Prophet()
            model.fit(df)

            next_timestamp = self.start_date + timedelta(seconds=self.curr_step)
            future_df = pd.DataFrame({"ds": [next_timestamp]})
            forecast = model.predict(future_df)
            return float(forecast["yhat"].iloc[0])
        except ImportError:
            logger.warning("Prophet not installed, falling back to last value")
            return self.get_last_value()
        except Exception as e:
            logger.warning(f"Prophet prediction failed: {e}, using last value")
            return self.get_last_value()


LOAD_PREDICTORS = {
    "constant": ConstantPredictor,
    "arima": ARIMAPredictor,
    "prophet": ProphetPredictor,
}
