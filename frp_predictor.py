from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


SEQUENCE_LENGTH = 5
DATA_FILE = "frp_fully_extended.csv"
MODEL_FILE = "frp_rnn_model_optimized.h5"
RESULTS_FILE = "frp_rnn_model_results.xlsx"


class PredictorLoadError(RuntimeError):
    """Raised when the trained model or its runtime dependencies cannot be loaded."""


@dataclass
class PredictionResult:
    nano_silica: float
    tensile_mpa: float
    flexural_mpa: float
    dataset_min: float
    dataset_max: float
    nearest_sample_nano: float
    nearest_sample_tensile: float
    nearest_sample_flexural: float


class FRPPredictor:
    def __init__(self, base_dir: str | Path | None = None) -> None:
        self.base_dir = Path(base_dir or Path(__file__).resolve().parent)
        self.data_path = self.base_dir / DATA_FILE
        self.model_path = self.base_dir / MODEL_FILE
        self.results_path = self.base_dir / RESULTS_FILE
        self._dataset: pd.DataFrame | None = None
        self._results_cache: dict[str, pd.DataFrame] = {}
        self._scaler_x: StandardScaler | None = None
        self._scaler_y: StandardScaler | None = None
        self._model: Any = None

    @property
    def dataset(self) -> pd.DataFrame:
        if self._dataset is None:
            self._dataset = pd.read_csv(self.data_path)
        return self._dataset

    def get_feature_bounds(self) -> tuple[float, float]:
        series = self.dataset["Nano Silica %"]
        return float(series.min()), float(series.max())

    def get_results_sheet(self, sheet_name: str) -> pd.DataFrame:
        if sheet_name not in self._results_cache:
            self._results_cache[sheet_name] = pd.read_excel(self.results_path, sheet_name=sheet_name)
        return self._results_cache[sheet_name].copy()

    def _prepare_scalers(self) -> None:
        if self._scaler_x is not None and self._scaler_y is not None:
            return

        df = self.dataset
        x_raw = df[["Nano Silica %"]].values
        y_raw = df[["Tensile Stress (MPa)", "Flexural Stress (MPa)"]].values

        x_seq, y_seq = [], []
        for idx in range(len(x_raw) - SEQUENCE_LENGTH):
            x_seq.append(x_raw[idx : idx + SEQUENCE_LENGTH])
            y_seq.append(y_raw[idx + SEQUENCE_LENGTH])

        x_seq_np = np.array(x_seq)
        y_seq_np = np.array(y_seq)

        scaler_x = StandardScaler()
        scaler_y = StandardScaler()

        scaler_x.fit(x_seq_np.reshape(-1, 1))
        scaler_y.fit(y_seq_np)

        self._scaler_x = scaler_x
        self._scaler_y = scaler_y

    def _load_model(self) -> None:
        if self._model is not None:
            return

        try:
            from tensorflow.keras.models import load_model
        except ModuleNotFoundError as exc:
            raise PredictorLoadError(
                "TensorFlow is not installed. Install dependencies before running predictions."
            ) from exc

        if not self.model_path.exists():
            raise PredictorLoadError(f"Model file not found: {self.model_path}")

        self._model = load_model(self.model_path, compile=False)

    def _nearest_row(self, nano_silica: float) -> pd.Series:
        return self.dataset.iloc[
            (self.dataset["Nano Silica %"] - nano_silica).abs().argsort().iloc[0]
        ]

    def _build_prediction_result(
        self,
        nano_silica: float,
        tensile_mpa: float,
        flexural_mpa: float,
    ) -> PredictionResult:
        dataset_min, dataset_max = self.get_feature_bounds()
        nearest_row = self._nearest_row(nano_silica)
        return PredictionResult(
            nano_silica=float(nano_silica),
            tensile_mpa=float(tensile_mpa),
            flexural_mpa=float(flexural_mpa),
            dataset_min=dataset_min,
            dataset_max=dataset_max,
            nearest_sample_nano=float(nearest_row["Nano Silica %"]),
            nearest_sample_tensile=float(nearest_row["Tensile Stress (MPa)"]),
            nearest_sample_flexural=float(nearest_row["Flexural Stress (MPa)"]),
        )

    def predict(self, nano_silica: float) -> PredictionResult:
        self._prepare_scalers()
        self._load_model()

        dataset_min, dataset_max = self.get_feature_bounds()
        clamped_value = float(np.clip(nano_silica, dataset_min, dataset_max))

        sequence = np.full((1, SEQUENCE_LENGTH, 1), clamped_value, dtype=np.float32)
        scaled_sequence = self._scaler_x.transform(sequence.reshape(-1, 1)).reshape(sequence.shape)

        prediction_scaled = self._model.predict(scaled_sequence, verbose=0)
        prediction = self._scaler_y.inverse_transform(prediction_scaled)[0]
        return self._build_prediction_result(clamped_value, prediction[0], prediction[1])

    def predict_many(self, nano_silica_values: list[float] | np.ndarray) -> list[PredictionResult]:
        self._prepare_scalers()
        self._load_model()

        dataset_min, dataset_max = self.get_feature_bounds()
        values = np.asarray(nano_silica_values, dtype=np.float32)
        clamped = np.clip(values, dataset_min, dataset_max)

        sequences = np.repeat(clamped[:, None, None], SEQUENCE_LENGTH, axis=1)
        scaled_sequences = self._scaler_x.transform(sequences.reshape(-1, 1)).reshape(sequences.shape)
        predictions_scaled = self._model.predict(scaled_sequences, verbose=0)
        predictions = self._scaler_y.inverse_transform(predictions_scaled)

        return [
            self._build_prediction_result(float(nano), float(pred[0]), float(pred[1]))
            for nano, pred in zip(clamped, predictions)
        ]
