"""PyTorch MLP challenger model and artifact utilities."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.base import clone
from sklearn.pipeline import Pipeline

from ml_risk_control.evaluation.metrics import evaluate_binary_classifier
from ml_risk_control.features.build import (
    DEFAULT_MISSING_INDICATOR_COLUMNS,
    FeatureSchema,
    build_feature_schema,
    create_preprocessing_pipeline,
)


def _load_torch_modules() -> tuple[Any, Any, Any, Any]:
    """Load torch lazily so package import survives optional runtime issues."""
    try:
        import torch
        from torch import nn
        from torch.utils.data import DataLoader, TensorDataset
    except Exception as error:  # pragma: no cover - depends on runtime native library setup
        msg = (
            "torch is not available or its native library could not be loaded. "
            "Ensure the package is installed before using the PyTorch challenger."
        )
        raise ImportError(msg) from error
    return torch, nn, DataLoader, TensorDataset


class _TorchMLPBinaryClassifier:
    """Factory wrapper that constructs the underlying nn.Module lazily."""

    def __init__(self, input_dim: int, hidden_dims: tuple[int, ...], dropout: float) -> None:
        torch, nn, _, _ = _load_torch_modules()
        del torch

        layers: list[Any] = []
        current_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(current_dim, hidden_dim))
            layers.append(nn.ReLU())
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            current_dim = hidden_dim
        layers.append(nn.Linear(current_dim, 1))
        self.module = nn.Sequential(*layers)


@dataclass(frozen=True)
class TorchMLPConfig:
    """Configuration for the PyTorch challenger classifier."""

    hidden_dims: tuple[int, ...] = (64, 32)
    dropout: float = 0.10
    learning_rate: float = 1e-3
    batch_size: int = 512
    max_epochs: int = 50
    patience: int = 8
    min_delta: float = 1e-4
    weight_decay: float = 1e-4
    positive_class_weight_strategy: str = "auto_from_train_ratio"
    positive_class_weight_value: float | None = None
    random_state: int = 42
    device: str = "cpu"

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable configuration payload."""
        return asdict(self)


class TorchMLPCreditRiskModel:
    """Reusable PyTorch challenger with shared preprocessing and artifact helpers."""

    def __init__(
        self,
        *,
        id_column: str,
        target_column: str,
        model_version: str = "0.1.0",
        schema_version: str = "1.0.0",
        config: TorchMLPConfig | None = None,
        feature_schema: FeatureSchema | None = None,
        preprocessing_pipeline: Pipeline | None = None,
        missing_indicator_columns: tuple[str, ...] = DEFAULT_MISSING_INDICATOR_COLUMNS,
    ) -> None:
        self.id_column = id_column
        self.target_column = target_column
        self.model_version = model_version
        self.schema_version = schema_version
        self.config = config or TorchMLPConfig()
        self.missing_indicator_columns = missing_indicator_columns
        self.feature_schema = feature_schema or build_feature_schema(
            id_column=id_column,
            target_column=target_column,
            missing_indicator_columns=missing_indicator_columns,
            schema_version=schema_version,
        )
        self.preprocessing_pipeline = preprocessing_pipeline or create_preprocessing_pipeline(
            feature_columns=self.feature_schema.raw_feature_columns,
            missing_indicator_columns=missing_indicator_columns,
        )
        self.preprocessing_pipeline_: Pipeline | None = None
        self.classifier_: Any | None = None
        self.training_summary_: dict[str, Any] | None = None
        self.training_history_: dict[str, Any] | None = None
        self.transformed_feature_names_: tuple[str, ...] | None = None
        self.input_dim_: int | None = None

    def fit(
        self,
        dataframe: pd.DataFrame,
        *,
        eval_dataframe: pd.DataFrame | None = None,
        verbose: bool = False,
    ) -> TorchMLPCreditRiskModel:
        """Fit the shared preprocessing pipeline and PyTorch MLP classifier."""
        torch, nn, DataLoader, TensorDataset = _load_torch_modules()

        X_train, y_train = self._split_features_and_target(dataframe)
        self.preprocessing_pipeline_ = clone(self.preprocessing_pipeline)
        train_matrix = self._fit_transform_features(self.preprocessing_pipeline_, X_train)
        self.input_dim_ = int(train_matrix.shape[1])

        eval_matrix = None
        y_eval = None
        eval_row_count = 0
        if eval_dataframe is not None:
            X_eval, y_eval = self._split_features_and_target(eval_dataframe)
            eval_matrix = self._transform_features(self.preprocessing_pipeline_, X_eval)
            eval_row_count = int(len(eval_dataframe))

        torch.manual_seed(self.config.random_state)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.config.random_state)

        device = self._resolve_device(torch)
        network = self._build_network(self.input_dim_, nn).to(device)
        optimizer = torch.optim.Adam(
            network.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )

        positive_class_weight = self._resolve_positive_class_weight(y_train)
        if positive_class_weight is None:
            criterion = nn.BCEWithLogitsLoss()
        else:
            pos_weight_tensor = torch.tensor(
                positive_class_weight,
                dtype=torch.float32,
                device=device,
            )
            criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)

        train_tensor = torch.tensor(train_matrix.to_numpy(), dtype=torch.float32)
        target_tensor = torch.tensor(y_train.to_numpy(), dtype=torch.float32)
        train_loader = DataLoader(
            TensorDataset(train_tensor, target_tensor),
            batch_size=self.config.batch_size,
            shuffle=True,
        )

        eval_features_tensor = None
        eval_target_tensor = None
        if eval_matrix is not None and y_eval is not None:
            eval_features_tensor = torch.tensor(
                eval_matrix.to_numpy(),
                dtype=torch.float32,
                device=device,
            )
            eval_target_tensor = torch.tensor(
                y_eval.to_numpy(),
                dtype=torch.float32,
                device=device,
            )

        epoch_history: list[dict[str, Any]] = []
        best_validation_loss = float("inf")
        best_epoch = 0
        best_state_dict = deepcopy(network.state_dict())
        best_validation_metrics: dict[str, Any] | None = None
        patience_counter = 0

        for epoch in range(1, self.config.max_epochs + 1):
            network.train()
            batch_losses: list[float] = []
            for batch_features, batch_target in train_loader:
                batch_features = batch_features.to(device)
                batch_target = batch_target.to(device)

                optimizer.zero_grad()
                logits = network(batch_features).squeeze(-1)
                loss = criterion(logits, batch_target)
                loss.backward()
                optimizer.step()
                batch_losses.append(float(loss.detach().cpu().item()))

            train_loss = float(sum(batch_losses) / len(batch_losses))
            validation_loss = train_loss
            validation_metrics = None

            if eval_features_tensor is not None and eval_target_tensor is not None:
                network.eval()
                with torch.no_grad():
                    validation_logits = network(eval_features_tensor).squeeze(-1)
                    validation_loss = float(
                        criterion(validation_logits, eval_target_tensor).detach().cpu().item()
                    )
                    validation_probabilities = torch.sigmoid(validation_logits).detach().cpu().numpy()
                validation_metrics = evaluate_binary_classifier(
                    y_eval,
                    validation_probabilities,
                    threshold=0.5,
                )

            epoch_record = {
                "epoch": epoch,
                "train_loss": train_loss,
                "validation_loss": validation_loss,
            }
            if validation_metrics is not None:
                epoch_record["validation_average_precision"] = validation_metrics[
                    "average_precision"
                ]
                epoch_record["validation_roc_auc"] = validation_metrics["roc_auc"]
                epoch_record["validation_brier_score"] = validation_metrics["brier_score"]
            epoch_history.append(epoch_record)

            if verbose:
                print(
                    f"[epoch {epoch:03d}] train_loss={train_loss:.6f} "
                    f"validation_loss={validation_loss:.6f}"
                )

            if validation_loss < best_validation_loss - self.config.min_delta:
                best_validation_loss = validation_loss
                best_epoch = epoch
                best_state_dict = deepcopy(network.state_dict())
                best_validation_metrics = validation_metrics
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= self.config.patience:
                    break

        network.load_state_dict(best_state_dict)
        network.eval()
        self.classifier_ = network
        self.training_history_ = {
            "epochs_completed": int(len(epoch_history)),
            "best_epoch": int(best_epoch),
            "best_validation_loss": float(best_validation_loss),
            "history": epoch_history,
        }
        self.training_summary_ = {
            "row_count": int(len(dataframe)),
            "positive_rate": float((y_train == 1).mean()),
            "eval_row_count": eval_row_count,
            "trained_at_utc": datetime.now(UTC).isoformat(),
            "classifier_class": network.__class__.__name__,
            "best_epoch": int(best_epoch),
            "best_validation_loss": float(best_validation_loss),
            "best_validation_average_precision": (
                None
                if best_validation_metrics is None
                else float(best_validation_metrics["average_precision"])
            ),
            "effective_positive_class_weight": positive_class_weight,
            "device": str(device),
        }
        return self

    def predict_proba(self, dataframe: pd.DataFrame) -> pd.Series:
        """Return positive-class probabilities indexed like the input dataframe."""
        torch, _, _, _ = _load_torch_modules()
        classifier = self._require_fitted_classifier()
        preprocessing_pipeline = self._require_fitted_preprocessing()
        feature_frame = self._select_feature_frame(dataframe)
        model_matrix = self._transform_features(preprocessing_pipeline, feature_frame)
        model_tensor = torch.tensor(model_matrix.to_numpy(), dtype=torch.float32)
        device = next(classifier.parameters()).device
        classifier.eval()
        with torch.no_grad():
            logits = classifier(model_tensor.to(device)).squeeze(-1)
            probabilities = torch.sigmoid(logits).detach().cpu().numpy()
        return pd.Series(probabilities, index=dataframe.index, name="predicted_probability")

    def predict(
        self,
        dataframe: pd.DataFrame,
        *,
        threshold: float = 0.5,
    ) -> pd.Series:
        """Return binary predictions at the provided probability threshold."""
        probabilities = self.predict_proba(dataframe)
        predictions = probabilities.ge(threshold).astype(int)
        predictions.name = "predicted_label"
        return predictions

    def score_records(
        self,
        dataframe: pd.DataFrame,
        *,
        threshold: float = 0.5,
        include_identifier: bool = True,
    ) -> pd.DataFrame:
        """Return scored records with optional identifier passthrough."""
        scored = pd.DataFrame(index=dataframe.index)
        if include_identifier and self.id_column in dataframe.columns:
            scored[self.id_column] = dataframe[self.id_column].values
        scored["predicted_probability"] = self.predict_proba(dataframe)
        scored["predicted_label"] = scored["predicted_probability"].ge(threshold).astype(int)
        return scored

    def build_artifact_metadata(self) -> dict[str, Any]:
        """Return model metadata for artifact persistence and reporting."""
        return {
            "model_name": "torch_mlp_challenger",
            "model_version": self.model_version,
            "schema_version": self.feature_schema.schema_version,
            "id_column": self.id_column,
            "target_column": self.target_column,
            "raw_feature_columns": list(self.feature_schema.raw_feature_columns),
            "model_input_features": list(self.feature_schema.model_input_features),
            "transformed_feature_names": list(self._require_transformed_feature_names()),
            "missing_indicator_columns": list(self.missing_indicator_columns),
            "classifier_config": self.config.to_dict(),
            "training_summary": self.training_summary_,
            "training_history": self.training_history_,
            "input_dim": self.input_dim_,
        }

    def save(self, path: Path) -> Path:
        """Persist the fitted PyTorch artifact bundle."""
        torch, _, _, _ = _load_torch_modules()
        preprocessing_pipeline = self._require_fitted_preprocessing()
        classifier = self._require_fitted_classifier()
        path.parent.mkdir(parents=True, exist_ok=True)
        state_dict = {
            key: value.detach().cpu()
            for key, value in classifier.state_dict().items()
        }
        torch.save(
            {
                "state_dict": state_dict,
                "config": self.config.to_dict(),
                "feature_schema": self.feature_schema.to_dict(),
                "model_version": self.model_version,
                "schema_version": self.schema_version,
                "missing_indicator_columns": self.missing_indicator_columns,
                "training_summary": self.training_summary_,
                "training_history": self.training_history_,
                "transformed_feature_names": self._require_transformed_feature_names(),
                "input_dim": self._require_input_dim(),
                "preprocessing_pipeline": preprocessing_pipeline,
                "artifact_metadata": self.build_artifact_metadata(),
            },
            path,
        )
        return path

    @classmethod
    def load(cls, path: Path) -> TorchMLPCreditRiskModel:
        """Load a persisted PyTorch artifact bundle from disk."""
        torch, nn, _, _ = _load_torch_modules()
        bundle = torch.load(path, map_location="cpu", weights_only=False)
        feature_schema = FeatureSchema(**bundle["feature_schema"])
        config = TorchMLPConfig(**bundle["config"])
        model = cls(
            id_column=feature_schema.id_column,
            target_column=feature_schema.target_column,
            model_version=bundle["model_version"],
            schema_version=bundle["schema_version"],
            config=config,
            feature_schema=feature_schema,
            preprocessing_pipeline=bundle["preprocessing_pipeline"],
            missing_indicator_columns=tuple(bundle["missing_indicator_columns"]),
        )
        model.preprocessing_pipeline_ = bundle["preprocessing_pipeline"]
        model.training_summary_ = bundle.get("training_summary")
        model.training_history_ = bundle.get("training_history")
        transformed_feature_names = bundle.get("transformed_feature_names")
        if transformed_feature_names is not None:
            model.transformed_feature_names_ = tuple(transformed_feature_names)
        model.input_dim_ = int(bundle["input_dim"])
        classifier = model._build_network(model.input_dim_, nn)
        classifier.load_state_dict(bundle["state_dict"])
        classifier.eval()
        model.classifier_ = classifier
        return model

    def _build_network(self, input_dim: int, nn: Any) -> Any:
        del nn
        return _TorchMLPBinaryClassifier(
            input_dim=input_dim,
            hidden_dims=self.config.hidden_dims,
            dropout=self.config.dropout,
        ).module

    def _resolve_device(self, torch: Any) -> Any:
        if self.config.device == "cuda" and torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    def _resolve_positive_class_weight(self, target: pd.Series) -> float | None:
        strategy = self.config.positive_class_weight_strategy
        if strategy == "none":
            return None

        positive_count = int((target == 1).sum())
        negative_count = int((target == 0).sum())
        if positive_count == 0 or negative_count == 0:
            msg = "PyTorch challenger requires both classes in the training partition."
            raise ValueError(msg)

        if strategy == "auto_from_train_ratio":
            return float(negative_count / positive_count)
        if strategy == "manual":
            if self.config.positive_class_weight_value is None:
                msg = (
                    "positive_class_weight_value is required when "
                    "positive_class_weight_strategy='manual'."
                )
                raise ValueError(msg)
            if self.config.positive_class_weight_value <= 0.0:
                msg = "positive_class_weight_value must be strictly positive."
                raise ValueError(msg)
            return float(self.config.positive_class_weight_value)

        msg = f"Unsupported positive_class_weight_strategy: {strategy}"
        raise ValueError(msg)

    def _split_features_and_target(
        self,
        dataframe: pd.DataFrame,
    ) -> tuple[pd.DataFrame, pd.Series]:
        if self.target_column not in dataframe.columns:
            msg = f"Target column '{self.target_column}' is missing from the training dataframe."
            raise ValueError(msg)
        feature_frame = self._select_feature_frame(dataframe)
        target = pd.to_numeric(dataframe[self.target_column], errors="raise").astype(int)
        return feature_frame, target

    def _select_feature_frame(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        missing_columns = [
            column
            for column in self.feature_schema.raw_feature_columns
            if column not in dataframe.columns
        ]
        if missing_columns:
            msg = f"Input dataframe is missing model feature columns: {missing_columns}"
            raise ValueError(msg)
        return dataframe.loc[:, self.feature_schema.raw_feature_columns].copy()

    def _fit_transform_features(
        self,
        preprocessing_pipeline: Pipeline,
        feature_frame: pd.DataFrame,
    ) -> pd.DataFrame:
        transformed = preprocessing_pipeline.fit_transform(feature_frame)
        feature_names = self._extract_transformed_feature_names(preprocessing_pipeline)
        self.transformed_feature_names_ = tuple(feature_names)
        return pd.DataFrame(transformed, columns=feature_names, index=feature_frame.index)

    def _transform_features(
        self,
        preprocessing_pipeline: Pipeline,
        feature_frame: pd.DataFrame,
    ) -> pd.DataFrame:
        transformed = preprocessing_pipeline.transform(feature_frame)
        feature_names = self._require_transformed_feature_names()
        return pd.DataFrame(transformed, columns=feature_names, index=feature_frame.index)

    def _extract_transformed_feature_names(self, preprocessing_pipeline: Pipeline) -> list[str]:
        feature_builder = preprocessing_pipeline.named_steps["feature_builder"]
        return feature_builder.get_feature_names_out()

    def _require_fitted_preprocessing(self) -> Pipeline:
        if self.preprocessing_pipeline_ is None:
            msg = "TorchMLPCreditRiskModel must be fitted or loaded before inference."
            raise ValueError(msg)
        return self.preprocessing_pipeline_

    def _require_fitted_classifier(self) -> Any:
        if self.classifier_ is None:
            msg = "TorchMLPCreditRiskModel must be fitted or loaded before inference."
            raise ValueError(msg)
        return self.classifier_

    def _require_transformed_feature_names(self) -> tuple[str, ...]:
        if self.transformed_feature_names_ is None:
            msg = "TorchMLPCreditRiskModel must be fitted or loaded before feature export."
            raise ValueError(msg)
        return self.transformed_feature_names_

    def _require_input_dim(self) -> int:
        if self.input_dim_ is None:
            msg = "TorchMLPCreditRiskModel must be fitted or loaded before persistence."
            raise ValueError(msg)
        return self.input_dim_
