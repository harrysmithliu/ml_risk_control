"""Application configuration for local and Snowflake-backed execution."""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Literal

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - optional before dependencies are installed
    load_dotenv = None

BackendType = Literal["local", "snowflake"]

PACKAGE_DIR = Path(__file__).resolve().parent
SRC_DIR = PACKAGE_DIR.parent
PROJECT_ROOT = SRC_DIR.parent


def _load_environment() -> None:
    """Load variables from a local .env file when python-dotenv is available."""
    if load_dotenv is None:
        return

    load_dotenv(PROJECT_ROOT / ".env", override=False)


def _get_env(name: str, default: str | None = None) -> str:
    value = os.getenv(name, default)
    if value is None:
        msg = f"Missing required environment variable: {name}"
        raise ValueError(msg)
    return value


def _get_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _get_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    return int(value)


def _get_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None:
        return default
    return float(value)


def _resolve_project_path(value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def _get_backend() -> BackendType:
    value = _get_env("DATA_BACKEND", "local").strip().lower()
    if value not in {"local", "snowflake"}:
        msg = "DATA_BACKEND must be either 'local' or 'snowflake'."
        raise ValueError(msg)
    return value  # type: ignore[return-value]


@dataclass(frozen=True)
class AppSettings:
    environment: str
    debug: bool


@dataclass(frozen=True)
class DataSettings:
    backend: BackendType
    raw_data_dir: Path
    interim_data_dir: Path
    processed_data_dir: Path
    train_file: str
    test_file: str
    data_dictionary_file: str

    @property
    def train_path(self) -> Path:
        return self.raw_data_dir / self.train_file

    @property
    def test_path(self) -> Path:
        return self.raw_data_dir / self.test_file

    @property
    def data_dictionary_path(self) -> Path:
        return self.raw_data_dir / self.data_dictionary_file


@dataclass(frozen=True)
class ArtifactSettings:
    artifact_dir: Path
    model_dir: Path
    report_dir: Path
    model_name: str
    model_version: str


@dataclass(frozen=True)
class TrainingSettings:
    target_column: str
    id_column: str
    random_state: int
    test_size: float
    cv_folds: int
    positive_class_label: int
    decision_threshold: float


@dataclass(frozen=True)
class StreamlitSettings:
    server_port: int
    server_address: str


@dataclass(frozen=True)
class SnowflakeSettings:
    account: str
    user: str
    password: str
    warehouse: str
    database: str
    schema: str
    role: str
    raw_schema: str
    curated_schema: str
    features_schema: str
    serving_schema: str

    @property
    def is_configured(self) -> bool:
        required_values = (
            self.account,
            self.user,
            self.password,
            self.warehouse,
            self.database,
        )
        return all(bool(value) for value in required_values)

    def connection_parameters(self) -> dict[str, str]:
        parameters = {
            "account": self.account,
            "user": self.user,
            "password": self.password,
            "warehouse": self.warehouse,
            "database": self.database,
            "schema": self.schema,
        }
        if self.role:
            parameters["role"] = self.role
        return parameters


@dataclass(frozen=True)
class FeatureFlags:
    enable_snowflake_writeback: bool
    enable_monitoring_export: bool


@dataclass(frozen=True)
class Settings:
    project_root: Path
    app: AppSettings
    data: DataSettings
    artifacts: ArtifactSettings
    training: TrainingSettings
    streamlit: StreamlitSettings
    snowflake: SnowflakeSettings
    features: FeatureFlags

    @property
    def use_snowflake(self) -> bool:
        return self.data.backend == "snowflake"


def _build_settings() -> Settings:
    _load_environment()

    return Settings(
        project_root=PROJECT_ROOT,
        app=AppSettings(
            environment=_get_env("APP_ENV", "development"),
            debug=_get_bool("APP_DEBUG", True),
        ),
        data=DataSettings(
            backend=_get_backend(),
            raw_data_dir=_resolve_project_path(_get_env("RAW_DATA_DIR", "data/raw/GiveMeSomeCredit")),
            interim_data_dir=_resolve_project_path(_get_env("INTERIM_DATA_DIR", "data/interim")),
            processed_data_dir=_resolve_project_path(
                _get_env("PROCESSED_DATA_DIR", "data/processed")
            ),
            train_file=_get_env("TRAIN_FILE", "cs-training.csv"),
            test_file=_get_env("TEST_FILE", "cs-test.csv"),
            data_dictionary_file=_get_env("DATA_DICTIONARY_FILE", "Data Dictionary.xls"),
        ),
        artifacts=ArtifactSettings(
            artifact_dir=_resolve_project_path(_get_env("ARTIFACT_DIR", "artifacts")),
            model_dir=_resolve_project_path(_get_env("MODEL_DIR", "artifacts/models")),
            report_dir=_resolve_project_path(_get_env("REPORT_DIR", "reports/figures")),
            model_name=_get_env("MODEL_NAME", "xgboost_credit_risk"),
            model_version=_get_env("MODEL_VERSION", "0.1.0"),
        ),
        training=TrainingSettings(
            target_column=_get_env("TARGET_COLUMN", "SeriousDlqin2yrs"),
            id_column=_get_env("ID_COLUMN", "Unnamed: 0"),
            random_state=_get_int("RANDOM_STATE", 42),
            test_size=_get_float("TEST_SIZE", 0.2),
            cv_folds=_get_int("CV_FOLDS", 5),
            positive_class_label=_get_int("POSITIVE_CLASS_LABEL", 1),
            decision_threshold=_get_float("DECISION_THRESHOLD", 0.5),
        ),
        streamlit=StreamlitSettings(
            server_port=_get_int("STREAMLIT_SERVER_PORT", 8501),
            server_address=_get_env("STREAMLIT_SERVER_ADDRESS", "0.0.0.0"),
        ),
        snowflake=SnowflakeSettings(
            account=_get_env("SNOWFLAKE_ACCOUNT", ""),
            user=_get_env("SNOWFLAKE_USER", ""),
            password=_get_env("SNOWFLAKE_PASSWORD", ""),
            warehouse=_get_env("SNOWFLAKE_WAREHOUSE", ""),
            database=_get_env("SNOWFLAKE_DATABASE", "ML_RISK_CONTROL"),
            schema=_get_env("SNOWFLAKE_SCHEMA", "RAW"),
            role=_get_env("SNOWFLAKE_ROLE", ""),
            raw_schema=_get_env("SNOWFLAKE_RAW_SCHEMA", "RAW"),
            curated_schema=_get_env("SNOWFLAKE_CURATED_SCHEMA", "CURATED"),
            features_schema=_get_env("SNOWFLAKE_FEATURES_SCHEMA", "FEATURES"),
            serving_schema=_get_env("SNOWFLAKE_SERVING_SCHEMA", "SERVING"),
        ),
        features=FeatureFlags(
            enable_snowflake_writeback=_get_bool("ENABLE_SNOWFLAKE_WRITEBACK", False),
            enable_monitoring_export=_get_bool("ENABLE_MONITORING_EXPORT", False),
        ),
    )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached application settings object."""
    return _build_settings()
