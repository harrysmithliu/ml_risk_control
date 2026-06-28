"""Repository abstractions for local files and Snowflake-backed data access."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import pandas as pd

from ml_risk_control.config import Settings, get_settings
from ml_risk_control.data.contracts import (
    FileContract,
    RawDataContracts,
    build_gmsc_raw_data_contracts,
)
from ml_risk_control.data.validation import (
    build_raw_validation_report,
    validate_dataframe_contract,
    validate_file_contract,
)


class RepositoryError(RuntimeError):
    """Base error raised by repository implementations."""


class RepositoryValidationError(RepositoryError):
    """Raised when repository data does not satisfy the declared contract."""


@dataclass(frozen=True)
class SnowflakeTableConfig:
    """Logical table names for Snowflake-backed raw data access."""

    train_table: str = "GMSC_TRAIN"
    test_table: str = "GMSC_TEST"
    sample_submission_table: str = "GMSC_SAMPLE_SUBMISSION"


class CreditRiskRepository(Protocol):
    """Protocol shared by local and Snowflake repository implementations."""

    backend: str

    def get_raw_contracts(self) -> RawDataContracts:
        ...

    def validate_raw_data(self) -> dict[str, Any]:
        ...

    def load_training_data(self) -> pd.DataFrame:
        ...

    def load_scoring_data(self) -> pd.DataFrame:
        ...

    def load_sample_submission_data(self) -> pd.DataFrame | None:
        ...


class BaseRepository(ABC):
    """Shared contract-aware behavior for repository implementations."""

    backend: str

    def __init__(
        self,
        settings: Settings,
        *,
        contracts: RawDataContracts | None = None,
    ) -> None:
        self.settings = settings
        self._contracts = contracts or build_gmsc_raw_data_contracts(
            id_column=settings.training.id_column,
            target_column=settings.training.target_column,
            train_file_name=settings.data.train_file,
            test_file_name=settings.data.test_file,
        )

    def get_raw_contracts(self) -> RawDataContracts:
        return self._contracts

    @abstractmethod
    def validate_raw_data(self) -> dict[str, Any]:
        """Validate the currently configured raw dataset bundle."""

    @abstractmethod
    def load_training_data(self) -> pd.DataFrame:
        """Load the labeled training dataset."""

    @abstractmethod
    def load_scoring_data(self) -> pd.DataFrame:
        """Load the unlabeled scoring dataset."""

    @abstractmethod
    def load_sample_submission_data(self) -> pd.DataFrame | None:
        """Load the optional sample submission template."""


class LocalRepository(BaseRepository):
    """CSV-based repository used for local development and testing."""

    backend = "local"

    def validate_raw_data(self) -> dict[str, Any]:
        return build_raw_validation_report(
            raw_data_dir=self.settings.data.raw_data_dir,
            backend=self.backend,
            contracts=self._contracts,
        )

    def load_training_data(self) -> pd.DataFrame:
        return self._load_validated_csv(
            self.settings.data.train_path,
            self._contracts.train,
        )

    def load_scoring_data(self) -> pd.DataFrame:
        return self._load_validated_csv(
            self.settings.data.test_path,
            self._contracts.test,
        )

    def load_sample_submission_data(self) -> pd.DataFrame | None:
        path = self.settings.data.raw_data_dir / self._contracts.sample_submission.file_name
        if not path.exists():
            return None
        return self._load_validated_csv(path, self._contracts.sample_submission)

    def _load_validated_csv(self, path: Path, contract: FileContract) -> pd.DataFrame:
        report = validate_file_contract(path, contract)
        if report["errors"]:
            raise RepositoryValidationError(
                f"Validation failed for {path.name}: {'; '.join(report['errors'])}"
            )
        return pd.read_csv(path)


class SnowflakeRepository(BaseRepository):
    """Warehouse-backed repository that reads source-aligned tables from Snowflake."""

    backend = "snowflake"

    def __init__(
        self,
        settings: Settings,
        *,
        contracts: RawDataContracts | None = None,
        tables: SnowflakeTableConfig | None = None,
    ) -> None:
        super().__init__(settings, contracts=contracts)
        self.tables = tables or SnowflakeTableConfig()

    def validate_raw_data(self) -> dict[str, Any]:
        files = {
            "train": self._validate_table(self.tables.train_table, self._contracts.train),
            "test": self._validate_table(self.tables.test_table, self._contracts.test),
            "sample_submission": self._validate_table(
                self.tables.sample_submission_table,
                self._contracts.sample_submission,
            ),
        }
        all_errors = [error for report in files.values() for error in report["errors"]]

        return {
            "status": "passed" if not all_errors else "failed",
            "backend": self.backend,
            "raw_data_dir": f"{self.settings.snowflake.database}.{self.settings.snowflake.raw_schema}",
            "dataset_fingerprint": None,
            "files": files,
        }

    def load_training_data(self) -> pd.DataFrame:
        return self._load_validated_table(self.tables.train_table, self._contracts.train)

    def load_scoring_data(self) -> pd.DataFrame:
        return self._load_validated_table(self.tables.test_table, self._contracts.test)

    def load_sample_submission_data(self) -> pd.DataFrame | None:
        try:
            return self._load_validated_table(
                self.tables.sample_submission_table,
                self._contracts.sample_submission,
            )
        except RepositoryError:
            return None

    def _load_validated_table(self, table_name: str, contract: FileContract) -> pd.DataFrame:
        dataframe = self._read_table(table_name)
        report = validate_dataframe_contract(
            dataframe,
            contract,
            source=self._table_locator(table_name),
        )
        if report["errors"]:
            raise RepositoryValidationError(
                f"Validation failed for {table_name}: {'; '.join(report['errors'])}"
            )
        return dataframe

    def _validate_table(self, table_name: str, contract: FileContract) -> dict[str, Any]:
        try:
            dataframe = self._read_table(table_name)
        except RepositoryError as error:
            return {
                "path": self._table_locator(table_name),
                "exists": False,
                "errors": [str(error)] if contract.required else [],
                "warnings": [] if contract.required else [str(error)],
            }

        return validate_dataframe_contract(
            dataframe,
            contract,
            source=self._table_locator(table_name),
        )

    def _read_table(self, table_name: str) -> pd.DataFrame:
        if not self.settings.snowflake.is_configured:
            raise RepositoryError("Snowflake credentials are not fully configured.")

        try:
            import snowflake.connector
        except ImportError as error:  # pragma: no cover - depends on environment setup
            raise RepositoryError("snowflake-connector-python is not installed.") from error

        query = f"SELECT * FROM {self._table_locator(table_name)}"
        connection = snowflake.connector.connect(
            **self.settings.snowflake.connection_parameters()
        )
        try:
            return pd.read_sql(query, connection)
        finally:
            connection.close()

    def _table_locator(self, table_name: str) -> str:
        return (
            f"{self.settings.snowflake.database}."
            f"{self.settings.snowflake.raw_schema}."
            f"{table_name}"
        )


def build_repository(settings: Settings | None = None) -> CreditRiskRepository:
    """Create the repository implementation selected by configuration."""
    resolved_settings = settings or get_settings()
    if resolved_settings.data.backend == "snowflake":
        return SnowflakeRepository(resolved_settings)
    return LocalRepository(resolved_settings)
