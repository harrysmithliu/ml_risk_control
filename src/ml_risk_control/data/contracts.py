"""Dataset contracts for raw ingestion and downstream schema validation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

TargetMode = Literal["required", "optional_empty", "not_expected"]


@dataclass(frozen=True)
class FileContract:
    """Declarative schema contract for a source-aligned file."""

    name: str
    file_name: str
    expected_columns: tuple[str, ...]
    optional_columns: tuple[str, ...] = ()
    required: bool = True
    id_column: str | None = None
    target_column: str | None = None
    target_mode: TargetMode = "not_expected"

    @property
    def allowed_columns(self) -> tuple[str, ...]:
        return self.expected_columns + tuple(
            column for column in self.optional_columns if column not in self.expected_columns
        )

    def expects_target(self) -> bool:
        return self.target_column is not None and self.target_mode == "required"

    def allows_empty_target_placeholder(self) -> bool:
        return self.target_column is not None and self.target_mode == "optional_empty"


@dataclass(frozen=True)
class RawDataContracts:
    """Container for the benchmark dataset file contracts."""

    train: FileContract
    test: FileContract
    sample_submission: FileContract

    def as_dict(self) -> dict[str, FileContract]:
        return {
            "train": self.train,
            "test": self.test,
            "sample_submission": self.sample_submission,
        }


GMSC_FEATURE_COLUMNS: tuple[str, ...] = (
    "RevolvingUtilizationOfUnsecuredLines",
    "age",
    "NumberOfTime30-59DaysPastDueNotWorse",
    "DebtRatio",
    "MonthlyIncome",
    "NumberOfOpenCreditLinesAndLoans",
    "NumberOfTimes90DaysLate",
    "NumberRealEstateLoansOrLines",
    "NumberOfTime60-89DaysPastDueNotWorse",
    "NumberOfDependents",
)

SAMPLE_SUBMISSION_COLUMNS: tuple[str, ...] = ("Id", "Probability")
SAMPLE_SUBMISSION_FILE_NAME = "sampleEntry.csv"


def build_gmsc_raw_data_contracts(
    *,
    id_column: str,
    target_column: str,
    train_file_name: str,
    test_file_name: str,
    sample_submission_file_name: str = SAMPLE_SUBMISSION_FILE_NAME,
) -> RawDataContracts:
    """Build file contracts for the Give Me Some Credit raw dataset."""

    train_columns = (id_column, target_column, *GMSC_FEATURE_COLUMNS)
    test_columns = (id_column, *GMSC_FEATURE_COLUMNS)

    return RawDataContracts(
        train=FileContract(
            name="train",
            file_name=train_file_name,
            expected_columns=train_columns,
            id_column=id_column,
            target_column=target_column,
            target_mode="required",
        ),
        test=FileContract(
            name="test",
            file_name=test_file_name,
            expected_columns=test_columns,
            optional_columns=(target_column,),
            id_column=id_column,
            target_column=target_column,
            target_mode="optional_empty",
        ),
        sample_submission=FileContract(
            name="sample_submission",
            file_name=sample_submission_file_name,
            expected_columns=SAMPLE_SUBMISSION_COLUMNS,
            required=False,
            id_column="Id",
            target_column=None,
            target_mode="not_expected",
        ),
    )
