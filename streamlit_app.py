"""Stage 6 Streamlit entry point for local single-applicant scoring."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import streamlit as st

from ml_risk_control.config import get_settings
from ml_risk_control.inference.service import (
    ApplicantValidationError,
    ArtifactLoadError,
    LocalXGBoostInferenceService,
)


st.set_page_config(
    page_title="Credit Risk Intelligence Platform",
    page_icon="📉",
    layout="wide",
)


PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_FIGURE_DIR = PROJECT_ROOT / "reports" / "figures" / "model"
MODEL_FIGURES: tuple[tuple[str, str], ...] = (
    ("Validation Precision-Recall Curve", "pr_curve_validation.png"),
    ("Validation ROC Curve", "roc_curve_validation.png"),
    ("Native XGBoost Gain Importance", "native_importance_gain.png"),
    ("Validation Permutation Importance", "permutation_importance.png"),
    ("Validation Threshold Selection", "threshold_selection_validation.png"),
    ("Validation Cost Analysis", "cost_analysis_validation.png"),
)
DEMO_APPLICANT: dict[str, Any] = {
    "RevolvingUtilizationOfUnsecuredLines": 0.45,
    "age": 42,
    "NumberOfTime30-59DaysPastDueNotWorse": 0,
    "DebtRatio": 0.32,
    "MonthlyIncome": "6500",
    "NumberOfOpenCreditLinesAndLoans": 8,
    "NumberOfTimes90DaysLate": 0,
    "NumberRealEstateLoansOrLines": 1,
    "NumberOfTime60-89DaysPastDueNotWorse": 0,
    "NumberOfDependents": "2",
}


@st.cache_resource(show_spinner=False)
def get_inference_service() -> LocalXGBoostInferenceService:
    """Load and cache the local inference service."""
    return LocalXGBoostInferenceService().load()


def _initialize_form_state() -> None:
    for field_name, value in DEMO_APPLICANT.items():
        st.session_state.setdefault(field_name, value)


def _apply_demo_example() -> None:
    for field_name, value in DEMO_APPLICANT.items():
        st.session_state[field_name] = value


def _render_header() -> None:
    st.title("Credit Risk Intelligence Platform")
    st.caption(
        "Local artifact-backed Streamlit demo for single-applicant delinquency scoring."
    )
    st.warning(
        "This application is for educational and portfolio demonstration only. "
        "It must not be used to make real lending decisions.",
        icon="⚠️",
    )


def _render_status_section(service: LocalXGBoostInferenceService) -> None:
    snapshot = service.build_status_snapshot()

    st.subheader("Application Status")
    metric_columns = st.columns(4)
    metric_columns[0].metric("Selected Candidate", snapshot["selected_candidate_source"])
    metric_columns[1].metric("F1 Threshold", f"{snapshot['f1_threshold']:.3f}")
    metric_columns[2].metric("Cost Threshold", f"{snapshot['cost_threshold']:.3f}")
    metric_columns[3].metric("Calibration", snapshot["calibration_method"] or "none")

    with st.expander("Loaded Artifact Context", expanded=False):
        st.json(snapshot)


def _read_optional_numeric_text(field_name: str) -> float | int | None | str:
    raw_value = st.session_state.get(field_name, "")
    if raw_value is None:
        return None
    if isinstance(raw_value, str) and raw_value.strip() == "":
        return None
    return raw_value


def _build_applicant_payload() -> dict[str, Any]:
    return {
        "RevolvingUtilizationOfUnsecuredLines": st.session_state["RevolvingUtilizationOfUnsecuredLines"],
        "age": st.session_state["age"],
        "NumberOfTime30-59DaysPastDueNotWorse": st.session_state[
            "NumberOfTime30-59DaysPastDueNotWorse"
        ],
        "DebtRatio": st.session_state["DebtRatio"],
        "MonthlyIncome": _read_optional_numeric_text("MonthlyIncome"),
        "NumberOfOpenCreditLinesAndLoans": st.session_state["NumberOfOpenCreditLinesAndLoans"],
        "NumberOfTimes90DaysLate": st.session_state["NumberOfTimes90DaysLate"],
        "NumberRealEstateLoansOrLines": st.session_state["NumberRealEstateLoansOrLines"],
        "NumberOfTime60-89DaysPastDueNotWorse": st.session_state[
            "NumberOfTime60-89DaysPastDueNotWorse"
        ],
        "NumberOfDependents": _read_optional_numeric_text("NumberOfDependents"),
    }


def _render_input_section() -> bool:
    st.subheader("Applicant Input")
    st.caption(
        "Provide the current applicant profile. "
        "Monthly income and number of dependents may be left blank."
    )

    with st.form("single_applicant_scoring_form", clear_on_submit=False):
        left_column, right_column = st.columns(2)

        with left_column:
            st.number_input(
                "Revolving Utilization of Unsecured Lines",
                min_value=0.0,
                step=0.01,
                key="RevolvingUtilizationOfUnsecuredLines",
                help="Observed values above 1 may exist in the raw dataset but will be clipped by inference preprocessing.",
            )
            st.number_input(
                "Age",
                min_value=18,
                max_value=120,
                step=1,
                key="age",
            )
            st.number_input(
                "30-59 Days Past Due Count",
                min_value=0,
                step=1,
                key="NumberOfTime30-59DaysPastDueNotWorse",
            )
            st.number_input(
                "Debt Ratio",
                min_value=0.0,
                step=0.01,
                key="DebtRatio",
            )
            st.text_input(
                "Monthly Income",
                key="MonthlyIncome",
                help="Leave blank to preserve a missing value.",
            )

        with right_column:
            st.number_input(
                "Open Credit Lines and Loans",
                min_value=0,
                step=1,
                key="NumberOfOpenCreditLinesAndLoans",
            )
            st.number_input(
                "90+ Days Late Count",
                min_value=0,
                step=1,
                key="NumberOfTimes90DaysLate",
            )
            st.number_input(
                "Real Estate Loans or Lines",
                min_value=0,
                step=1,
                key="NumberRealEstateLoansOrLines",
            )
            st.number_input(
                "60-89 Days Past Due Count",
                min_value=0,
                step=1,
                key="NumberOfTime60-89DaysPastDueNotWorse",
            )
            st.text_input(
                "Number of Dependents",
                key="NumberOfDependents",
                help="Leave blank to preserve a missing value.",
            )

        action_left, action_right = st.columns([1, 1])
        with action_left:
            score_submitted = st.form_submit_button("Score Applicant", width="stretch")
        with action_right:
            demo_submitted = st.form_submit_button("Use Demo Example", width="stretch")

    if demo_submitted:
        _apply_demo_example()
        st.info("Demo example applied. Review the values and click 'Score Applicant'.")

    return bool(score_submitted)


def _render_result_section(service: LocalXGBoostInferenceService) -> None:
    st.subheader("Risk Result")

    result = st.session_state.get("latest_score_result")
    if result is None:
        st.info("Submit an applicant profile to generate a risk score.")
        return

    threshold_lookup = {item["name"]: item for item in result["threshold_decisions"]}
    probability = result["predicted_probability"]
    risk_band = result["risk_band"]
    f1_decision = threshold_lookup["f1_validation_threshold"]
    cost_decision = threshold_lookup["cost_validation_threshold"]

    columns = st.columns(4)
    columns[0].metric("Predicted Probability", f"{probability:.2%}")
    columns[1].metric("Risk Band", risk_band)
    columns[2].metric(
        "F1 Threshold Decision",
        "Flag" if f1_decision["predicted_label"] == 1 else "Pass",
        delta=f"threshold={f1_decision['threshold']:.3f}",
    )
    columns[3].metric(
        "Cost Threshold Decision",
        "Flag" if cost_decision["predicted_label"] == 1 else "Pass",
        delta=f"threshold={cost_decision['threshold']:.3f}",
    )

    calibration_summary = result["calibration_summary"]
    improved_brier = calibration_summary["improved_validation_brier"]
    if improved_brier is True:
        calibration_note = "Calibration improved validation Brier score."
    elif improved_brier is False:
        calibration_note = "Calibration was evaluated but did not improve validation Brier score."
    else:
        calibration_note = "Calibration metadata is available but improvement could not be determined."

    st.markdown(
        f"""
        **Interpretation**

        - This score uses the persisted local XGBoost artifact.
        - The saved model candidate source is `{result['selected_candidate_source']}`.
        - The primary selection metric remains `{result['primary_selection_metric']}`.
        - {calibration_note}
        """
    )

    with st.expander("Normalized Applicant Payload", expanded=False):
        st.json(result["input_frame"])

    with st.expander("Threshold and Calibration Details", expanded=False):
        st.json(
            {
                "threshold_decisions": result["threshold_decisions"],
                "calibration_summary": calibration_summary,
                "service_snapshot": service.build_status_snapshot(),
            }
        )


def _render_model_diagnostics() -> None:
    st.subheader("Model Diagnostics")

    tabs = st.tabs(
        [
            "Curves",
            "Explainability",
            "Thresholding",
        ]
    )

    grouped_figures = {
        "Curves": MODEL_FIGURES[:2],
        "Explainability": MODEL_FIGURES[2:4],
        "Thresholding": MODEL_FIGURES[4:],
    }

    for tab, tab_name in zip(tabs, grouped_figures, strict=True):
        with tab:
            for title, file_name in grouped_figures[tab_name]:
                figure_path = MODEL_FIGURE_DIR / file_name
                if figure_path.exists():
                    st.image(str(figure_path), caption=title, width="stretch")
                else:
                    st.warning(f"Figure not found: {figure_path}")


def main() -> None:
    _initialize_form_state()
    _render_header()

    try:
        service = get_inference_service()
    except ArtifactLoadError as error:
        st.error(f"Unable to load the local inference artifacts: {error}")
        st.stop()
    except Exception as error:  # pragma: no cover - defensive UI guard
        st.error(f"Unexpected application startup error: {error}")
        st.stop()

    _render_status_section(service)
    layout_left, layout_right = st.columns([1.15, 1.0])

    with layout_left:
        score_requested = _render_input_section()
        if score_requested:
            try:
                applicant_payload = _build_applicant_payload()
                result = service.score_applicant(applicant_payload)
                st.session_state["latest_score_result"] = result.to_dict()
            except ApplicantValidationError as error:
                st.error(f"Input validation error: {error}")
            except Exception as error:  # pragma: no cover - defensive UI guard
                st.error(f"Unexpected scoring error: {error}")

    with layout_right:
        _render_result_section(service)

    st.divider()
    _render_model_diagnostics()

    settings = get_settings()
    with st.expander("Environment and Runtime Metadata", expanded=False):
        st.json(
            {
                "project_root": str(settings.project_root),
                "artifact_dir": str(settings.artifacts.artifact_dir),
                "report_dir": str(settings.artifacts.report_dir),
                "streamlit_host": settings.streamlit.server_address,
                "streamlit_port": settings.streamlit.server_port,
            }
        )


if __name__ == "__main__":
    main()
