"""C.R.A.M. V2 customer-risk dashboard (kept separate from the V1 app.py)."""

from __future__ import annotations

from pathlib import Path
import hashlib
import json

import numpy as np
import pandas as pd
import streamlit as st

from ml.predict import ChurnPredictor
from ml.preprocessing import (
    CATEGORICAL_INPUTS,
    INPUT_FEATURES,
    INTEGER_FIELDS,
    VALID_RANGES,
    CustomerDataValidator,
)
from ml.risk_intelligence import (
    RiskThresholds,
    analyze_portfolio,
    classify_risk,
    choose_dimension_column,
    churn_rate_by_group,
    derive_usage_charge_proxy,
    portfolio_summary,
)
from ml.recommendations import recommend_customer, recommend_portfolio
from ml.simulation import simulate_scenario

ROOT = Path(__file__).resolve().parent

st.set_page_config(page_title="C.R.A.M. V2 | Risk Intelligence", page_icon="📊", layout="wide")


@st.cache_resource
def load_predictor() -> ChurnPredictor:
    return ChurnPredictor()


@st.cache_data(show_spinner=False)
def _cached_portfolio_analysis(data: pd.DataFrame, _predictor: ChurnPredictor):
    return analyze_portfolio(data, _predictor, RiskThresholds.from_metadata(_predictor.metadata))


def _cached_profile_explanation(predictor: ChurnPredictor, profile: dict) -> dict:
    serialized = json.dumps(profile, sort_keys=True, default=str, allow_nan=True)
    signature = hashlib.sha256(serialized.encode("utf-8")).hexdigest()
    cached = st.session_state.get("cram_selected_profile_explanation")
    if cached and cached.get("signature") == signature:
        return cached["explanation"]
    explanation = predictor.explain_single(profile)
    st.session_state["cram_selected_profile_explanation"] = {
        "signature": signature,
        "explanation": explanation,
    }
    return explanation


def _numeric_range(frame: pd.DataFrame, column: str, label: str, default_low: float, default_high: float):
    values = pd.to_numeric(frame[column], errors="coerce").dropna()
    if values.empty:
        st.sidebar.info(f"No numeric values available for {label}.")
        return None
    data_min, data_max = float(values.min()), float(values.max())
    if data_min == data_max:
        st.sidebar.caption(f"{label}: {data_min:g} (only observed value)")
        return (data_min, data_max)
    lo = min(max(default_low, data_min), data_max)
    hi = min(max(default_high, lo), data_max)
    return st.sidebar.slider(label, min_value=data_min, max_value=data_max, value=(lo, hi))


def _filtered_view(frame: pd.DataFrame) -> pd.DataFrame:
    view = frame
    st.sidebar.subheader("Portfolio filters")
    chosen_risks = st.sidebar.multiselect("Risk category", ["High Risk", "Medium Risk", "Low Risk"], default=["High Risk", "Medium Risk", "Low Risk"])
    view = view[view["Risk_Category"].isin(chosen_risks)]

    search = st.sidebar.text_input("Search customer ID or profile values")
    if search:
        searchable = ["Customer ID", *[feature for feature in INPUT_FEATURES if feature in view.columns]]
        for candidates in (
            ("Contract Type", "Contract", "Contract type", "ContractType"),
            ("Tenure", "Customer tenure"),
            ("Payment Method", "PaymentMethod", "Payment method"),
            ("Monthly Charges", "MonthlyCharge", "Monthly charge", "MonthlyCharges", "Monthly_Charges"),
        ):
            column = choose_dimension_column(view, candidates)
            if column and column not in searchable:
                searchable.append(column)
        matches = view[searchable].astype("string").apply(
            lambda column: column.str.contains(search, case=False, na=False)
        ).any(axis=1)
        view = view[matches]

    contract_column = choose_dimension_column(frame, ("Contract Type", "Contract", "Contract type", "ContractType"))
    if contract_column:
        options = sorted(frame[contract_column].dropna().astype(str).unique())
        if options:
            selected = st.sidebar.multiselect("Contract type", options, default=options)
            view = view[view[contract_column].astype(str).isin(selected)]
        else:
            st.sidebar.caption("No non-blank contract types are available to filter.")
    else:
        st.sidebar.caption("Contract type filter unavailable: this file has no contract-type column.")

    for column, label in (("International plan", "International plan"), ("Voice mail plan", "Voice mail plan")):
        if column in frame:
            options = sorted(frame[column].dropna().astype(str).unique())
            if options:
                selected = st.sidebar.multiselect(label, options, default=options, key=f"filter_{column}")
                view = view[view[column].astype(str).isin(selected)]
            else:
                st.sidebar.caption(f"No non-blank values are available for {label.lower()}.")

    tenure_column = choose_dimension_column(frame, ("Tenure", "Customer tenure", "Account length"))
    if tenure_column:
        tenure_label = "Tenure" if tenure_column.casefold() != "account length" else "Account length (available proxy)"
        limits = _numeric_range(frame, tenure_column, tenure_label, -np.inf, np.inf)
        if limits:
            vals = pd.to_numeric(view[tenure_column], errors="coerce")
            view = view[vals.between(*limits) | vals.isna()]
    else:
        st.sidebar.caption("Tenure filter unavailable: this file has no tenure or account-length column.")

    monthly_column = choose_dimension_column(frame, ("Monthly Charges", "MonthlyCharge", "Monthly charge", "MonthlyCharges", "Monthly_Charges"))
    if monthly_column:
        amount_column, amount_label = monthly_column, "Monthly charges"
    else:
        amount_column, amount_label = "Derived Usage Charge Total", "Derived usage-charge total ($ proxy; not a monthly bill)"
        st.sidebar.caption("No monthly-bill field supplied. Filtering the model's derived usage-charge proxy instead.")
    limits = _numeric_range(frame, amount_column, amount_label, -np.inf, np.inf)
    if limits:
        vals = pd.to_numeric(view[amount_column], errors="coerce")
        view = view[vals.between(*limits) | vals.isna()]
    return view.copy()


def _rate_chart(frame: pd.DataFrame, column: Optional[str], title: str) -> None:
    if column is None:
        st.info(f"{title} is unavailable because this upload does not contain that field.")
        return
    chart = churn_rate_by_group(frame, column)
    if chart.empty:
        st.info(f"No usable values are available for {title.lower()}.")
        return
    chart = chart.sort_values("rate", ascending=False)
    has_observed = "Observed_Churn" in frame and frame["Observed_Churn"].notna().any()
    st.bar_chart(chart.set_index(column)["rate"], y_label="Observed churn rate" if has_observed else "Mean predicted churn probability")
    st.caption("Counts by group: " + ", ".join(f"{row[column]}: {int(row['customers'])}" for _, row in chart.iterrows()))


def _clear_scenario_widgets(customer_widget_key: str, slot: str) -> None:
    st.session_state.pop(f"sim_features_{customer_widget_key}_{slot}", None)
    for feature in INPUT_FEATURES:
        st.session_state.pop(f"sim_value_{customer_widget_key}_{slot}_{feature}", None)


def _reset_simulation_scenario(customer_widget_key: str, slot: str) -> None:
    _clear_scenario_widgets(customer_widget_key, slot)


def _remove_simulation_scenario(customer_widget_key: str, slot: str) -> None:
    state = st.session_state.get("cram_simulation_state", {})
    state.get("scenarios", {}).pop(slot, None)
    _clear_scenario_widgets(customer_widget_key, slot)


def _create_simulation_scenario(customer_id: str) -> None:
    state = st.session_state.get("cram_simulation_state", {})
    if state.get("customer_id") != customer_id:
        return
    scenarios = state.get("scenarios", {})
    for slot in ("A", "B", "C"):
        if slot not in scenarios:
            scenarios[slot] = {"name": f"Scenario {slot}"}
            return


def _render_simulation_lab(predictor: ChurnPredictor, customer_id: str, profile: dict) -> None:
    st.subheader("What-If Simulation Lab")
    st.warning(
        "These simulations show how the existing machine-learning model responds to hypothetical input changes. "
        "They are not causal estimates and do not prove that making these changes would produce the displayed outcome in real customers."
    )
    st.caption(
        "Predictions depend on this model and its training data. Hypothetical combinations may differ from real customer situations; "
        "the model may not have observed every combination. Output changes do not establish business effectiveness and do not account "
        "for intervention costs or customer responses."
    )
    state_key = "cram_simulation_state"
    state = st.session_state.get(state_key)
    if not isinstance(state, dict) or state.get("customer_id") != customer_id:
        state = {"customer_id": customer_id, "scenarios": {}}
        st.session_state[state_key] = state
    customer_widget_key = str(abs(hash(customer_id)))
    scenarios = state["scenarios"]

    baseline_result = predictor.predict_single(profile, explain_sensitivities=False)
    st.metric("Baseline predicted churn probability", f"{baseline_result['churn_probability']:.1%}")
    st.caption(
        f"Selected customer: {customer_id}. Binary classification uses the existing Phase 1 threshold "
        f"({predictor.optimal_threshold:.4f}); the probability remains the primary comparison."
    )
    available = [slot for slot in ("A", "B", "C") if slot not in scenarios]
    st.button(
        "Create scenario",
        disabled=not available,
        key=f"sim_create_{customer_widget_key}",
        on_click=_create_simulation_scenario,
        args=(customer_id,),
    )

    cleaned = CustomerDataValidator().validate(
        pd.DataFrame([profile]).reindex(columns=INPUT_FEATURES),
        remove_duplicates=False,
    ).cleaned_df.iloc[0]
    original_inputs = pd.DataFrame([
        {
            "Model input": feature,
            "Original value": "Missing" if feature not in profile or pd.isna(profile.get(feature)) else str(profile.get(feature)),
            "Validation status": (
                "genuinely missing; pipeline imputes"
                if feature not in profile or pd.isna(profile.get(feature))
                else "rejected/coerced; pipeline imputes"
                if pd.isna(cleaned[feature])
                else "accepted"
            ),
        }
        for feature in INPUT_FEATURES
    ])
    with st.expander("View original model inputs", expanded=False):
        st.dataframe(original_inputs, hide_index=True, width="stretch")
    if baseline_result["estimated_fields"]:
        st.info("The baseline probability uses pipeline imputation for: " + ", ".join(baseline_result["estimated_fields"]))
    output_rows = [{
        "Scenario": "Baseline",
        "Predicted churn probability": baseline_result["churn_probability"],
        "Change from baseline (percentage points)": 0.0,
        "Model classification": baseline_result["prediction"],
        "Risk category": baseline_result["risk_category"],
        "Changed inputs": "—",
    }]
    for slot in ("A", "B", "C"):
        if slot not in scenarios:
            continue
        scenario = scenarios[slot]
        with st.expander(f"{scenario['name']} — edit inputs", expanded=True):
            scenario["name"] = st.text_input(
                "Scenario name",
                value=scenario["name"],
                max_chars=40,
                key=f"sim_name_{customer_widget_key}_{slot}",
            )
            feature_key = f"sim_features_{customer_widget_key}_{slot}"
            selected_features = st.multiselect(
                "Attributes to vary",
                INPUT_FEATURES,
                key=feature_key,
                help="Only the model's original 13 validated inputs are editable. IDs, observed churn labels, and derived charges are not model inputs.",
            )
            changes = {}
            columns = st.columns(2)
            for index, feature in enumerate(selected_features):
                low, high = VALID_RANGES[feature]
                value = cleaned[feature]
                if pd.isna(value):
                    value = predictor.medians.get(feature, 0.0)
                value = min(max(float(value), float(low)), float(high))
                label = feature
                help_text = "Hypothetical model input only; changing this value does not imply a real-world intervention can change it."
                if feature == "Account length":
                    label = "Account length (not verified tenure)"
                    help_text = "This is the source model's Account length field; it is not verified customer tenure."
                with columns[index % 2]:
                    if feature in CATEGORICAL_INPUTS:
                        setting = st.selectbox(
                            label,
                            options=[0.0, 1.0],
                            index=int(round(value)),
                            format_func=lambda option: "Yes" if option == 1.0 else "No",
                            help=help_text,
                            key=f"sim_value_{customer_widget_key}_{slot}_{feature}",
                        )
                    elif feature in INTEGER_FIELDS:
                        setting = st.slider(
                            label,
                            min_value=int(low),
                            max_value=int(high),
                            value=int(round(value)),
                            step=1,
                            help=help_text,
                            key=f"sim_value_{customer_widget_key}_{slot}_{feature}",
                        )
                    else:
                        setting = st.slider(
                            label,
                            min_value=float(low),
                            max_value=float(high),
                            value=float(value),
                            step=0.1,
                            help=help_text,
                            key=f"sim_value_{customer_widget_key}_{slot}_{feature}",
                        )
                    changes[feature] = setting

            reset_col, remove_col = st.columns(2)
            with reset_col:
                st.button(
                    "Reset scenario",
                    key=f"sim_reset_{customer_widget_key}_{slot}",
                    on_click=_reset_simulation_scenario,
                    args=(customer_widget_key, slot),
                )
            with remove_col:
                st.button(
                    "Remove scenario",
                    key=f"sim_remove_{customer_widget_key}_{slot}",
                    on_click=_remove_simulation_scenario,
                    args=(customer_widget_key, slot),
                )

        result = simulate_scenario(
            predictor,
            profile,
            changes,
            threshold=predictor.optimal_threshold,
        )
        changed_summary = "; ".join(
            f"{feature}: {values['original_value']} → {values['simulated_value']}"
            for feature, values in result.changed_features.items()
        ) or "No input changes"
        direction = (
            "lower model probability" if result.difference < 0
            else "higher model probability" if result.difference > 0
            else "no model probability change"
        )
        output_rows.append({
            "Scenario": scenario["name"],
            "Predicted churn probability": result.simulated_probability,
            "Change from baseline (percentage points)": result.difference_percentage_points,
            "Model classification": result.simulated_prediction,
            "Risk category": result.simulated_risk_category,
            "Changed inputs": changed_summary,
        })
        if result.out_of_training_range_features:
            st.warning(
                f"{scenario['name']} includes values outside observed single-feature training ranges: "
                + ", ".join(result.out_of_training_range_features)
                + ". This is a marginal range check, not a validated detector for unusual feature combinations."
            )
        if result.estimated_features:
            st.info(
                f"{scenario['name']} still relies on pipeline imputation for: "
                + ", ".join(result.estimated_features)
            )
        st.caption(
            f"{scenario['name']}: {direction} ({result.difference_percentage_points:+.2f} percentage points). "
            "This describes only the model output for the hypothetical inputs."
        )

    output = pd.DataFrame(output_rows)
    output["Predicted churn probability"] = output["Predicted churn probability"].map(lambda value: f"{value:.2%}")
    output["Change from baseline (percentage points)"] = output["Change from baseline (percentage points)"].map(lambda value: f"{value:+.2f} pp")
    st.dataframe(output, hide_index=True, width="stretch")
    st.caption(
        "Account length is not verified tenure. Derived usage-charge totals are not editable because the serving pipeline derives them from usage minutes. "
        "The scenario editor accepts the source model's canonical inputs only."
    )


predictor = load_predictor()
default_risk = RiskThresholds.from_metadata(predictor.metadata)

st.title("Customer Risk Intelligence")
st.caption("Explore churn risk across a customer portfolio, then inspect an individual prediction and its model explanation.")

with st.sidebar:
    st.header("Analyze a customer file")
    uploaded = st.file_uploader("Upload a CSV", type=["csv"], help="Include all 13 model input columns. Optional: Customer ID and Churn (observed label).")
    st.markdown("**Required model inputs**")
    st.caption(", ".join(INPUT_FEATURES))
    st.divider()
    st.subheader("Risk-band thresholds")
    st.caption("These bands are separate from the model's binary churn decision. Defaults reuse Phase 1 validation thresholds.")
    high_threshold = st.slider("High risk starts at", min_value=0.02, max_value=1.0, value=float(default_risk.high_threshold), step=0.01)
    medium_max = max(0.0, round(high_threshold - 0.01, 4))
    medium_threshold = st.slider(
        "Medium risk starts at",
        min_value=0.0,
        max_value=medium_max,
        value=min(float(default_risk.medium_threshold), medium_max),
        step=0.005,
        help="Below this is Low Risk; at or above the high threshold is High Risk.",
    )

if uploaded is None:
    st.info("Upload a customer CSV to score the portfolio. The file stays in this app session and is not saved by this dashboard.")
    st.markdown("The dashboard requires all 13 model inputs. `Customer ID` is optional; if it is absent, temporary row IDs are created. A `Churn` column is optional and is used only to show observed rates when valid labels are provided.")
    st.stop()

try:
    uploaded.seek(0)
    raw = pd.read_csv(uploaded)
    thresholds = RiskThresholds(medium_threshold=float(medium_threshold), high_threshold=float(high_threshold))
    portfolio = _cached_portfolio_analysis(raw, predictor)
    portfolio.customers["Risk_Category"] = classify_risk(
        portfolio.customers["Churn_Probability"],
        thresholds,
    )
    portfolio.risk_thresholds = thresholds
except (ValueError, TypeError, pd.errors.ParserError, UnicodeDecodeError) as exc:
    st.error(str(exc))
    st.stop()

for warning in portfolio.warnings:
    st.warning(warning)

all_scored = portfolio.customers
st.caption(
    f"Binary prediction cutoff: {portfolio.prediction_threshold:.4f} (Phase 1 validation cost threshold). "
    f"Risk bands: Low < {thresholds.medium_threshold:.4f}; Medium ≥ {thresholds.medium_threshold:.4f}; "
    f"High ≥ {thresholds.high_threshold:.2f}. Changing bands does not change model probabilities."
)
filtered = _filtered_view(all_scored)
summary = portfolio_summary(filtered) if not filtered.empty else None

st.subheader("Portfolio overview")
if summary is None:
    st.warning("The current filters match no customers. Adjust the filters to see portfolio metrics.")
else:
    metrics = st.columns(5)
    metrics[0].metric("Customers analyzed", f"{len(all_scored):,}", help=f"{summary['customers_analyzed']:,} customers in the current filtered view")
    metrics[1].metric("Predicted churners", f"{summary['predicted_churners']:,}", help=f"Probability ≥ {portfolio.prediction_threshold:.4f}; separate from risk band")
    metrics[2].metric("Average predicted probability", f"{summary['average_predicted_probability']:.1%}")
    metrics[3].metric("High-risk customers", f"{summary['high_risk_customers']:,}", help=f"Probability ≥ {thresholds.high_threshold:.2f}")
    if summary["observed_churn_rate"] is None:
        metrics[4].metric("Observed churn rate", "Unavailable", help="Provide a valid Churn label column to calculate this measure")
    else:
        metrics[4].metric("Observed churn rate", f"{summary['observed_churn_rate']:.1%}", help=f"Based on {summary['observed_label_count']:,} labeled customers in the current view")
    st.caption("Portfolio rates and risk counts below use the current filters; the analyzed-customer total reflects the full upload.")

st.subheader("Risk distribution")
risk_counts = filtered["Risk_Category"].value_counts().reindex(["High Risk", "Medium Risk", "Low Risk"], fill_value=0)
left, right = st.columns([1, 1])
with left:
    st.bar_chart(risk_counts, y_label="Customers")
with right:
    if portfolio.label_column_present and "Observed_Churn" in filtered:
        labeled = filtered["Observed_Churn"].dropna()
        if len(labeled):
            st.write("Observed labels available for", f"{len(labeled):,}", "customers in this view.")
            st.caption("Observed label rates are descriptive dataset statistics, not intervention outcomes.")
        else:
            st.info("The upload includes a Churn column, but it has no valid observed labels in the current view.")
    else:
        st.info("No observed Churn labels were supplied. Group charts show model probabilities, not actual churn rates.")

st.subheader("Customer patterns")
contract = choose_dimension_column(filtered, ("Contract Type", "Contract", "Contract type", "ContractType"))
tenure = choose_dimension_column(filtered, ("Tenure", "Customer tenure", "Account length"))
payment = choose_dimension_column(filtered, ("Payment Method", "PaymentMethod", "Payment method"))
monthly = choose_dimension_column(filtered, ("Monthly Charges", "MonthlyCharge", "Monthly charge", "MonthlyCharges", "Monthly_Charges"))
plan_columns = [name for name in ("International plan", "Voice mail plan") if name in filtered]
chart1, chart2 = st.columns(2)
with chart1:
    if contract:
        _rate_chart(filtered, contract, "Churn by contract type")
    elif plan_columns:
        _rate_chart(filtered, plan_columns[0], "Churn by available service plan")
with chart2:
    if tenure:
        tenure_label = "Account length group (proxy; not a verified tenure field)" if tenure.casefold() == "account length" else "Churn by tenure group"
        numeric = pd.to_numeric(filtered[tenure], errors="coerce")
        valid = filtered.loc[numeric.notna()].copy()
        if numeric.nunique() > 1:
            valid["_tenure_group"] = pd.qcut(numeric.loc[valid.index], q=min(5, numeric.nunique()), duplicates="drop").astype(str)
            _rate_chart(valid, "_tenure_group", tenure_label)
        else:
            st.info("Not enough distinct tenure values to form groups.")
    else:
        st.info("Tenure chart unavailable: the file has no tenure or account-length field.")
chart3, chart4 = st.columns(2)
with chart3:
    if payment:
        _rate_chart(filtered, payment, "Churn by payment method")
    elif len(plan_columns) > 1:
        _rate_chart(filtered, plan_columns[1], "Churn by available service plan")
with chart4:
    charge_column = monthly or "Derived Usage Charge Total"
    charge_label = "Churn by monthly charges" if monthly else "Mean predicted probability by derived usage charge proxy"
    numeric = pd.to_numeric(filtered[charge_column], errors="coerce")
    valid = filtered.loc[numeric.notna()].copy()
    if not valid.empty and numeric.nunique() > 1:
        valid["_charge_group"] = pd.qcut(numeric.loc[valid.index], q=min(5, numeric.nunique()), duplicates="drop").astype(str)
        chart_title = (
            "Observed churn by monthly charge group" if monthly and "Observed_Churn" in valid and valid["Observed_Churn"].notna().any()
            else "Observed churn by derived usage-charge group" if "Observed_Churn" in valid and valid["Observed_Churn"].notna().any()
            else charge_label
        )
        _rate_chart(valid, "_charge_group", chart_title)
        if not monthly:
            st.caption("The source schema has no monthly bill. This grouping uses the model's derived telecom usage-charge total, not a monthly invoice.")
    else:
        st.info("Charge grouping is unavailable for the current data.")
if not contract or not payment:
    unavailable = [name for name, column in (("Contract type", contract), ("Payment method", payment)) if column is None]
    st.caption("Unavailable source fields (not inferred): " + ", ".join(unavailable) + ".")

importance_path = ROOT / "docs" / "phase_2_global_importance.csv"
if importance_path.exists():
    importance = pd.read_csv(importance_path)
    if {"feature", "mean_abs_contribution"}.issubset(importance.columns):
        st.subheader("Model-wide feature importance")
        st.caption("Training-sample global SHAP results from Phase 2; this chart describes the model, not only the filtered upload.")
        st.bar_chart(importance.head(10).set_index("feature")["mean_abs_contribution"], y_label="Mean absolute probability contribution")

st.subheader("Customers to inspect")
sort_choice = st.selectbox("Sort by churn probability", ["Highest first", "Lowest first"])
table_columns = ["Customer ID", "Churn_Probability", "Prediction", "Risk_Category", "Estimated_Input_Count"]
for optional in ("Account length", "International plan", "Voice mail plan", "Observed_Churn"):
    if optional in filtered:
        table_columns.append(optional)
sorted_view = filtered.sort_values("Churn_Probability", ascending=sort_choice == "Lowest first")
st.dataframe(
    sorted_view[table_columns].rename(columns={"Churn_Probability": "Churn probability", "Risk_Category": "Risk category", "Estimated_Input_Count": "Estimated inputs", "Observed_Churn": "Observed churn"}),
    width="stretch",
    hide_index=True,
)
download = sorted_view.to_csv(index=False).encode("utf-8")
st.download_button("Download current filtered results as CSV", data=download, file_name="cram_customer_risk_results.csv", mime="text/csv")

st.subheader("Potential retention actions for review")
st.caption("Transparent rules identify review opportunities for Medium and High Risk customers. These are hypotheses, require human approval, and are not verified churn-prevention outcomes.")
portfolio_recommendations = recommend_portfolio(
    filtered,
    prediction_threshold=portfolio.prediction_threshold,
)
if portfolio_recommendations.empty:
    st.info("No rule-based review suggestions apply to the current filtered customers.")
else:
    counts = portfolio_recommendations["rule_name"].value_counts()
    st.write("Rule categories in the current filtered view")
    st.bar_chart(counts, y_label="Customer recommendations")
    st.caption(f"Suggestions for {portfolio_recommendations['customer_id'].nunique():,} customer(s); open a profile to attach that customer's local SHAP context.")
    recommendation_view = portfolio_recommendations[["customer_id", "title", "priority", "risk_category", "churn_probability", "action"]].copy()
    recommendation_view["churn_probability"] = recommendation_view["churn_probability"].map(lambda value: f"{value:.1%}")
    st.dataframe(recommendation_view.rename(columns={"customer_id": "Customer ID", "title": "Potential action", "priority": "Review priority", "risk_category": "Risk category", "churn_probability": "Churn probability", "action": "Suggested action"}), hide_index=True, width="stretch")
    export = portfolio_recommendations.copy()
    for column in ("triggering_characteristics", "model_context"):
        export[column] = export[column].apply(lambda value: json.dumps(value, ensure_ascii=False))
    st.download_button("Download customer recommendations as CSV", data=export.to_csv(index=False).encode("utf-8"), file_name="cram_customer_recommendations.csv", mime="text/csv")

if not filtered.empty:
    ids = filtered["Customer ID"].astype(str).tolist()
    selected_id = st.selectbox("Open a customer profile", ids)
    selected = filtered.loc[filtered["Customer ID"].astype(str) == selected_id].iloc[0]
    st.subheader(f"Customer profile — {selected_id}")
    detail1, detail2, detail3, detail4 = st.columns(4)
    detail1.metric("Predicted churn probability", f"{selected['Churn_Probability']:.1%}")
    detail2.metric("Binary decision", selected["Prediction"])
    detail3.metric("Risk category", selected["Risk_Category"])
    detail4.metric("Estimated input values", int(selected["Estimated_Input_Count"]))
    if "Observed_Churn" in selected.index and pd.notna(selected["Observed_Churn"]):
        st.write("Observed churn label in uploaded data:", "Churn" if bool(selected["Observed_Churn"]) else "Did not churn")
    profile = {name: selected[name] for name in INPUT_FEATURES}
    try:
        with st.spinner("Preparing the customer's SHAP explanation…"):
            explanation = _cached_profile_explanation(predictor, profile)
    except (ValueError, FileNotFoundError) as exc:
        st.error(f"The customer profile could not be explained: {exc}")
        st.stop()
    factors = pd.DataFrame(explanation["top_contributions"])
    if not factors.empty:
        factors["Contribution (percentage points)"] = factors["contribution"] * 100
        factors["Submitted value"] = factors["value"].astype("string")
        st.dataframe(
            factors[["feature", "Submitted value", "input_status", "direction", "Contribution (percentage points)"]]
            .rename(columns={"feature": "Input", "input_status": "Input status", "direction": "Model contribution direction"}),
            hide_index=True,
            width="stretch",
        )
    st.write(explanation["summary"])
    st.caption("SHAP contributions describe this model's prediction relative to its fixed Phase 2 reference population. They are not causal effects or recommended actions.")
    if explanation["rejected_or_coerced_features"]:
        st.warning("Some submitted values were rejected and imputed: " + ", ".join(explanation["rejected_or_coerced_features"]))
    if explanation["genuinely_missing_features"]:
        st.info("Genuinely blank inputs were imputed: " + ", ".join(explanation["genuinely_missing_features"]))
    if int(selected["Estimated_Input_Count"]) > 3:
        st.warning("This profile has more than three estimated inputs; treat its probability and explanation as especially uncertain.")
    st.markdown("#### Recommendation details")
    recommendations = recommend_customer(
        profile,
        float(selected["Churn_Probability"]),
        str(selected["Risk_Category"]),
        explanation,
        prediction_threshold=portfolio.prediction_threshold,
    )
    if not recommendations:
        st.info("No recommendation rule applies to this customer at the current risk and prediction thresholds.")
    for item in recommendations:
        with st.container(border=True):
            st.markdown(f"**{item['title']} · {item['priority']} review priority**")
            st.write(item["action"])
            st.caption("Why it appeared: " + item["why_triggered"])
            characteristics = ", ".join(f"{key}: {value}" for key, value in item["triggering_characteristics"].items())
            st.caption("Observed characteristics: " + characteristics)
            if item["model_context"]:
                context = "; ".join(f"{row['feature']} {row['direction']} model churn probability ({row['contribution']:+.3f})" for row in item["model_context"])
                st.caption("Model context (SHAP probability contributions): " + context)
            else:
                st.caption("No reliable local SHAP value is available for the triggering features; recommendation is based on the accepted customer characteristic and risk context.")
            st.write("**Business objective:**", item["objective"])
            st.write("**Human review:** Required before any customer contact or account change.")
            st.caption("Limitations: " + item["limitations"] + " SHAP describes model behavior; it does not show that an action will reduce churn.")
    _render_simulation_lab(predictor, str(selected_id), profile)
