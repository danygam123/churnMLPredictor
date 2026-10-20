from __future__ import annotations

import json
import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from streamlit.errors import StreamlitAPIException
from model_features import FEATURES as EXPECTED_FEATURES
from model_features import INPUT_FEATURES as EXPECTED_INPUT_FEATURES
from model_features import build_model_matrix


ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "random_forest_model.joblib"
METADATA_PATH = ROOT / "model_metadata.json"
PAPER_PATH = ROOT / "Paper draft (2).pdf"

st.set_page_config(
    page_title="Churn Compass | Customer Retention Research",
    page_icon="📡",
    layout="wide",
)

st.markdown(
    """
    <style>
      .block-container {max-width: 1180px; padding-top: 2.2rem; padding-bottom: 3rem;}
      [data-testid="stMetric"] {background: #f4f7fb; border: 1px solid #e5eaf2;
        padding: 1rem 1.1rem; border-radius: 14px;}
      div[data-testid="stChatMessage"] {border: 1px solid #e7ebf1; border-radius: 14px;}
      .small-note {color: #617082; font-size: 0.92rem;}
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_assets():
    model = joblib.load(MODEL_PATH)
    metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    if list(model.feature_names_in_) != metadata["features"]:
        raise ValueError("The model feature list does not match model_metadata.json.")
    return model, metadata


model, metadata = load_assets()
INPUT_FEATURES = metadata["input_features"]
FEATURES = metadata["features"]
MEDIANS = metadata["served_model"]["imputation_values"]
RANGES = metadata["input_ranges"]
METRICS = metadata["evaluation_model"]["test_metrics"]
if INPUT_FEATURES != EXPECTED_INPUT_FEATURES or FEATURES != EXPECTED_FEATURES:
    raise ValueError("The shared feature schema does not match model_metadata.json.")

FIELD_LABELS = {
    "Account length": "Time with provider (days)",
    "International plan": "International calling plan",
    "Voice mail plan": "Voicemail plan",
    "Number vmail messages": "Voicemail messages",
    "Total day minutes": "Daytime phone use (minutes)",
    "Total day calls": "Daytime calls",
    "Total eve minutes": "Evening phone use (minutes)",
    "Total eve calls": "Evening calls",
    "Total night minutes": "Nighttime phone use (minutes)",
    "Total night calls": "Nighttime calls",
    "Total intl minutes": "International phone use (minutes)",
    "Total intl calls": "International calls",
    "Customer service calls": "Calls to customer support",
    "Total day charge": "Daytime charges",
    "Total eve charge": "Evening charges",
    "Total night charge": "Nighttime charges",
    "Total intl charge": "International charges",
}

QUESTIONS = [
    ("Account length", "About how many days has this customer been with their provider?"),
    ("International plan", "Do they have an international calling plan? Answer yes or no."),
    ("Voice mail plan", "Do they have a voicemail plan? Answer yes or no."),
    ("Number vmail messages", "About how many voicemail messages do they get in a typical month?"),
    ("Total day minutes", "Roughly how many minutes do they use their phone during the day in a month?"),
    ("Total day calls", "About how many daytime calls do they make in a month?"),
    ("Total eve minutes", "Roughly how many minutes do they use their phone in the evening in a month?"),
    ("Total eve calls", "About how many evening calls do they make in a month?"),
    ("Total night minutes", "Roughly how many minutes do they use their phone at night in a month?"),
    ("Total night calls", "About how many nighttime calls do they make in a month?"),
    ("Total intl minutes", "About how many international minutes do they use in a month?"),
    ("Total intl calls", "About how many international calls do they make in a month?"),
    ("Customer service calls", "How many times has this customer called customer support?"),
]
INTEGER_FIELDS = {
    "Account length",
    "Number vmail messages",
    "Total day calls",
    "Total eve calls",
    "Total night calls",
    "Total intl calls",
    "Customer service calls",
}


def make_model_matrix(profile: dict) -> tuple[pd.DataFrame, list[str]]:
    """Convert user facts to model columns and fill unknowns transparently."""
    normalized = dict(profile)
    missing = []
    for feature in INPUT_FEATURES:
        value = normalized.get(feature)
        if value is None or pd.isna(value):
            missing.append(feature)
            normalized[feature] = np.nan
        elif feature in INTEGER_FIELDS:
            normalized[feature] = round(float(value))
    return build_model_matrix(normalized, MEDIANS), missing


def normalize_uploaded_data(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Prepare a customer CSV. Missing or unrecognized values remain explicit."""
    original = data.copy()
    facts = pd.DataFrame(index=data.index)
    warnings = []
    binary = {"International plan", "Voice mail plan"}
    for feature in INPUT_FEATURES:
        if feature not in data.columns:
            facts[feature] = np.nan
            continue
        values = data[feature]
        if feature in binary:
            mapping = {
                "yes": 1, "y": 1, "true": 1, "1": 1,
                "no": 0, "n": 0, "false": 0, "0": 0,
                "": np.nan, "not sure": np.nan, "unknown": np.nan,
            }
            normalized = values.astype(str).str.strip().str.lower().map(mapping)
            numeric = pd.to_numeric(values, errors="coerce")
            facts[feature] = normalized.where(normalized.notna(), numeric)
            invalid = values.notna() & facts[feature].isna()
            if invalid.any():
                warnings.append(f"{feature}: {int(invalid.sum())} unrecognized value(s) treated as unknown")
        else:
            facts[feature] = pd.to_numeric(values, errors="coerce")
            invalid = values.notna() & facts[feature].isna()
            if invalid.any():
                warnings.append(f"{feature}: {int(invalid.sum())} non-numeric value(s) treated as unknown")
        low, high = RANGES[feature]["min"], RANGES[feature]["max"]
        out_of_range = facts[feature].notna() & ((facts[feature] < low) | (facts[feature] > high))
        if out_of_range.any():
            warnings.append(f"{feature}: {int(out_of_range.sum())} value(s) outside the dataset range treated as unknown")
            facts.loc[out_of_range, feature] = np.nan

    model_rows = []
    missing_counts = facts[INPUT_FEATURES].isna().sum(axis=1)
    for _, values in facts.iterrows():
        row, _ = make_model_matrix(values.to_dict())
        model_rows.append(row.iloc[0])
    model_frame = pd.DataFrame(model_rows, columns=FEATURES, index=data.index)
    result = original.copy()
    probabilities = model.predict_proba(model_frame)[:, 1]
    result["Churn risk score"] = probabilities
    result["Prediction"] = np.where(probabilities >= 0.5, "Likely to churn", "Less likely to churn")
    result["Unknown input count"] = missing_counts
    return result, model_frame, warnings


def format_value(feature: str, value) -> str:
    if feature in {"International plan", "Voice mail plan"}:
        return "Yes" if int(value) == 1 else "No"
    if feature in INTEGER_FIELDS:
        return f"{int(round(float(value))):,}"
    return f"{float(value):,.1f}"


def parse_answer(feature: str, answer: str) -> tuple[bool, float | None, str | None]:
    text = answer.strip().lower()
    unknown_markers = ("not sure", "unknown", "skip", "don't know", "dont know", "i don't know", "idk", "n/a", "unsure")
    if any(marker in text for marker in unknown_markers) or text in {"?", "no idea"}:
        return True, None, None

    if feature in {"International plan", "Voice mail plan"}:
        if re.search(r"\b(no|nope|nah|n|not|doesn't|doesnt|without|don't|dont)\b", text):
            return True, 0.0, None
        if re.search(r"\b(yes|yeah|yep|y|have|has|included|subscribed)\b", text):
            return True, 1.0, None
        return False, None, "Please answer yes, no, or 'not sure'."

    if feature == "Account length":
        duration = re.search(r"(\d+(?:\.\d+)?)\s*(years?|yrs?|months?|mos?|days?)\b", text)
        if duration:
            amount = float(duration.group(1))
            unit = duration.group(2)
            if unit.startswith(("year", "yr")):
                value = amount * 365
            elif unit.startswith(("month", "mo")):
                value = amount * 30
            else:
                value = amount
            low, high = RANGES[feature]["min"], RANGES[feature]["max"]
            if value < low or value > high:
                return False, None, f"After converting that to days, it falls outside this demo's range of {low:g} to {high:g}. Enter days in that range or say 'not sure'."
            return True, float(round(value)), None

    range_match = re.search(r"(\d+(?:\.\d+)?)\s*(?:-|to)\s*(\d+(?:\.\d+)?)", text)
    if range_match:
        value = (float(range_match.group(1)) + float(range_match.group(2))) / 2
    else:
        numbers = re.findall(r"\d+(?:\.\d+)?", text.replace(",", ""))
        value = float(numbers[0]) if numbers else None
    if value is None:
        return False, None, "Please enter a number, a rough estimate, or 'not sure'."
    low, high = RANGES[feature]["min"], RANGES[feature]["max"]
    if value < low or value > high:
        return False, None, f"For this demo, the data ranges from {low:g} to {high:g}. Enter a value in that range or say 'not sure'."
    if feature in INTEGER_FIELDS:
        value = float(round(value))
    return True, value, None


def reset_chat() -> None:
    st.session_state.chat_messages = [
        ("assistant", "Hi! I’ll help estimate churn risk using the information you have. I’ll ask one everyday question at a time. If you don’t know something, say “not sure”; I’ll use a typical value and flag that in the result."),
    ]
    st.session_state.chat_profile = {}
    st.session_state.chat_step = 0
    st.session_state.chat_error = None


def risk_result(profile: dict, heading: str = "Prediction") -> None:
    model_row, missing = make_model_matrix(profile)
    score = float(model.predict_proba(model_row)[0, 1])
    predicted_churn = score >= 0.5
    st.subheader(heading)
    if predicted_churn:
        st.warning("The model flags this profile as more likely to churn at the default 50% cutoff.")
    else:
        st.success("The model does not flag this profile as likely to churn at the default 50% cutoff.")
    left, right = st.columns(2)
    left.metric("Model churn score", f"{score:.1%}")
    right.metric("Fields estimated", f"{len(missing)} of {len(INPUT_FEATURES)}")
    if missing:
        labels = ", ".join(FIELD_LABELS[item] for item in missing)
        st.info(
            f"I used typical dataset values for: {labels}. This makes the result a rougher estimate; "
            "adding those details may change the score."
        )
    st.caption("This score is a model estimate from a small telecom research dataset, not a guarantee or a reason to make a customer decision by itself.")
    if len(missing) >= 3:
        sensitivity = METRICS.get("median_imputation_sensitivity", {}).get(str(len(missing)))
        if sensitivity:
            st.warning(
                f"In a test where {len(missing)} randomly selected details were hidden, "
                f"average churn recall was {sensitivity['churn_recall']['mean']:.0%}. "
                "This estimate may be unreliable with several unknown details."
            )

    with st.expander("What shifted the model score?", expanded=True):
        median_profile = {name: MEDIANS[name] for name in INPUT_FEATURES}
        median_row, _ = make_model_matrix(median_profile)
        typical_score = float(model.predict_proba(median_row)[0, 1])
        sensitivities = []
        for feature in INPUT_FEATURES:
            if feature in missing:
                continue
            changed = dict(profile)
            changed[feature] = MEDIANS[feature]
            changed_row, _ = make_model_matrix(changed)
            changed_score = float(model.predict_proba(changed_row)[0, 1])
            delta = score - changed_score
            sensitivities.append({
                "Customer detail": FIELD_LABELS[feature],
                "Entered value": format_value(feature, profile[feature]),
                "Score change vs. typical value": f"{delta:+.1%}",
                "Typical profile score": f"{typical_score:.1%}",
            })
        if sensitivities:
            table = pd.DataFrame(sensitivities)
            table["_magnitude"] = table["Score change vs. typical value"].str.rstrip("%").astype(float).abs()
            table = table.sort_values("_magnitude", ascending=False).drop(columns="_magnitude").head(5)
            st.dataframe(table, hide_index=True, width="stretch")
        st.caption("Each comparison changes one detail to its typical dataset value. These are model sensitivities, not causes of churn.")


def render_guided_chat() -> None:
    if "chat_messages" not in st.session_state:
        reset_chat()
    for role, message in st.session_state.chat_messages:
        with st.chat_message(role):
            st.markdown(message)
    if st.session_state.chat_error:
        st.warning(st.session_state.chat_error)

    step = st.session_state.chat_step
    if step < len(QUESTIONS):
        feature, question = QUESTIONS[step]
        with st.chat_message("assistant"):
            st.markdown(f"**{question}**")
            if feature in RANGES:
                value_range = RANGES[feature]
                st.caption(f"In this dataset, values range from {value_range['min']:g} to {value_range['max']:g}.")
        answer = st.chat_input("Type an answer, a rough estimate, or 'not sure'", key="guided_chat_input")
        if answer:
            st.session_state.chat_messages.append(("user", answer))
            accepted, value, error = parse_answer(feature, answer)
            if not accepted:
                st.session_state.chat_error = error
            else:
                st.session_state.chat_error = None
                st.session_state.chat_profile[feature] = value
                st.session_state.chat_step += 1
                if value is None:
                    reply = f"No problem. I’ll use a typical value for {FIELD_LABELS[feature].lower()} and mark it as estimated."
                else:
                    reply = f"Got it — {FIELD_LABELS[feature].lower()}: {format_value(feature, value)}."
                st.session_state.chat_messages.append(("assistant", reply))
                if st.session_state.chat_step >= len(QUESTIONS):
                    st.session_state.chat_messages.append(("assistant", "Thanks. I have enough to estimate a churn score."))
                st.rerun()
    else:
        risk_result(st.session_state.chat_profile, "Customer churn estimate")
        if st.button("Start a new customer interview", key="restart_chat"):
            reset_chat()
            st.rerun()


def render_quick_form() -> None:
    st.write("Enter what you know. For any detail you cannot provide, select **Not sure**; the result will clearly identify estimated fields.")
    profile = {}
    account, plan = st.columns(2)
    with account:
        profile["Account length"] = numeric_widget("Account length", "About how many days have they been with the provider?")
    with plan:
        profile["International plan"] = categorical_widget("International plan")
        profile["Voice mail plan"] = categorical_widget("Voice mail plan")
        profile["Number vmail messages"] = numeric_widget("Number vmail messages", "Voicemail messages in a typical month")

    st.markdown("#### Phone use in a typical month")
    c1, c2, c3 = st.columns(3)
    for column, prefix in zip((c1, c2, c3), ("day", "eve", "night")):
        time_name = {"day": "daytime", "eve": "evening", "night": "nighttime"}[prefix]
        with column:
            st.markdown(f"**{time_name.title()}**")
            profile[f"Total {prefix} minutes"] = numeric_widget(f"Total {prefix} minutes", "Minutes used")
            profile[f"Total {prefix} calls"] = numeric_widget(f"Total {prefix} calls", "Number of calls")
    intl, support = st.columns(2)
    with intl:
        st.markdown("#### International use")
        profile["Total intl minutes"] = numeric_widget("Total intl minutes", "International minutes per month")
        profile["Total intl calls"] = numeric_widget("Total intl calls", "International calls per month")
    with support:
        st.markdown("#### Customer support")
        profile["Customer service calls"] = numeric_widget("Customer service calls", "Customer support calls in this record")

    st.caption("Usage charges are calculated from minutes using the rates present in the research dataset.")
    if st.button("Estimate churn risk", type="primary", key="quick_predict"):
        st.session_state.quick_profile = profile
    if "quick_profile" in st.session_state:
        risk_result(st.session_state.quick_profile, "Customer churn estimate")


def numeric_widget(feature: str, help_text: str) -> float | None:
    bounds = RANGES[feature]
    median = MEDIANS[feature]
    value = st.number_input(
        FIELD_LABELS[feature],
        min_value=float(bounds["min"]),
        max_value=float(bounds["max"]),
        value=float(median),
        step=1.0 if feature in INTEGER_FIELDS else 0.1,
        help=help_text,
        key=f"input_{feature}",
    )
    unknown = st.checkbox("Not sure", key=f"unknown_{feature}")
    return None if unknown else value


def categorical_widget(feature: str) -> float | None:
    answer = st.selectbox(
        FIELD_LABELS[feature],
        ["No", "Yes", "Not sure"],
        key=f"input_{feature}",
    )
    if answer == "Not sure":
        return None
    return float(answer == "Yes")


def render_batch() -> None:
    st.write("Upload a CSV with customer details. Use the plain-language column names in the template; leave unknown cells blank. Usage charges are calculated from minutes.")
    template = pd.DataFrame(
        [{feature: MEDIANS[feature] for feature in INPUT_FEATURES}],
        columns=INPUT_FEATURES,
    )
    for feature in ("International plan", "Voice mail plan"):
        template[feature] = template[feature].map({0.0: "No", 1.0: "Yes"}).fillna("No")
    st.download_button(
        "Download CSV template",
        template.to_csv(index=False).encode("utf-8"),
        file_name="churn_customer_template.csv",
        mime="text/csv",
    )
    upload = st.file_uploader("Choose a customer CSV", type=["csv"], key="batch_upload")
    if upload is None:
        st.info("Your file is processed in this app session. It is not added to the project dataset.")
        return
    try:
        uploaded = pd.read_csv(upload)
    except Exception as exc:
        st.error(f"I could not read this CSV: {exc}")
        return
    if uploaded.empty:
        st.error("The file has column headers but no customer rows.")
        return
    if len(uploaded) > 10000:
        st.error("This demo accepts up to 10,000 customer rows per upload.")
        return
    results, _, warnings = normalize_uploaded_data(uploaded)
    for warning in warnings:
        st.warning(warning)
    missing_inputs = int(results["Unknown input count"].gt(0).sum())
    if missing_inputs:
        st.info(f"{missing_inputs:,} row(s) have at least one missing or estimated input. Typical dataset values were used for those fields.")
    st.dataframe(results.head(100), width="stretch")
    st.download_button(
        "Download predictions",
        results.to_csv(index=False).encode("utf-8"),
        file_name="churn_predictions.csv",
        mime="text/csv",
        type="primary",
    )
    if len(results) > 100:
        st.caption(f"Showing the first 100 of {len(results):,} customers. The download includes all rows.")


def render_model_evaluation() -> None:
    st.subheader("Held-out evaluation")
    st.write(
        f"The model was trained on {metadata['dataset']['rows']:,} telecom customer records. "
        f"Evaluation used a stratified {metadata['split']['test_size']:.0%} test split "
        f"({metadata['split']['test_rows']:,} customers) that was held apart from model fitting."
    )
    c1, c2, c3 = st.columns(3)
    c1.metric("Accuracy", f"{METRICS['accuracy']:.1%}")
    c2.metric("Churn recall", f"{METRICS['churn_recall']:.1%}")
    c3.metric("Churn F1", f"{METRICS['churn_f1']:.1%}")
    c1, c2, c3 = st.columns(3)
    c1.metric("Churn precision", f"{METRICS['churn_precision']:.1%}")
    c2.metric("ROC-AUC", f"{METRICS['roc_auc']:.1%}")
    c3.metric("PR-AUC", f"{METRICS['pr_auc_average_precision']:.1%}")
    st.caption(
        f"Churn is {metadata['dataset']['churn_rate']:.1%} of the source data. "
        f"A majority-class baseline reaches {metadata['baseline']['test_accuracy']:.1%} accuracy "
        "but detects no churners."
    )
    st.write("Confusion matrix — actual class by predicted class")
    st.dataframe(
        pd.DataFrame(
            METRICS["confusion_matrix_labels_0_1"],
            index=["Actual: no churn", "Actual: churn"],
            columns=["Predicted: no churn", "Predicted: churn"],
        ),
        width="stretch",
    )

    st.subheader("When some details are unknown")
    st.write("The following check hides a random number of details from each test customer, fills them with training-set medians, and repeats with five different masks.")
    sensitivity = metadata["evaluation_model"].get("median_imputation_sensitivity", {})
    for count, results in sensitivity.items():
        st.markdown(f"**{count} unknown detail(s) per profile**")
        a, b, c = st.columns(3)
        a.metric("Churn recall", f"{results['churn_recall']['mean']:.1%}", f"±{results['churn_recall']['std']:.1%}")
        b.metric("Churn F1", f"{results['churn_f1']['mean']:.1%}", f"±{results['churn_f1']['std']:.1%}")
        c.metric("Accuracy", f"{results['accuracy']['mean']:.1%}", f"±{results['accuracy']['std']:.1%}")

    st.subheader("Most-used model signals")
    combined_importance = {
        name: float(model.feature_importances_[FEATURES.index(name)])
        for name in FEATURES
    }
    for charge, formula in metadata["charge_formulas"].items():
        minutes = formula["minutes_feature"]
        combined_importance[minutes] += combined_importance[charge]
        del combined_importance[charge]
    importance = pd.DataFrame({
        "Customer detail": [FIELD_LABELS[name] for name in INPUT_FEATURES],
        "Relative model importance": [combined_importance[name] for name in INPUT_FEATURES],
    }).sort_values("Relative model importance", ascending=False)
    st.bar_chart(importance.set_index("Customer detail"))
    st.caption("Derived charge importance is grouped with its corresponding usage minutes. This describes model patterns, not causes of churn.")
    st.caption("Results apply to this telecom dataset. They do not establish performance for another company, industry, or customer population.")


def render_project_overview() -> None:
    st.subheader("Project overview")
    st.write("This research project explores whether a telecom customer is likely to churn using account tenure, service plans, calling patterns, and support contacts.")
    c1, c2, c3 = st.columns(3)
    c1.metric("Source records", f"{metadata['dataset']['rows']:,}")
    c2.metric("Churned customers", f"{metadata['dataset']['churn_rows']:,}")
    c3.metric("Input details", f"{len(INPUT_FEATURES)}")
    st.markdown("#### How to use the demo")
    st.markdown("- **Guided chat:** answer simple questions; skip details you do not know.\n- **Quick form:** enter a profile directly.\n- **Batch CSV:** download the template, add rows, upload it, and download results.")
    st.markdown("#### About the model")
    st.write("The predictor is a Random Forest trained on a public telecom churn dataset. Its usage-charge fields are calculated from minutes using the same rates found in that dataset. Unknown values are filled with typical training-data values and clearly marked in the result.")
    st.warning("This is an educational research prototype. Do not use its scores as the sole basis for contacting, pricing, or treating real customers.")
    with st.expander("Project paper"):
        try:
            st.pdf(PAPER_PATH.read_bytes())
        except (StreamlitAPIException, OSError):
            st.info("The inline PDF viewer is unavailable here. You can download the paper instead.")
            if PAPER_PATH.exists():
                st.download_button(
                    "Download project paper",
                    PAPER_PATH.read_bytes(),
                    file_name="Customer_Churn_Project_Paper.pdf",
                    mime="application/pdf",
                )


st.title("Churn Compass")
st.write("Explore telecom churn risk with a guided customer interview, direct entry, or batch predictions.")
st.caption("Research demo · Telecom customer data · Model scores are estimates")

predict_tab, batch_tab, evaluation_tab, overview_tab = st.tabs(
    ["Predict a customer", "Batch predictions", "Model & evaluation", "Project overview"]
)

with predict_tab:
    mode = st.radio("How would you like to enter customer details?", ["Guided chat", "Quick form"], horizontal=True)
    if mode == "Guided chat":
        st.caption("This guided interview uses a fixed set of conversational follow-up questions. No customer details are sent to an external chatbot service.")
        render_guided_chat()
    else:
        render_quick_form()

with batch_tab:
    render_batch()

with evaluation_tab:
    render_model_evaluation()

with overview_tab:
    render_project_overview()
