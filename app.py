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
      .block-container {max-width: 1440px; padding-top: 1.25rem; padding-bottom: 4rem;}
      header[data-testid="stHeader"] {background: transparent;}
      footer {visibility:hidden;height:0;}
      section[data-testid="stSidebar"] {background:linear-gradient(180deg,color-mix(in srgb,#0f766e 9%,var(--background-color)),var(--background-color));border-right:1px solid color-mix(in srgb,var(--text-color) 10%,transparent);}
      section[data-testid="stSidebar"] > div {padding-top:1.25rem;}
      .side-brand {display:flex;align-items:center;gap:.7rem;padding:.35rem .2rem 1rem;}
      .brand-mark {width:38px;height:38px;display:grid;place-items:center;border-radius:12px;background:#0f766e;color:white;font-size:1.15rem;font-weight:800;}
      .brand-name {font-weight:780;letter-spacing:-.03em;font-size:1rem;}
      .brand-sub {font-size:.68rem;letter-spacing:.09em;opacity:.6;}
      .side-label {font-size:.69rem;font-weight:750;letter-spacing:.12em;opacity:.55;margin:.7rem 0 .3rem;}
      section[data-testid="stSidebar"] div[role="radiogroup"] {gap:.38rem;}
      section[data-testid="stSidebar"] div[role="radiogroup"] label {border-radius:11px;padding:.48rem .65rem;border:1px solid transparent;}
      section[data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked) {background:color-mix(in srgb,#0f766e 15%,var(--secondary-background-color));border-color:color-mix(in srgb,#0f766e 27%,transparent);}
      .page-head {display:flex;align-items:center;justify-content:space-between;gap:1rem;padding:.2rem 0 1rem;border-bottom:1px solid color-mix(in srgb,var(--text-color) 12%,transparent);margin-bottom:1.3rem;}
      .page-eyebrow {font-size:.7rem;letter-spacing:.13em;font-weight:800;color:#0f9f91;margin-bottom:.25rem;}
      .page-title {font-size:clamp(1.55rem,2.6vw,2.1rem);line-height:1.15;letter-spacing:-.04em;font-weight:780;margin:0;}
      .page-copy {font-size:.9rem;opacity:.68;margin:.35rem 0 0;}
      .live-pill {display:inline-flex;align-items:center;gap:.4rem;padding:.38rem .68rem;border-radius:999px;border:1px solid color-mix(in srgb,#14b8a6 35%,transparent);font-size:.73rem;white-space:nowrap;background:color-mix(in srgb,#14b8a6 9%,var(--background-color));}
      .live-dot {width:7px;height:7px;border-radius:50%;background:#14b8a6;}
      h2,h3 {letter-spacing:-.025em;}
      div.stButton > button {border-radius:11px;transition:transform .16s ease,box-shadow .16s ease;}
      div.stButton > button:hover {transform:translateY(-1px);box-shadow:0 6px 16px color-mix(in srgb,var(--text-color) 10%,transparent);}
      div[data-testid="stVerticalBlockBorderWrapper"] {border-radius:17px;}
      [data-testid="stMetric"] {background:var(--secondary-background-color);border:1px solid color-mix(in srgb,var(--text-color) 10%,transparent);padding:.9rem 1rem;border-radius:15px;transition:border-color .18s ease;}
      [data-testid="stMetric"]:hover {border-color:color-mix(in srgb,#14b8a6 50%,transparent);}
      [data-testid="stMetric"] [data-testid="stMetricLabel"], [data-testid="stMetric"] [data-testid="stMetricLabel"] *, [data-testid="stMetric"] [data-testid="stMetricValue"], [data-testid="stMetric"] [data-testid="stMetricValue"] *, [data-testid="stMetric"] [data-testid="stMetricDelta"], [data-testid="stMetric"] [data-testid="stMetricDelta"] * {color:var(--text-color) !important;}
      div[data-testid="stChatMessage"] {border:1px solid color-mix(in srgb,var(--text-color) 10%,transparent);border-radius:17px;padding:.35rem .7rem;}
      div[data-testid="stChatMessageAvatar"] {border-radius:12px!important;border:1px solid color-mix(in srgb,#14b8a6 28%,transparent);box-shadow:0 3px 12px color-mix(in srgb,#14b8a6 12%,transparent);}
      .chat-speaker {display:flex;align-items:center;gap:.42rem;margin:0 0 .22rem;font-size:.69rem;font-weight:800;letter-spacing:.11em;text-transform:uppercase;color:#14b8a6;}
      .chat-presence {width:7px;height:7px;border-radius:50%;background:#14b8a6;box-shadow:0 0 0 3px color-mix(in srgb,#14b8a6 15%,transparent);}
      div[data-testid="stChatInput"] {border:1px solid color-mix(in srgb,#14b8a6 42%,transparent);border-radius:16px;}
      .chat-intro {background:color-mix(in srgb,#14b8a6 8%,var(--secondary-background-color));padding:.85rem 1rem;border-radius:15px;border:1px solid color-mix(in srgb,#14b8a6 20%,transparent);margin-bottom:.8rem;}
      .chat-kicker {color:#14b8a6;font-size:.7rem;font-weight:800;letter-spacing:.1em;text-transform:uppercase;}
      .snapshot-title {font-size:1.05rem;font-weight:750;letter-spacing:-.02em;}
      .snapshot-empty {padding:.9rem;border:1px dashed color-mix(in srgb,var(--text-color) 20%,transparent);border-radius:13px;opacity:.72;font-size:.9rem;}
      .status-dot {display:inline-block;width:7px;height:7px;border-radius:50%;background:#14b8a6;margin-right:.45rem;}
      @media (max-width: 760px) {.block-container{padding-top:.6rem;padding-left:.8rem;padding-right:.8rem}.page-head{align-items:flex-start}.live-pill{font-size:.66rem}}
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
QUESTION_BY_FEATURE = dict(QUESTIONS)
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


def extract_customer_details(text: str, existing: dict) -> tuple[dict, list[str]]:
    """Pick out common customer facts from a natural-language description."""
    profile = dict(existing)
    found = []
    word_numbers = {
        "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4",
        "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9",
        "ten": "10", "eleven": "11", "twelve": "12", "thirteen": "13",
        "fourteen": "14", "fifteen": "15", "sixteen": "16", "seventeen": "17",
        "eighteen": "18", "nineteen": "19", "twenty": "20", "dozen": "12",
        "couple": "2",
    }
    lowered = re.sub(
        r"\b(?:" + "|".join(word_numbers) + r")\b",
        lambda match: word_numbers[match.group()],
        text.lower(),
    )

    def set_value(feature: str, value: float) -> None:
        accepted, parsed, _ = parse_answer(feature, str(value))
        if accepted and parsed is not None:
            profile[feature] = parsed
            found.append(feature)

    plan_phrases = {
        "International plan": ("international",),
        "Voice mail plan": ("voicemail", "voice mail", "voice-mail"),
    }
    for feature, terms in plan_phrases.items():
        for term in terms:
            match = re.search(rf"(?:no|without|doesn't have|doesnt have|not on)\s+(?:an?\s+)?{re.escape(term)}\s+(?:calling\s+)?(?:plan|service)?", lowered)
            positive = re.search(rf"(?:has|have|with|yes|on)\s+(?:an?\s+)?{re.escape(term)}\s+(?:calling\s+)?(?:plan|service)?", lowered)
            if match:
                profile[feature] = 0.0
                found.append(feature)
                break
            if positive:
                profile[feature] = 1.0
                found.append(feature)
                break
            explicit = re.search(rf"{re.escape(term)}\s+(?:calling\s+)?(?:plan|service)\s*(?:is|:)?\s*(yes|no|true|false)", lowered)
            if explicit:
                profile[feature] = float(explicit.group(1) in {"yes", "true"})
                found.append(feature)
                break

    unknown_matches = list(re.finditer(r"not sure|don't know|dont know|unsure|unknown|skip|\bn/?a\b", lowered))
    unknown_features = []
    for unknown in unknown_matches:
        left_boundaries = [lowered.rfind(mark, 0, unknown.start()) for mark in (".", "?", "!", ";", "\n")]
        context_start = max(left_boundaries) + 1
        right_boundaries = [position for mark in (".", "?", "!", ";", "\n") if (position := lowered.find(mark, unknown.end())) >= 0]
        context_end = min(right_boundaries) if right_boundaries else len(lowered)
        context = lowered[context_start:context_end]
        unknown_feature = None
        if re.search(r"international.{0,20}(?:plan|calling plan)|(?:plan|calling plan).{0,20}international", context):
            unknown_feature = "International plan"
        elif re.search(r"voicemail|voice mail|voice-mail", context):
            unknown_feature = "Number vmail messages" if re.search(r"message|msg", context) else "Voice mail plan"
        elif re.search(r"customer\s*(?:service|support)|support team|help desk", context):
            unknown_feature = "Customer service calls"
        elif re.search(r"account|tenure|time with (?:the )?(?:provider|company)|how long", context):
            unknown_feature = "Account length"
        else:
            period_name = None
            for period, pattern in (("intl", r"international|intl"), ("eve", r"evening|\beve\b"), ("night", r"night|overnight"), ("day", r"daytime|during the day|\bday\b")):
                if re.search(pattern, context):
                    period_name = period
                    break
            if period_name:
                if re.search(r"call", context):
                    unknown_feature = f"Total {period_name} calls"
                elif re.search(r"minute|usage|phone use", context):
                    unknown_feature = f"Total {period_name} minutes"
        if unknown_feature in INPUT_FEATURES:
            unknown_features.append(unknown_feature)

    number_tokens = list(re.finditer(r"\b\d+(?:,\d{3})*(?:\.\d+)?\b", lowered))
    for token in number_tokens:
        value = float(token.group().replace(",", ""))
        start, end = token.span()
        tenure_context = lowered[max(0, start - 55): min(len(lowered), end + 30)]
        context_start = max(0, start - 15)
        context = lowered[context_start: min(len(lowered), end + 30)]

        # Tenure is expressed in days, months, or years and gets converted to days.
        duration = re.match(r"\s*(days?|weeks?|months?|mos?|years?|yrs?)\b", lowered[end:])
        if duration and re.search(r"account|customer|tenure|with (?:the )?(?:provider|company|us)|been (?:a |an )?(?:customer|subscriber)", tenure_context):
            unit = duration.group(1)
            multiplier = 7 if unit.startswith("week") else 30 if unit.startswith(("month", "mo")) else 365 if unit.startswith(("year", "yr")) else 1
            set_value("Account length", value * multiplier)
            continue

        if re.search(r"voicemail|voice mail|voice-mail", context) and re.search(r"message|msg", context):
            set_value("Number vmail messages", value)
            continue
        if re.search(r"customer\s*(?:service|support)|support team|help desk", context) and re.search(r"call|contact|time|reach", context):
            set_value("Customer service calls", value)
            continue

        period_patterns = {
            "intl": r"international|intl",
            "eve": r"evening|\beve\b",
            "night": r"night|overnight",
            "day": r"daytime|during the day|\bday\b",
        }
        nearest_period = []
        for period_name, pattern in period_patterns.items():
            for match in re.finditer(pattern, context):
                # Compare keyword position to the number within the local context.
                number_position = start - context_start
                distance = max(match.start() - number_position, number_position - match.end(), 0)
                nearest_period.append((distance, period_name))
        nearest_unit = []
        following_units = []
        for unit_match in re.finditer(r"minutes?|mins?|calls?", context):
            number_position = start - context_start
            distance = max(unit_match.start() - number_position, number_position - unit_match.end(), 0)
            unit_entry = (distance, unit_match.group().lower())
            nearest_unit.append(unit_entry)
            if 0 <= unit_match.start() - number_position <= 24:
                following_units.append((unit_match.start() - number_position, unit_match.group().lower()))
        if not nearest_period or not nearest_unit:
            continue
        period = min(nearest_period)[1]
        unit = min(following_units)[1] if following_units else min(nearest_unit)[1]
        if unit.startswith(("minute", "min")):
            feature = f"Total {period} minutes"
        elif unit.startswith("call"):
            feature = f"Total {period} calls"
        else:
            continue
        if feature in INPUT_FEATURES:
            set_value(feature, value)

    for feature in unknown_features:
        profile[feature] = None
        found.append(feature)

    # Keep the latest occurrence for each fact so a correction in the user's
    # latest message replaces an earlier value.
    latest = {}
    for feature in found:
        latest[feature] = profile[feature]
    profile.update(latest)
    return profile, list(latest)


def reset_chat() -> None:
    st.session_state.chat_messages = [
        ("assistant", "Hey! Tell me about the customer in your own words. You can share a few details at once, like how long they’ve been with the provider, their phone use, plans, or support calls. I’ll pick out what I can and ask only about details I still need. If you’re unsure, just say so."),
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
        avatar = "🧭" if role == "assistant" else "🙂"
        with st.chat_message(role, avatar=avatar):
            if role == "assistant":
                st.markdown("<div class='chat-speaker'><span class='chat-presence'></span>Compass guide</div>", unsafe_allow_html=True)
            st.markdown(message)
    known = {key: value for key, value in st.session_state.chat_profile.items() if value is not None}
    missing = [feature for feature in INPUT_FEATURES if feature not in st.session_state.chat_profile]
    if missing:
        next_feature = missing[0]
        st.markdown(
            "<div class='chat-intro'><div class='chat-kicker'>Your customer interview</div>"
            "<b>Tell me what you know.</b> There’s no special format — share a few details at once, "
            "skip anything you don’t know, and I’ll guide you from there.</div>",
            unsafe_allow_html=True,
        )
        st.markdown("**Try a quick start**")
        quick_starts = [
            ("☎️ Phone use", "They use about 120 minutes during the day and 80 minutes in the evening."),
            ("🧭 Plans & tenure", "They have been with the provider for about 2 years, no international plan, voicemail plan yes."),
            ("💬 Support", "They have called customer support 3 times. I’m not sure about the rest."),
        ]
        cols = st.columns(len(quick_starts))
        for col, (label, message) in zip(cols, quick_starts):
            if col.button(label, key=f"quick_start_{label}", use_container_width=True):
                st.session_state.guided_chat_input = message
                st.rerun()
        bounds = RANGES[next_feature]
        with st.popover("Need help?", use_container_width=False):
            st.markdown(f"### {FIELD_LABELS[next_feature]}")
            st.write(help_for_feature(next_feature))
            st.caption(f"Values in the dataset range from {bounds['min']:g} to {bounds['max']:g}.")
            st.write(f"Example: {example_for_feature(next_feature)}")
            if st.button("Use a typical value", key="use_median_help"):
                st.session_state.chat_profile[next_feature] = None
                st.session_state.chat_messages.append(("assistant", f"Sure — I’ll estimate {FIELD_LABELS[next_feature].lower()} using a typical dataset value and mark it as estimated."))
                st.rerun()
        prompt = "Describe the customer or add another detail…"
    else:
        st.success("That’s enough information to make an estimate. You can still add or correct details in the chat.")
        prompt = "Add or correct a detail, or ask a question…"

    answer = st.chat_input(prompt, key="guided_chat_input")
    if answer:
        st.session_state.chat_messages.append(("user", answer))
        profile, updated = extract_customer_details(answer, st.session_state.chat_profile)
        lowered = answer.lower().strip()
        parse_error = None
        # When asked the current short follow-up, attach the answer to that field.
        unknown_answer = bool(re.fullmatch(r"(?:i'?m )?(?:not sure|unsure|unknown|skip|idk|n/?a|i don't know|dont know|no idea)[.! ]*", lowered))
        if not updated and missing and unknown_answer:
            profile[missing[0]] = None
            updated = [missing[0]]
        elif not updated and missing and re.search(r"not sure|don't know|dont know|unsure|unknown|skip|\bn/?a\b", lowered):
            parse_error = "I hear you. Which detail are you unsure about? Mention the type of detail, like nighttime use, plan, tenure, or support calls, so I flag the right one."
        elif not updated and missing:
            accepted, value, error = parse_answer(missing[0], answer)
            if accepted:
                profile[missing[0]] = value
                updated = [missing[0]]
            else:
                parse_error = error

        st.session_state.chat_profile = profile
        remaining = [feature for feature in INPUT_FEATURES if feature not in profile]
        if updated:
            summary = ", ".join(
                f"{FIELD_LABELS[feature]}: " + ("not sure" if profile[feature] is None else format_value(feature, profile[feature]))
                for feature in updated
            )
            response = f"Got it — I noted {summary}."
        elif parse_error:
            response = parse_error + " Your other details are still saved."
        else:
            response = "I’m not sure which customer detail that refers to yet. Try mentioning a time period like daytime, evening, or international, or tap **Need help?** for examples."
        if remaining:
            response += f"\n\nI still need {len(remaining)} detail(s). Next: {QUESTION_BY_FEATURE[remaining[0]]}"
        else:
            response += "\n\nI have enough details now. I’ll show the estimate below. You can still correct anything by telling me the updated value."
        st.session_state.chat_messages.append(("assistant", response))
        st.rerun()

    if st.session_state.chat_profile and len(st.session_state.chat_profile) == len(INPUT_FEATURES):
        with st.expander("Review details I understood", expanded=True):
            review = []
            for feature in INPUT_FEATURES:
                value = st.session_state.chat_profile[feature]
                review.append({"Customer detail": FIELD_LABELS[feature], "Your information": "Not sure" if value is None else format_value(feature, value)})
            st.dataframe(pd.DataFrame(review), hide_index=True, width="stretch")
            if st.button("Start over", key="restart_chat"):
                reset_chat()
                st.rerun()
        risk_result(st.session_state.chat_profile, "Your churn estimate")


def render_live_snapshot() -> None:
    if "chat_profile" not in st.session_state:
        reset_chat()
    profile = st.session_state.chat_profile
    known = [key for key, value in profile.items() if value is not None]
    estimated = [key for key, value in profile.items() if value is None]
    st.markdown("<div class='snapshot-title'>Customer snapshot</div>", unsafe_allow_html=True)
    st.caption("Updates as you chat")
    st.progress(len(profile) / len(INPUT_FEATURES), text=f"{len(profile)} / {len(INPUT_FEATURES)} details")
    if not profile:
        st.markdown("<div class='snapshot-empty'>✨ Your customer profile will take shape here as you share details.</div>", unsafe_allow_html=True)
    else:
        for feature in INPUT_FEATURES:
            if feature not in profile:
                continue
            value = profile[feature]
            detail = "Estimated with a typical value" if value is None else format_value(feature, value)
            st.markdown(f"<span class='status-dot'></span><b>{FIELD_LABELS[feature]}</b><br><span style='opacity:.68'>{detail}</span>", unsafe_allow_html=True)
            st.write("")
    if estimated:
        st.caption(f"{len(estimated)} detail(s) will use typical values and be marked in the result.")
    with st.expander("How this works"):
        st.write("Share what you know in everyday language. The assistant maps it to the model’s inputs, asks about missing details, and shows what it understood before scoring.")
        st.caption("Research demo · results are estimates, not guarantees.")


def help_for_feature(feature: str) -> str:
    descriptions = {
        "Account length": "How long the customer has been with this phone provider. You can answer in days, months, or years.",
        "International plan": "Whether international calling is included in the customer's plan.",
        "Voice mail plan": "Whether the customer has the voicemail add-on.",
        "Number vmail messages": "The number of voicemail messages in a typical month.",
        "Customer service calls": "How many calls the customer has made to support.",
    }
    return descriptions.get(feature, "A rough monthly estimate is fine. You can include a number and say whether it refers to minutes or calls.")


def example_for_feature(feature: str) -> str:
    examples = {
        "Account length": "They have been with the provider for about 6 months.",
        "International plan": "They do not have international calling.",
        "Voice mail plan": "Voicemail is included in their plan.",
        "Number vmail messages": "They get around 8 voicemail messages each month.",
        "Customer service calls": "They have called support 3 times.",
    }
    return examples.get(feature, "About 120 minutes of daytime phone use each month.")


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


with st.sidebar:
    st.markdown(
        "<div class='side-brand'><div class='brand-mark'>✦</div><div><div class='brand-name'>Churn Compass</div><div class='brand-sub'>CUSTOMER INTELLIGENCE</div></div></div>",
        unsafe_allow_html=True,
    )
    st.markdown("<div class='side-label'>WORKSPACE</div>", unsafe_allow_html=True)
    selected_page = st.radio(
        "Workspace",
        ["Predict", "Batch scoring", "Model performance", "Project info"],
        label_visibility="collapsed",
        key="workspace_navigation",
    )
    st.divider()
    st.markdown("<div class='side-label'>ABOUT THIS DEMO</div>", unsafe_allow_html=True)
    st.caption("A research prototype for exploring telecom customer churn.")
    st.caption(f"{metadata['dataset']['rows']:,} source records · 13 customer details")

page_details = {
    "Predict": ("CUSTOMER WORKSPACE", "Customer risk", "Describe a customer in your own words and explore an estimated churn score."),
    "Batch scoring": ("BATCH WORKSPACE", "Score a customer list", "Upload a CSV to explore predictions across multiple customer profiles."),
    "Model performance": ("MODEL TRANSPARENCY", "How the model performs", "Review held-out results, missing-detail sensitivity, and the patterns the model uses."),
    "Project info": ("PROJECT NOTES", "About this research", "Explore the data, approach, and intended limits of this project."),
}
eyebrow, title, subtitle = page_details[selected_page]
st.markdown(
    f"<div class='page-head'><div><div class='page-eyebrow'>{eyebrow}</div><h1 class='page-title'>{title}</h1><p class='page-copy'>{subtitle}</p></div><div class='live-pill'><span class='live-dot'></span> Research preview</div></div>",
    unsafe_allow_html=True,
)

if selected_page == "Predict":
    mode = st.radio("Choose your starting point", ["Chat with Compass", "Quick form"], horizontal=True, label_visibility="collapsed")
    if mode == "Chat with Compass":
        chat_col, snapshot_col = st.columns([1.8, 0.82], gap="large")
        with chat_col:
            render_guided_chat()
        with snapshot_col:
            with st.container(border=True):
                render_live_snapshot()
    else:
        form_col, tips_col = st.columns([1.55, 1], gap="large")
        with form_col:
            with st.container(border=True):
                render_quick_form()
        with tips_col:
            with st.container(border=True):
                st.markdown("### A helpful starting point")
                st.write("A rough estimate is okay. If you leave a detail blank, the model uses a typical value from its training data and marks that estimate in the result.")
                st.info("This prototype is trained on telecom customer records. Scores are estimates for research and exploration.")
elif selected_page == "Batch scoring":
    render_batch()
elif selected_page == "Model performance":
    render_model_evaluation()
else:
    render_project_overview()
