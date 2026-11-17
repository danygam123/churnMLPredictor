"""Streamlit smoke test for the V2 customer-risk workflow."""

from io import StringIO

import pandas as pd
from streamlit.testing.v1 import AppTest

from ml.preprocessing import INPUT_FEATURES


def _csv_bytes():
    rows = []
    for i in range(4):
        rows.append({
            "Customer ID": f"C-{i + 1}",
            "Account length": 80 + i * 20,
            "International plan": "Yes" if i % 2 else "No",
            "Voice mail plan": "No" if i % 2 else "Yes",
            "Number vmail messages": 10 + i,
            "Total day minutes": 120.0 + i * 30,
            "Total day calls": 90 + i,
            "Total eve minutes": 150.0 + i * 10,
            "Total eve calls": 80 + i,
            "Total night minutes": 180.0 + i * 10,
            "Total night calls": 90 + i,
            "Total intl minutes": 5.0 + i,
            "Total intl calls": 2 + i,
            "Customer service calls": i,
            "Churn": [False, True, False, True][i],
        })
    frame = pd.DataFrame(rows)
    assert all(name in frame.columns for name in INPUT_FEATURES)
    buffer = StringIO()
    frame.to_csv(buffer, index=False)
    return buffer.getvalue().encode("utf-8")


def test_v2_app_upload_renders_dashboard_table_and_customer_profile():
    app = AppTest.from_file("v2_app.py").run(timeout=30)
    app.get("file_uploader")[0].upload("sample_portfolio.csv", _csv_bytes(), "text/csv")
    app.run(timeout=60)

    assert len(app.exception) == 0
    assert len(app.metric) >= 8  # five portfolio metrics plus three profile metrics
    assert any("Open a customer profile" in selectbox.label for selectbox in app.selectbox)
    assert any("Customer profile" in heading.value for heading in app.subheader)
    assert any("Potential retention actions for review" in heading.value for heading in app.subheader)
    assert any("What-If Simulation Lab" in heading.value for heading in app.subheader)


def test_v2_app_without_upload_shows_the_upload_prompt_without_errors():
    app = AppTest.from_file("v2_app.py").run(timeout=30)
    assert len(app.exception) == 0
    assert any("Upload a customer CSV" in item.value for item in app.info)


def test_v2_simulation_scenarios_are_independent_editable_resettable_and_removable():
    app = AppTest.from_file("v2_app.py").run(timeout=30)
    app.get("file_uploader")[0].upload("sample_portfolio.csv", _csv_bytes(), "text/csv")
    app.run(timeout=60)
    assert any("not causal estimates" in warning.value for warning in app.warning)
    explanation_signature = app.session_state["cram_selected_profile_explanation"]["signature"]
    explanation_signature = app.session_state["cram_selected_profile_explanation"]["signature"]

    next(button for button in app.button if button.label == "Create scenario").click().run(timeout=60)
    next(widget for widget in app.text_input if widget.label == "Scenario name").set_value("Support review case").run(timeout=60)
    first_features = next(widget for widget in app.multiselect if widget.label == "Attributes to vary")
    first_features.set_value(["Customer service calls"]).run(timeout=60)
    first_calls = next(widget for widget in app.slider if widget.label == "Customer service calls")
    first_calls.set_value(3).run(timeout=60)
    assert len(app.exception) == 0
    assert len([widget for widget in app.slider if widget.label == "Customer service calls"]) == 1

    next(button for button in app.button if button.label == "Create scenario").click().run(timeout=60)
    feature_selectors = [widget for widget in app.multiselect if widget.label == "Attributes to vary"]
    assert len(feature_selectors) == 2
    feature_selectors[1].set_value(["Total intl minutes"]).run(timeout=60)
    intl_sliders = [widget for widget in app.slider if widget.label == "Total intl minutes"]
    assert len(intl_sliders) == 1
    intl_sliders[0].set_value(15.0).run(timeout=60)

    # Editing A again keeps B's separate widget state and both comparisons render.
    call_sliders = [widget for widget in app.slider if widget.label == "Customer service calls"]
    call_sliders[0].set_value(5).run(timeout=60)
    feature_selectors = [widget for widget in app.multiselect if widget.label == "Attributes to vary"]
    assert feature_selectors[0].value == ["Customer service calls"]
    assert feature_selectors[1].value == ["Total intl minutes"]
    assert [widget.value for widget in app.slider if widget.label == "Total intl minutes"] == [15.0]
    assert any("Support review case:" in caption.value and "percentage points" in caption.value for caption in app.caption)
    assert any("Scenario B:" in caption.value and "percentage points" in caption.value for caption in app.caption)
    comparison = next(dataframe.value for dataframe in app.dataframe if "Scenario" in dataframe.value.columns)
    assert set(comparison["Scenario"]) == {"Baseline", "Support review case", "Scenario B"}
    assert "Customer service calls:" in comparison.loc[comparison["Scenario"] == "Support review case", "Changed inputs"].iloc[0]
    assert app.session_state["cram_selected_profile_explanation"]["signature"] == explanation_signature
    assert app.session_state["cram_selected_profile_explanation"]["signature"] == explanation_signature

    next(button for button in app.button if button.label == "Create scenario").click().run(timeout=60)
    feature_selectors = [widget for widget in app.multiselect if widget.label == "Attributes to vary"]
    assert len(feature_selectors) == 3
    assert next(button for button in app.button if button.label == "Create scenario").disabled
    comparison = next(dataframe.value for dataframe in app.dataframe if "Scenario" in dataframe.value.columns)
    assert set(comparison["Scenario"]) == {"Baseline", "Support review case", "Scenario B", "Scenario C"}
    comparison = next(dataframe.value for dataframe in app.dataframe if "Scenario" in dataframe.value.columns)
    assert set(comparison["Scenario"]) == {"Baseline", "Support review case", "Scenario B", "Scenario C"}

    reset = next(button for button in app.button if button.label == "Reset scenario")
    reset.click().run(timeout=60)
    feature_selectors = [widget for widget in app.multiselect if widget.label == "Attributes to vary"]
    assert feature_selectors[0].value == []
    assert feature_selectors[1].value == ["Total intl minutes"]
    assert feature_selectors[2].value == []
    assert len(app.exception) == 0

    remove_buttons = [button for button in app.button if button.label == "Remove scenario"]
    remove = remove_buttons[1]  # Remove Scenario B while preserving A and C.
    remove.click().run(timeout=60)
    assert len([widget for widget in app.multiselect if widget.label == "Attributes to vary"]) == 2
    assert len(app.exception) == 0

    customer_select = next(widget for widget in app.selectbox if widget.label == "Open a customer profile")
    customer_select.set_value("C-2").run(timeout=60)
    assert any("Customer profile — C-2" in heading.value for heading in app.subheader)
    assert any("Selected customer: C-2" in caption.value for caption in app.caption)
    assert not [widget for widget in app.multiselect if widget.label == "Attributes to vary"]
    assert len(app.exception) == 0


# ---------------------------------------------------------------------------
# Phase 6 — AI Business Analyst
# ---------------------------------------------------------------------------


def test_v2_app_analyst_answers_with_computed_facts_when_no_api_key(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    app = AppTest.from_file("v2_app.py").run(timeout=30)
    app.get("file_uploader")[0].upload("sample_portfolio.csv", _csv_bytes(), "text/csv")
    app.run(timeout=60)

    assert len(app.exception) == 0
    assert any("AI Business Analyst" in heading.value for heading in app.subheader)
    # Asking is disabled until the user types a question.
    ask = next(button for button in app.button if button.label == "Ask the analyst")
    assert ask.disabled

    question_box = next(widget for widget in app.text_input if widget.label == "Your question")
    question_box.set_value("How many customers are in this dataset and how many are predicted to churn?")
    app.run(timeout=60)
    ask = next(button for button in app.button if button.label == "Ask the analyst")
    assert not ask.disabled
    ask.click().run(timeout=60)

    assert len(app.exception) == 0
    assert any("Computed results shown directly" in caption.value for caption in app.caption)
    facts_tables = [frame.value for frame in app.dataframe if "Metric" in frame.value.columns]
    assert facts_tables and facts_tables[0]["Metric"].str.contains("customers_analyzed").any()
    assert any("not configured" in caption.value or "GEMINI_API_KEY" in caption.value for caption in app.caption)


def test_v2_app_analyst_explains_answered_question_with_mocked_llm(monkeypatch):
    from ml import analyst

    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    calls = {"count": 0}

    def fake_call_llm(api_key, system, user, transport=None, timeout=30):
        calls["count"] += 1
        if calls["count"] == 1:
            return '{"function": "portfolio_overview", "arguments": {}}'
        return "The uploaded dataset contains 4 customers, and the verified counts confirm this number."

    monkeypatch.setattr(analyst, "_call_llm", fake_call_llm)
    app = AppTest.from_file("v2_app.py").run(timeout=30)
    app.get("file_uploader")[0].upload("sample_portfolio.csv", _csv_bytes(), "text/csv")
    app.run(timeout=60)

    question_box = next(widget for widget in app.text_input if widget.label == "Your question")
    question_box.set_value("How many customers are in this dataset?")
    app.run(timeout=60)
    ask = next(button for button in app.button if button.label == "Ask the analyst")
    ask.click().run(timeout=60)

    assert len(app.exception) == 0
    assert calls["count"] == 2  # planner + explainer, both mocked
    assert any("4 customers" in markdown.value for markdown in app.markdown)
    assert any(
        "verified against" in caption.value for caption in app.caption
    )
    facts_tables = [frame.value for frame in app.dataframe if "Metric" in frame.value.columns]
    assert facts_tables and facts_tables[0]["Metric"].str.contains("customers_analyzed").any()
