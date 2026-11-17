"use client";

// Simulation Lab — Phase 5 A/B/C scenario behavior through the V2 engine.
// Every scenario is evaluated on a fresh copy of the baseline; editing one
// scenario never mutates the baseline or another scenario.

import { useMemo, useState } from "react";
import { FIELDS, LABELS, Profile } from "@/lib/chat";
import {
  Portfolio,
  ScoredRow,
  SimulationResult,
  postJson,
} from "@/lib/portfolio";
import v2Metadata from "../../models/model_metadata.json";

const RANGE = v2Metadata.input_ranges as Record<string, { min: number; max: number; median: number }>;
const INTEGER_SET = new Set(["Account length", "Number vmail messages", "Total day calls", "Total eve calls", "Total night calls", "Total intl calls", "Customer service calls"]);

type ScenarioState = { name: string; values: Record<string, string>; changes: string[] };

function blankScenario(name: string): ScenarioState {
  return { name, values: {}, changes: [] };
}

export default function SimulationView({
  portfolio,
  baseline,
  onBaseline,
}: {
  portfolio: Portfolio | null;
  baseline: { profile: Profile; source: string } | null;
  onBaseline: (baseline: { profile: Profile; source: string } | null) => void;
}) {
  const [scenarios, setScenarios] = useState<ScenarioState[]>([blankScenario("Scenario A")]);
  const [results, setResults] = useState<SimulationResult[] | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");

  const customers = useMemo(() =>
    portfolio ? [...portfolio.rows].sort((a, b) => Number(b.Churn_Probability) - Number(a.Churn_Probability)).slice(0, 500) : [],
    [portfolio?.name, portfolio?.rows]);

  if (!baseline) {
    return <div className="surface panel empty-state">
      <div className="project-icon">⇄</div>
      <h2 className="subhead">Pick a baseline customer</h2>
      <p className="muted">
        Simulations compare a starting profile with up to three hypothetical variants. Choose a
        customer from your dataset below, or score one first on the Predict page and it will
        appear here automatically.
      </p>
      {portfolio ? <div className="field" style={{ maxWidth: 420, marginTop: 14 }}>
        <label htmlFor="sim-customer">Baseline from the customer database</label>
        <select id="sim-customer" onChange={(event) => {
          const row = portfolio.rows.find((item) => String(item["Customer ID"]) === event.target.value);
          if (row) onBaseline({ profile: Object.fromEntries(FIELDS.map((field) => [field, row[field] ?? null])) as Profile, source: `Customer ${row["Customer ID"]}` });
        }} value="">
          <option value="">Select a customer…</option>
          {customers.map((row) => <option key={String(row["Customer ID"])} value={String(row["Customer ID"])}>
            {String(row["Customer ID"])} · {(Number(row.Churn_Probability) * 100).toFixed(1)}% · {String(row.Risk_Category).replace(" Risk", "")}
          </option>)}
        </select>
        <small>Suggestions are the highest-risk customers; search for any ID in the Customers page.</small>
      </div> : <p className="section-copy" style={{ marginTop: 12 }}>Load a dataset on the Portfolio page to pick a customer, or enter one on the Predict page.</p>}
    </div>;
  }

  const run = async () => {
    setBusy(true); setError(""); setResults(null);
    try {
      const response = await postJson<{ results: SimulationResult[] }>("/api/simulate", {
        profile: baseline.profile,
        scenarios: scenarios.map((scenario) => ({
          name: scenario.name,
          changes: Object.fromEntries(scenario.changes.map((field) => [field, scenario.values[field]])),
        })),
        thresholds: portfolio?.thresholds,
      });
      setResults(response.results);
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : "The simulation could not run.");
    } finally { setBusy(false); }
  };

  return <div className="simulation-layout">
    <section className="surface panel">
      <div className="results-head">
        <div>
          <h2 className="subhead">Baseline · {baseline.source}</h2>
          <p className="section-copy">Only the 13 model inputs can change. Identifiers, labels, and derived charge fields are not editable.</p>
        </div>
        <button className="text-button" onClick={() => { onBaseline(null); setScenarios([blankScenario("Scenario A")]); setResults(null); }}>Change baseline</button>
      </div>
      <div className="detail-grid">
        <div className="profile-list">{FIELDS.filter((field) => baseline.profile[field] !== null && baseline.profile[field] !== undefined).map((field) => <div className="profile-item" key={field}><div className="profile-label">{LABELS[field]}</div><div className="profile-value">{String(baseline.profile[field])}</div></div>)}</div>
        <div>
          <div className="form-section">Scenarios ({scenarios.length} of 3)</div>
          <div className="scenario-grid">
            {scenarios.map((scenario, index) => <ScenarioCard key={index} scenario={scenario} onChange={(next) => setScenarios(scenarios.map((item, i) => i === index ? next : item))} onRemove={scenarios.length > 1 ? () => setScenarios(scenarios.filter((_, i) => i !== index)) : undefined} />)}
            {scenarios.length < 3 && <button className="outline-button" style={{ alignSelf: "start", justifySelf: "start" }} onClick={() => setScenarios([...scenarios, blankScenario(`Scenario ${["A", "B", "C"][scenarios.length]}`)])}>+ Add scenario</button>}
          </div>
          <div className="action-row">
            <button className="primary-button" onClick={() => void run()} disabled={busy || scenarios.every((scenario) => !scenario.changes.length)}>{busy ? "Simulating…" : "Run comparison"}</button>
            <button className="outline-button" onClick={() => { setScenarios([blankScenario("Scenario A")]); setResults(null); }}>Reset scenarios</button>
          </div>
        </div>
      </div>
      {error && <div className="error">{error}</div>}
    </section>

    {results && <section className="surface panel">
      <h2 className="subhead">Baseline versus scenarios</h2>
      <p className="section-copy">Binary decision at 50%; risk bands use the session thresholds. Scenarios are hypothetical model inputs — the model cannot confirm that a change would cause an outcome.</p>
      <div className="table-wrap" style={{ marginTop: 12 }}>
        <table className="data-table">
          <thead><tr>
            <th style={{ textAlign: "left" }}>View</th><th>Churn probability</th><th>Change</th><th>Prediction</th><th>Risk band</th><th style={{ textAlign: "left" }}>Changed inputs</th><th style={{ textAlign: "left" }}>Notes</th>
          </tr></thead>
          <tbody>
            <tr>
              <td style={{ textAlign: "left", fontWeight: 600 }}>Baseline</td>
              <td>{(results[0].original_probability * 100).toFixed(1)}%</td><td>—</td>
              <td style={{ textAlign: "right" }}>{results[0].original_prediction.replace(" to churn", "")}</td>
              <td style={{ textAlign: "right" }}>{results[0].baseline_risk_category.replace(" Risk", "")}</td>
              <td style={{ textAlign: "left" }}>—</td><td style={{ textAlign: "left" }}>—</td>
            </tr>
            {results.map((result) => <tr key={result.name}>
              <td style={{ textAlign: "left", fontWeight: 600 }}>{result.name}</td>
              <td>{(result.simulated_probability * 100).toFixed(1)}%</td>
              <td style={{ textAlign: "right", color: result.difference > 0 ? "var(--red)" : result.difference < 0 ? "var(--teal)" : undefined }}>
                {result.difference > 0 ? "+" : ""}{result.difference_percentage_points.toFixed(1)} pt
              </td>
              <td style={{ textAlign: "right" }}>{result.simulated_prediction.replace(" to churn", "")}</td>
              <td style={{ textAlign: "right" }}>{result.simulated_risk_category.replace(" Risk", "")}</td>
              <td style={{ textAlign: "left" }}>{Object.entries(result.changed_features).map(([field, change]) => `${LABELS[field] || field}: ${change.original_value ?? "typical"} → ${change.simulated_value}`).join(", ") || "—"}</td>
              <td style={{ textAlign: "left" }}>
                {[
                  result.out_of_training_range_features.length ? `outside training range: ${result.out_of_training_range_features.join(", ")}` : "",
                  result.estimated_features.length ? `${result.estimated_features.length} estimated input(s)` : "",
                ].filter(Boolean).join(" · ") || "—"}
              </td>
            </tr>)}
          </tbody>
        </table>
      </div>
    </section>}
  </div>;
}

function ScenarioCard({ scenario, onChange, onRemove }: { scenario: ScenarioState; onChange: (scenario: ScenarioState) => void; onRemove?: () => void }) {
  const [field, setField] = useState<string>("Total day minutes");
  const value = scenario.values[field] ?? "";
  return <div className="scenario-card">
    <div className="results-head">
      <strong>{scenario.name}</strong>
      <div className="action-row" style={{ marginTop: 0 }}>
        {onRemove && <button className="text-button" onClick={onRemove}>Remove</button>}
      </div>
    </div>
    <div className="field">
      <label htmlFor={`sim-field-${scenario.name}`}>Input to change</label>
      <select id={`sim-field-${scenario.name}`} value={field} onChange={(event) => setField(event.target.value)}>
        {FIELDS.map((option) => <option key={option} value={option}>{LABELS[option]}</option>)}
      </select>
    </div>
    <div className="field">
      <label htmlFor={`sim-value-${scenario.name}`}>New value</label>
      {field === "International plan" || field === "Voice mail plan"
        ? <select id={`sim-value-${scenario.name}`} value={value} onChange={(event) => onChange({ ...scenario, values: { ...scenario.values, [field]: event.target.value }, changes: [...new Set([...scenario.changes, field])] })}>
            <option value="">Unchanged</option><option value="Yes">Yes</option><option value="No">No</option>
          </select>
        : <input id={`sim-value-${scenario.name}`} type="number" min={RANGE[field].min} max={RANGE[field].max} step={INTEGER_SET.has(field) ? 1 : 0.1} value={value} placeholder={`Baseline value used if left blank (${RANGE[field].median})`} onChange={(event) => onChange({ ...scenario, values: { ...scenario.values, [field]: event.target.value }, changes: [...new Set([...scenario.changes, field])] })} />}
      <small>Accepted range {RANGE[field].min}–{RANGE[field].max}. Leave blank to keep the baseline value.</small>
    </div>
    {scenario.changes.filter((change) => change !== field).length > 0 &&
      <div className="chip-row" style={{ marginTop: 8 }}>
        {scenario.changes.filter((change) => change !== field).map((change) => <button key={change} className="chip selected" onClick={() => onChange({ ...scenario, changes: scenario.changes.filter((item) => item !== change) })}>{LABELS[change]} ×</button>)}
      </div>}
  </div>;
}

export type { ScoredRow };
