"use client";

// Customers — searchable, filterable portfolio table with per-customer detail:
// prediction, risk band, SHAP explanation, and rule-based recommendations.
// Filters and exports always operate on the same active filter state.

import { useEffect, useMemo, useState } from "react";
import { FIELDS, LABELS, Profile } from "@/lib/chat";
import {
  ALL_BANDS,
  BAND_CLASS,
  BAND_COLORS,
  Filters,
  Portfolio,
  Recommendation,
  ScoredRow,
  ShapExplanation,
  customerExportRows,
  displayPlan,
  filterRows,
  postJson,
  saveCsv,
  shapExportRows,
} from "@/lib/portfolio";

type Detail = { row: ScoredRow; explanation: ShapExplanation | null; recommendations: Recommendation[]; busy: boolean; error: string };

export default function CustomersView({
  portfolio,
  filters,
  onFilters,
  onUseAsBaseline,
}: {
  portfolio: Portfolio | null;
  filters: Filters;
  onFilters: (filters: Filters) => void;
  onUseAsBaseline: (row: ScoredRow) => void;
}) {
  const [sortDesc, setSortDesc] = useState(true);
  const [detail, setDetail] = useState<Detail | null>(null);

  const filtered = useMemo(() => {
    if (!portfolio) return [];
    const rows = filterRows(portfolio.rows, filters);
    return [...rows].sort((a, b) =>
      sortDesc ? Number(b.Churn_Probability) - Number(a.Churn_Probability) : String(a["Customer ID"]).localeCompare(String(b["Customer ID"])),
    );
  }, [portfolio, filters, sortDesc]);

  useEffect(() => { setDetail(null); }, [portfolio?.name]);

  if (!portfolio) {
    return <div className="surface panel empty-state">
      <div className="project-icon">☰</div>
      <h2 className="subhead">No customer dataset yet</h2>
      <p className="muted">Load the built-in sample or upload your CSV on the Portfolio page, then every customer lands here with search, filters, and explanations.</p>
      <p className="section-copy" style={{ marginTop: 10 }}>Customer data lives only in this browser session; “Clear session data” on the Project info page removes it.</p>
    </div>;
  }

  const toggleBand = (band: string) => {
    const risks = filters.risks.includes(band) ? filters.risks.filter((item) => item !== band) : [...filters.risks, band];
    onFilters({ ...filters, risks: risks.length ? risks : [...ALL_BANDS] });
  };
  const planOptions = (column: string) =>
    [...new Set(portfolio.rows.map((row) => displayPlan(row[column])).filter(Boolean))].sort();

  return <div className="customers-layout">
    <section className="surface panel">
      <div className="filter-bar">
        <div className="field" style={{ gridColumn: "1 / -1" }}>
          <label htmlFor="customer-search">Search customers</label>
          <input id="customer-search" type="search" placeholder="Search by ID or any recorded detail…" value={filters.search} onChange={(event) => onFilters({ ...filters, search: event.target.value })} />
        </div>
        <div className="field">
          <label>Risk bands</label>
          <div className="chip-row">
            {ALL_BANDS.map((band) => <button key={band} className={`chip ${filters.risks.includes(band) ? "selected" : ""}`} style={filters.risks.includes(band) ? { borderColor: BAND_COLORS[band] } : {}} onClick={() => toggleBand(band)}>{band.replace(" Risk", "")}</button>)}
          </div>
        </div>
        <div className="field">
          <label htmlFor="filter-intl">International plan</label>
          <select id="filter-intl" value={filters.internationalPlan.length === 1 ? filters.internationalPlan[0] : ""} onChange={(event) => onFilters({ ...filters, internationalPlan: event.target.value ? [event.target.value] : [] })}>
            <option value="">All</option>
            {planOptions("International plan").map((option) => <option key={option} value={option}>{option}</option>)}
          </select>
        </div>
        <div className="field">
          <label htmlFor="filter-vm">Voicemail plan</label>
          <select id="filter-vm" value={filters.voiceMailPlan.length === 1 ? filters.voiceMailPlan[0] : ""} onChange={(event) => onFilters({ ...filters, voiceMailPlan: event.target.value ? [event.target.value] : [] })}>
            <option value="">All</option>
            {planOptions("Voice mail plan").map((option) => <option key={option} value={option}>{option}</option>)}
          </select>
        </div>
        <div className="field">
          <label>Account length (days)</label>
          <div className="range-inputs">
            <input aria-label="Minimum account length" type="number" placeholder="min" value={filters.accountLength[0] ?? ""} onChange={(event) => onFilters({ ...filters, accountLength: [event.target.value === "" ? null : Number(event.target.value), filters.accountLength[1]] })} />
            <input aria-label="Maximum account length" type="number" placeholder="max" value={filters.accountLength[1] ?? ""} onChange={(event) => onFilters({ ...filters, accountLength: [filters.accountLength[0], event.target.value === "" ? null : Number(event.target.value)] })} />
          </div>
        </div>
        <div className="field">
          <label>Usage-charge proxy ($)</label>
          <div className="range-inputs">
            <input aria-label="Minimum usage charge" type="number" placeholder="min" value={filters.charge[0] ?? ""} onChange={(event) => onFilters({ ...filters, charge: [event.target.value === "" ? null : Number(event.target.value), filters.charge[1]] })} />
            <input aria-label="Maximum usage charge" type="number" placeholder="max" value={filters.charge[1] ?? ""} onChange={(event) => onFilters({ ...filters, charge: [filters.charge[0], event.target.value === "" ? null : Number(event.target.value)] })} />
          </div>
        </div>
      </div>
      <div className="results-head">
        <p className="section-copy">{filtered.length.toLocaleString()} of {portfolio.rows.length.toLocaleString()} customers shown{filtered.length !== portfolio.rows.length ? " · filters active" : ""}</p>
        <div className="action-row" style={{ marginTop: 0 }}>
          <button className="outline-button" onClick={() => setSortDesc(!sortDesc)}>{sortDesc ? "Sort: highest risk first" : "Sort: by customer ID"}</button>
          <button className="outline-button" onClick={() => saveCsv("churn_customer_predictions.csv", customerExportRows(filtered))} disabled={!filtered.length}>↓ Export these customers</button>
        </div>
      </div>
    </section>

    <div className="table-wrap surface panel" style={{ padding: 0 }}>
      <table className="data-table">
        <thead><tr>
          <th style={{ textAlign: "left" }}>Customer ID</th>
          <th>Churn probability</th>
          <th>Prediction</th>
          <th>Risk band</th>
          <th>Support calls</th>
          <th>Day minutes</th>
          <th>Estimated inputs</th>
          <th></th>
        </tr></thead>
        <tbody>
          {filtered.slice(0, 60).map((row) => <CustomerRow key={String(row["Customer ID"]) + String(row["__row"] ?? "")} row={row} onOpen={() => openCustomer(row)} />)}
        </tbody>
      </table>
    </div>
    {filtered.length > 60 && <p className="section-copy">Showing the first 60 customers of {filtered.length.toLocaleString()}. Exports include every filtered customer.</p>}

    {detail && <CustomerDetail detail={detail} onClose={() => setDetail(null)} onUseAsBaseline={() => { onUseAsBaseline(detail.row); setDetail(null); }} />}
  </div>;

  async function openCustomer(row: ScoredRow) {
    const id = String(row["Customer ID"] ?? "customer");
    setDetail({ row, explanation: null, recommendations: [], busy: true, error: "" });
    const profile = Object.fromEntries(FIELDS.map((field) => [field, row[field] ?? null]));
    try {
      const [explanation, recommendationResponse] = await Promise.all([
        postJson<ShapExplanation>("/api/explain", { profile, thresholds: portfolio?.thresholds }),
        postJson<{ recommendations: Recommendation[] }>("/api/recommend", { rows: [row], thresholds: portfolio?.thresholds }),
      ]);
      setDetail({ row, explanation, recommendations: recommendationResponse.recommendations, busy: false, error: "" });
    } catch (cause) {
      setDetail({ row, explanation: null, recommendations: [], busy: false, error: cause instanceof Error ? cause.message : "The explanation could not be prepared." });
    }
  }
}

function CustomerRow({ row, onOpen }: { row: ScoredRow; onOpen: () => void }) {
  const band = String(row.Risk_Category);
  return <tr onClick={onOpen} style={{ cursor: "pointer" }}>
    <td style={{ textAlign: "left", fontWeight: 600 }}>{String(row["Customer ID"])}</td>
    <td>{(Number(row.Churn_Probability) * 100).toFixed(1)}%</td>
    <td style={{ textAlign: "right", color: row.Predicted_Churner ? "var(--amber)" : undefined }}>{String(row.Prediction).replace(" to churn", "")}</td>
    <td style={{ textAlign: "right" }}><span className={`risk-pill ${BAND_CLASS[band] ?? ""}`} style={{ background: `${BAND_COLORS[band]}1f`, color: BAND_COLORS[band] }}>{band.replace(" Risk", "")}</span></td>
    <td>{String(row["Customer service calls"])}</td>
    <td>{Number(row["Total day minutes"]).toFixed(0)}</td>
    <td>{String(row.Estimated_Input_Count)}</td>
    <td style={{ textAlign: "right", color: "var(--teal)" }}>Open →</td>
  </tr>;
}

function CustomerDetail({ detail, onClose, onUseAsBaseline }: { detail: Detail; onClose: () => void; onUseAsBaseline: () => void }) {
  const { row, explanation, recommendations, busy, error } = detail;
  const id = String(row["Customer ID"]);
  const probability = Number(row.Churn_Probability);
  const band = String(row.Risk_Category);
  return <section className="surface panel customer-detail" aria-label={`Customer ${id}`}>
    <div className="results-head">
      <div>
        <h2 className="subhead">Customer {id}</h2>
        <div className="action-row" style={{ marginTop: 6 }}>
          <span className={`risk-pill ${BAND_CLASS[band] ?? ""}`} style={{ background: `${BAND_COLORS[band]}1f`, color: BAND_COLORS[band] }}>{band}</span>
          <span className="section-copy">{(probability * 100).toFixed(1)}% · {String(row.Prediction)} · {String(row.Estimated_Input_Count)} estimated input(s)</span>
        </div>
      </div>
      <div className="action-row" style={{ marginTop: 0 }}>
        <button className="outline-button" onClick={onUseAsBaseline}>⇄ Use in Simulation Lab</button>
        {explanation && <button className="outline-button" onClick={() => saveCsv(`cram_shap_explanation_${id}.csv`, shapExportRows(explanation, id))}>↓ Export SHAP explanation</button>}
        <button className="text-button" onClick={onClose}>Close</button>
      </div>
    </div>

    <div className="detail-grid">
      <div className="profile-list">
        {FIELDS.map((field) => <div className="profile-item" key={field}>
          <div className="profile-label">{LABELS[field]}</div>
          <div className="profile-value">{field === "International plan" || field === "Voice mail plan" ? displayPlan(row[field]) : row[field] === null || row[field] === undefined || row[field] === "" ? "not provided" : String(row[field])}</div>
        </div>)}
      </div>
      <div>
        {busy && <div className="loading">Preparing this customer’s explanation…</div>}
        {error && <div className="error">{error}</div>}
        {explanation && <ShapSection explanation={explanation} />}
        {recommendations.length > 0 && <div style={{ marginTop: 14 }}>
          <div className="form-section">Rule-based review suggestions</div>
          {recommendations.map((item) => <div className="note-card" key={item.rule_id} style={{ marginBottom: 8 }}>
            <strong>{item.title}</strong> <span className="section-copy">· priority {item.priority}</span>
            <p className="section-copy">{item.action}</p>
            <p className="section-copy" style={{ color: "#9fb2b8" }}>{item.why_triggered}</p>
            <p className="section-copy" style={{ fontSize: 10 }}>Limitations: {item.limitations}</p>
          </div>)}
        </div>}
        {!busy && !recommendations.length && !error && <p className="section-copy" style={{ marginTop: 14 }}>No rule-based suggestions for this customer at the current thresholds.</p>}
      </div>
    </div>
  </section>;
}

export function ShapSection({ explanation }: { explanation: ShapExplanation }) {
  return <div style={{ marginTop: 14 }}>
    <div className="form-section">What pushed this score (SHAP)</div>
    <p className="section-copy" style={{ marginBottom: 10 }}>{explanation.summary}</p>
    <div className="signal-list" style={{ marginTop: 4 }}>
      {explanation.all_contributions.map((item) => <div className="signal-row" key={item.feature}>
        <span title={item.input_status}>{LABELS[item.feature] || item.feature}{item.input_status !== "provided" ? " ◌" : ""}</span>
        <div className="signal-track">
          <span style={{ width: `${Math.max(2, Math.min(100, item.magnitude * 800))}%`, background: item.contribution > 0 ? "var(--red)" : item.contribution < 0 ? "var(--teal)" : "#5a6b74", transform: item.contribution > 0 ? "none" : "translateX(0)" }} />
        </div>
        <span>{item.contribution > 0 ? "+" : ""}{(item.contribution * 100).toFixed(1)} pt</span>
      </div>)}
    </div>
    <div className="score-caption" style={{ marginTop: 10 }}>
      Signed contributions in probability points relative to the fixed V2 reference population (base {(explanation.base_value * 100).toFixed(1)}%).
      ◌ marks inputs that were missing or invalid and used typical values. These are model associations, not causes.
    </div>
    <div className="score-caption" style={{ marginTop: 6 }}>
      Binary decision applied at {(explanation.decision_threshold_used * 100).toFixed(0)}%: {explanation.prediction}. Risk band: {explanation.risk_category}.
    </div>
  </div>;
}

export type { Profile };
