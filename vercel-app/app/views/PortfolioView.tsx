"use client";

// Portfolio overview — the Phase 7 Overview experience in Churn Compass idiom.
// Metrics are computed from the active scored dataset (never hardcoded), and
// uploads are validated with the Phase 3 validator + Phase 7 precheck before a
// working dataset can be replaced.

import { ChangeEvent, useRef } from "react";
import { LABELS } from "@/lib/chat";
import {
  ALL_BANDS,
  BAND_COLORS,
  Filters,
  GroupRate,
  Portfolio,
  ScoredRow,
  saveCsvText,
} from "@/lib/portfolio";

const SAMPLE_NOTE =
  "The built-in sample is an 80-customer, de-identified extract from the public telecom benchmark (synthetic IDs C-00xx). It exists for demonstration only and passes the same validation pipeline as any upload.";

export default function PortfolioView({
  portfolio,
  filters,
  onFilters,
  uploadBusy,
  uploadError,
  onFile,
  onLoadSample,
  onExportSummary,
  exportBusy,
  onNavigate,
}: {
  portfolio: Portfolio | null;
  filters: Filters;
  onFilters: (filters: Filters) => void;
  uploadBusy: boolean;
  uploadError: string;
  onFile: (event: ChangeEvent<HTMLInputElement>) => void;
  onLoadSample: () => void;
  onExportSummary: () => void;
  exportBusy: boolean;
  onNavigate: (view: string) => void;
}) {
  const fileRef = useRef<HTMLInputElement>(null);

  if (!portfolio) {
    return <div className="portfolio-layout">
      <section className="surface panel empty-state">
        <div className="project-icon">◈</div>
        <h2 className="subhead">Load a customer portfolio</h2>
        <p className="muted">
          Upload a CSV of telecom customers to explore portfolio risk, patterns, and predictions.
          Blank cells are filled with typical values and disclosed. Your file is processed for
          this session only and is not stored.
        </p>
        <div className="upload-box">
          <div style={{ fontSize: 22, marginBottom: 8, color: "var(--teal)" }}>⇧</div>
          <strong style={{ fontSize: 12 }}>Choose your customer file</strong>
          <p className="section-copy">CSV file · up to 4 MB · 10,000 customers · all 13 model input columns required</p>
          <input className="file-input" type="file" accept=".csv,text/csv" onChange={onFile} disabled={uploadBusy} />
        </div>
        <div className="action-row">
          <button className="primary-button" onClick={onLoadSample} disabled={uploadBusy}>✦ &nbsp;Load the built-in sample</button>
          <a className="outline-button" href="/sample_customers.csv" download="churn_sample_customers.csv">↓ &nbsp;Download the sample file</a>
        </div>
        {uploadBusy && <div className="loading" style={{ marginTop: 10 }}>Validating and scoring your customers…</div>}
        {uploadError && <div className="error">{uploadError}</div>}
        <p className="section-copy" style={{ marginTop: 14 }}>{SAMPLE_NOTE}</p>
      </section>
      <section className="surface panel">
        <h2 className="subhead">What you’ll get</h2>
        <div className="profile-list">
          {[
            ["Headline metrics", "Customers analyzed, predicted churners, average probability, high-risk count"],
            ["Observed churn rate", "When your file includes a Churn column with true/false labels"],
            ["Risk-band distribution", "Low, Medium, and High segments with adjustable thresholds"],
            ["Customer patterns", "Plan and support-call patterns across the portfolio"],
            ["Analytics export", "A CSV summary of everything on this page"],
          ].map(([label, note]) => <div className="profile-item" key={label}><div className="profile-label">{label}</div><div className="profile-value" style={{ fontWeight: 400, fontSize: 11, color: "#c3d2d6" }}>{note}</div></div>)}
        </div>
      </section>
    </div>;
  }

  const summary = portfolio.summary;
  const metrics: [string, string, string][] = [
    ["Customers analyzed", summary.customers_analyzed.toLocaleString(), "From the active dataset"],
    ["Predicted churners", summary.predicted_churners.toLocaleString(), "Model probability ≥ 50% cutoff"],
    ["Average probability", `${(summary.average_predicted_probability * 100).toFixed(1)}%`, "Mean predicted churn probability"],
    ["High-risk customers", summary.high_risk_customers.toLocaleString(), "High-risk band (session thresholds)"],
    ["Observed churn rate", summary.observed_churn_rate === null ? "—" : `${(summary.observed_churn_rate * 100).toFixed(1)}%`, summary.observed_churn_rate === null ? "No valid Churn labels in this file" : `From ${summary.observed_label_count} labeled customer(s)`],
  ];
  const total = Math.max(1, summary.customers_analyzed);

  return <div className="portfolio-layout">
    <div className="dataset-banner">
      <div>
        <strong>{portfolio.isSample ? "Sample data" : "Your upload"} · {portfolio.name}</strong>
        <span className="section-copy">{summary.customers_analyzed.toLocaleString()} customers scored with the V2 engine · binary decision at the 50% cutoff</span>
      </div>
      <div className="action-row" style={{ marginTop: 0 }}>
        <button className="text-button" onClick={onNavigate.bind(null, "Customers")}>Browse customers →</button>
        <button className="text-button" onClick={onNavigate.bind(null, "AI Analyst")}>Ask the AI Analyst →</button>
      </div>
    </div>

    {uploadError && <div className="error">Previous upload rejected: {uploadError}</div>}

    <div className="metric-grid">{metrics.map(([label, value, note]) => <div className="metric-card" key={label}><div className="metric-label">{label}</div><div className="metric-value">{value}</div><div className="metric-note">{note}</div></div>)}</div>

    <section className="surface panel">
      <h2 className="subhead">Risk-band distribution</h2>
      <p className="section-copy">Descriptive segments of model probability, separate from the 50% binary decision. Adjust bands in Project info.</p>
      <div className="bar-chart">
        {ALL_BANDS.map((band) => <div className="bar-item" key={band}>
          <span>{band}</span>
          <div className="bar-track"><span style={{ width: `${Math.max(2, (portfolio.bandCounts[band] / total) * 100)}%`, background: BAND_COLORS[band] }} /></div>
          <span>{portfolio.bandCounts[band].toLocaleString()} · {((portfolio.bandCounts[band] / total) * 100).toFixed(0)}%</span>
        </div>)}
      </div>
    </section>

    <section className="surface panel">
      <h2 className="subhead">Customer patterns</h2>
      <p className="section-copy">
        Observed churn rate by group when labels are present; otherwise the mean predicted probability.
        These are associations within this dataset, not causes.
      </p>
      {Object.entries(portfolio.groupRates).length ? Object.entries(portfolio.groupRates).map(([column, rates]) =>
        <GroupChart key={column} column={column} rates={rates} />) : <p className="muted">No groupable columns are available in this dataset.</p>}
    </section>

    <section className="surface panel">
      <div className="results-head">
        <div>
          <h2 className="subhead">Validation notes</h2>
          <p className="section-copy">How the validator handled this dataset. Unknown or invalid values are imputed with typical values and disclosed per customer.</p>
        </div>
        <button className="outline-button" onClick={onExportSummary} disabled={exportBusy}>{exportBusy ? "Preparing…" : "↓ Export analytics summary"}</button>
      </div>
      {portfolio.warnings.length || portfolio.notes.length ? <ul className="note-list">
        {portfolio.warnings.map((warning) => <li key={warning}>{warning}</li>)}
        {portfolio.notes.map((note) => <li key={note}>{note}</li>)}
      </ul> : <p className="success-line" style={{ marginTop: 10 }}>Every provided value was accepted by validation.</p>}
      <p className="section-copy" style={{ marginTop: 12 }}>
        “Derived Usage Charge Total” is the model’s usage-charge estimate (minutes × research rates);
        it is not an account monthly bill.
      </p>
    </section>
  </div>;
}

function GroupChart({ column, rates }: { column: string; rates: GroupRate[] }) {
  const max = Math.max(...rates.map((rate) => rate.rate), 0.0001);
  return <div style={{ marginTop: 14 }}>
    <div className="form-section" style={{ margin: "0 0 9px" }}>{column}</div>
    <div className="bar-chart" style={{ marginTop: 0 }}>
      {rates.map((rate) => <div className="bar-item" key={rate.group}>
        <span>{LABELS[column] || column}: {rate.group}</span>
        <div className="bar-track"><span style={{ width: `${Math.max(2, (rate.rate / max) * 100)}%` }} /></div>
        <span>{(rate.rate * 100).toFixed(1)}% · {rate.customers}</span>
      </div>)}
    </div>
  </div>;
}

export type { ScoredRow };
