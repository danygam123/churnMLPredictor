"use client";

import { ChangeEvent, FormEvent, KeyboardEvent, useEffect, useRef, useState } from "react";
import metadata from "../model_metadata.json";
import v2Metadata from "../models/model_metadata.json";
import { FIELDS, LABELS, Profile, parseCustomerText, parseSingleAnswer } from "@/lib/chat";
import {
  ALL_BANDS,
  BAND_CLASS,
  BAND_COLORS,
  DEFAULT_FILTERS,
  Filters,
  Portfolio,
  ScoredRow,
  ShapExplanation,
  Thresholds,
  applyBands,
  csvCell,
  postJson,
  saveCsvText,
} from "@/lib/portfolio";
import PortfolioView from "./views/PortfolioView";
import CustomersView from "./views/CustomersView";
import RetentionView from "./views/RetentionView";
import SimulationView from "./views/SimulationView";
import AnalystView from "./views/AnalystView";

type View = "Predict" | "Portfolio" | "Customers" | "Retention Strategies" | "Simulation Lab" | "AI Analyst" | "Model performance" | "Project info";
type Message = { role: "assistant" | "user"; text: string };
type Prediction = { score: number; label: string; estimatedFields: string[]; riskBand: string; signals?: { name: string; change: number }[] };
type Importance = { name: string; value: number }[];

const NAV: { name: View; icon: string }[] = [
  { name: "Predict", icon: "◉" },
  { name: "Portfolio", icon: "◈" },
  { name: "Customers", icon: "☰" },
  { name: "Retention Strategies", icon: "✚" },
  { name: "Simulation Lab", icon: "⇄" },
  { name: "AI Analyst", icon: "✦" },
  { name: "Model performance", icon: "⌁" },
  { name: "Project info", icon: "ⓘ" },
];
const RANGE = metadata.input_ranges as Record<string, { min: number; max: number; median: number }>;
const V2_RANGE = v2Metadata.input_ranges as Record<string, { min: number; max: number; median: number }>;
const DEFAULT_THRESHOLDS: Thresholds = {
  medium_threshold: v2Metadata.optimal_threshold,
  high_threshold: v2Metadata.default_threshold,
};
const QUESTIONS: Record<string, string> = {
  "Account length": "About how long have they been with their phone provider? Months or years are fine.",
  "International plan": "Do they have international calling included in their plan?",
  "Voice mail plan": "Do they have a voicemail plan?",
  "Number vmail messages": "Roughly how many voicemail messages do they get in a month?",
  "Total day minutes": "Roughly how many minutes do they use during the day in a month?",
  "Total day calls": "About how many daytime calls do they make in a month?",
  "Total eve minutes": "Roughly how many minutes do they use in the evening in a month?",
  "Total eve calls": "About how many evening calls do they make in a month?",
  "Total night minutes": "About how much phone time do they use at night in a month?",
  "Total night calls": "About how many nighttime calls do they make in a month?",
  "Total intl minutes": "Roughly how many international calling minutes do they use in a month?",
  "Total intl calls": "About how many international calls do they make in a month?",
  "Customer service calls": "How many times have they called customer support?",
};
const INITIAL: Message[] = [{ role: "assistant", text: "Hi, I’m Compass. Tell me about the customer in everyday language. Share a few details at once, skip what you don’t know, and I’ll help fill in the picture." }];
const QUICK_STARTS = [
  { icon: "☎", label: "Phone use", text: "They use about 120 minutes during the day and 80 minutes in the evening." },
  { icon: "✳", label: "Plans & tenure", text: "They have been with the provider for about 2 years, no international plan, and voicemail yes." },
  { icon: "☏", label: "Support calls", text: "They have called customer support 3 times." },
];
const EXAMPLES: Record<string, string> = {
  "Account length": "They have been with the provider for about 6 months.",
  "International plan": "They do not have an international calling plan.",
  "Voice mail plan": "They have a voicemail plan.",
  "Number vmail messages": "They get about 8 voicemail messages a month.",
  "Customer service calls": "They have called customer support 3 times.",
};
const METRICS = metadata.evaluation_model.test_metrics;
const MEDIAN_CHECK = metadata.evaluation_model.median_imputation_sensitivity;
const INTEGER_SET = new Set(["Account length", "Number vmail messages", "Total day calls", "Total eve calls", "Total night calls", "Total intl calls", "Customer service calls"]);

function formatValue(field: string, value: number | null) {
  if (value === null) return "Not sure · typical value will be used";
  if (field === "International plan" || field === "Voice mail plan") return value ? "Yes" : "No";
  const number = ["Account length", "Number vmail messages", "Total day calls", "Total eve calls", "Total night calls", "Total intl calls", "Customer service calls"].includes(field)
    ? Math.round(value).toLocaleString() : value.toFixed(1);
  if (field.includes("minutes")) return `${number} min / month`;
  if (field.includes("calls")) return `${number} / month`;
  if (field === "Account length") return `${number} days`;
  return number;
}

async function scoreProfile(profile: Profile, thresholds: Thresholds): Promise<Prediction> {
  return postJson<Prediction>("/api/score", { profile, thresholds });
}

function PageHeader({ eyebrow, title, description }: { eyebrow: string; title: string; description: string }) {
  return <div className="page-head"><div><p className="eyebrow">{eyebrow}</p><h1 className="page-title">{title}</h1><p className="page-description">{description}</p></div><span className="preview"><span className="online" /> RESEARCH PREVIEW</span></div>;
}

// ---------------------------------------------------------------------------
// Score card — existing idiom, extended with the V2 risk band and real SHAP
// ---------------------------------------------------------------------------

function ScoreCard({ result, profile, thresholds }: { result: Prediction; profile: Profile; thresholds: Thresholds }) {
  const [open, setOpen] = useState(false);
  const [shap, setShap] = useState<ShapExplanation | null>(null);
  const [shapBusy, setShapBusy] = useState(false);
  const [shapError, setShapError] = useState("");

  const loadShap = async () => {
    if (shap || shapBusy) { setOpen(!open); return; }
    setShapBusy(true); setShapError(""); setOpen(true);
    try {
      setShap(await postJson<ShapExplanation>("/api/explain", { profile, thresholds }));
    } catch (cause) {
      setShapError(cause instanceof Error ? cause.message : "The explanation could not be prepared.");
    } finally { setShapBusy(false); }
  };

  return <div className="score-card">
    <div className="score-top"><div><div className="score-label">ESTIMATED CHURN SCORE</div><div className="score-number">{(result.score * 100).toFixed(1)}%</div></div>
      <span style={{ display: "flex", gap: 6 }}>
        <span className="risk-pill low">{result.label}</span>
        <span className={`risk-pill ${BAND_CLASS[result.riskBand] ?? ""}`} style={{ background: `${BAND_COLORS[result.riskBand]}1f`, color: BAND_COLORS[result.riskBand] }}>{result.riskBand.replace(" Risk", "")} risk</span>
      </span></div>
    <div className="score-track"><span style={{ width: `${Math.max(1, result.score * 100)}%` }} /></div>
    <div className="score-caption">This is a model estimate, not a guarantee. {result.estimatedFields.length ? `${result.estimatedFields.length} detail(s) used typical values.` : "No details needed estimating."} The “Likely / less likely” label applies the 50% research cutoff; the colored band is a separate V2 classification.</div>
    {result.estimatedFields.length > 0 && <div className="alert">Estimated details: {result.estimatedFields.map((field) => LABELS[field]).join(", ")}. Predictions with more unknown details are less dependable.</div>}
    {result.signals?.length ? <>
      <button className="text-button" onClick={() => setOpen(!open)}>{open ? "Hide" : "Explore"} which details shifted this score</button>
      {open && <div className="signal-list">{result.signals.map((signal) => <div className="signal-row" key={signal.name}><span>{LABELS[signal.name]}</span><div className="signal-track"><span style={{ width: `${Math.max(5, Math.min(100, Math.abs(signal.change) * 500))}%` }} /></div><span>{signal.change > 0 ? "+" : ""}{(signal.change * 100).toFixed(1)} pt</span></div>)}<div className="score-caption">Comparison with the model’s typical value for one detail at a time. These are model patterns, not causes of churn.</div></div>}
    </> : null}
    <button className="text-button" onClick={() => void loadShap()}>{shap ? (open ? "Hide the model explanation (SHAP)" : "Show the model explanation (SHAP)") : "Why this score? Compute the model explanation (SHAP)"}</button>
    {open && (shapBusy ? <div className="loading" style={{ marginTop: 10 }}>Computing the SHAP explanation…</div>
      : shapError ? <div className="error">{shapError}</div>
      : shap ? <div className="signal-list">
        <div className="score-caption" style={{ marginBottom: 4 }}>{shap.summary}</div>
        {shap.all_contributions.map((item) => <div className="signal-row" key={item.feature}>
          <span title={item.input_status}>{LABELS[item.feature] || item.feature}{item.input_status !== "provided" ? " ◌" : ""}</span>
          <div className="signal-track"><span style={{ width: `${Math.max(2, Math.min(100, item.magnitude * 800))}%`, background: item.contribution > 0 ? "var(--red)" : item.contribution < 0 ? "var(--teal)" : "#5a6b74" }} /></div>
          <span>{item.contribution > 0 ? "+" : ""}{(item.contribution * 100).toFixed(1)} pt</span>
        </div>)}
        <div className="score-caption">Signed SHAP contributions in probability points relative to the fixed V2 reference population (base {(shap.base_value * 100).toFixed(1)}%). ◌ marks inputs that were missing or invalid and used typical values. These are model associations, not causes.</div>
      </div> : null)}
  </div>;
}

// ---------------------------------------------------------------------------
// Home shell
// ---------------------------------------------------------------------------

export default function Home() {
  const [view, setView] = useState<View>("Predict");
  const [entryMode, setEntryMode] = useState<"chat" | "form">("chat");
  const [profile, setProfile] = useState<Profile>({});
  const [messages, setMessages] = useState<Message[]>(INITIAL);
  const [composer, setComposer] = useState("");
  const [helpOpen, setHelpOpen] = useState(false);
  const [prediction, setPrediction] = useState<Prediction | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [quickValues, setQuickValues] = useState<Record<string, string>>(() => Object.fromEntries(FIELDS.map((field) => [field, String(RANGE[field].median)])));
  const [quickUnknown, setQuickUnknown] = useState<string[]>([]);
  const [importance, setImportance] = useState<Importance>([]);
  const composerRef = useRef<HTMLTextAreaElement>(null);

  // Portfolio (dataset) state — lives in this browser session only.
  const [portfolio, setPortfolio] = useState<Portfolio | null>(null);
  const [filters, setFilters] = useState<Filters>(DEFAULT_FILTERS);
  const [thresholds, setThresholds] = useState<Thresholds>(DEFAULT_THRESHOLDS);
  const [uploadBusy, setUploadBusy] = useState(false);
  const [uploadError, setUploadError] = useState("");
  const [exportBusy, setExportBusy] = useState(false);
  const [simBaseline, setSimBaseline] = useState<{ profile: Profile; source: string } | null>(null);

  const missing = FIELDS.filter((field) => !(field in profile));
  const captured = FIELDS.filter((field) => field in profile && profile[field] !== null);
  const estimated = FIELDS.filter((field) => field in profile && profile[field] === null);
  const complete = missing.length === 0;

  const score = async (updated: Profile) => {
    setBusy(true);
    try { setPrediction(await scoreProfile(updated, thresholds)); }
    catch (cause) { setError(cause instanceof Error ? cause.message : "Could not connect to the prediction service."); }
    finally { setBusy(false); }
  };

  const submitChat = async (event?: FormEvent) => {
    event?.preventDefault();
    const text = composer.trim(); if (!text || busy) return;
    setComposer(""); setError(""); setHelpOpen(false);
    const next = { ...profile };
    const parsed = parseCustomerText(text, next, RANGE);
    let updated = parsed.profile; let touched = parsed.changed;
    const waiting = FIELDS.find((field) => !(field in profile));
    if (!touched.length && waiting) {
      const value = parseSingleAnswer(waiting, text, RANGE);
      if (value !== undefined) { updated = { ...next, [waiting]: value }; touched = [waiting]; }
      else if (/^(i'?m )?(not sure|unsure|unknown|skip|idk|n\/?a|i don't know|dont know|no idea)[.! ]*$/i.test(text)) {
        updated = { ...next, [waiting]: null }; touched = [waiting];
      }
    }
    const stillMissing = FIELDS.filter((field) => !(field in updated));
    let reply = "";
    if (touched.length) {
      const summaries = touched.map((field) => `${LABELS[field]}: ${updated[field] === null ? "not sure; I’ll mark it as estimated" : formatValue(field, updated[field]!)}`);
      reply = `Got it — ${summaries.join(" · ")}.`;
      if (stillMissing.length) reply += `\n\n${QUESTIONS[stillMissing[0]]}`;
      else reply += "\n\nI’ve got a complete profile. I’ll score it now; you can still correct anything in the chat.";
    } else if (/not sure|don't know|dont know|unsure|unknown|skip/i.test(text)) {
      reply = "No problem. Which part are you unsure about? Try saying something like ‘not sure about nighttime use’ or name the plan or detail.";
    } else {
      reply = "I didn’t quite catch a customer detail there. Try including the time period and a number, like ‘about 100 evening minutes,’ or tell me whether a plan is included.";
      if (waiting) reply += `\n\n${QUESTIONS[waiting]}`;
    }
    setMessages((prior) => [...prior, { role: "user", text }, { role: "assistant", text: reply }]);
    if (touched.length) setProfile(updated);
    if (FIELDS.every((field) => field in updated)) {
      void score(updated);
    }
  };

  const useTypical = async () => {
    const field = missing[0]; if (!field) return;
    const updated = { ...profile, [field]: null };
    setProfile(updated); setHelpOpen(false);
    const next = FIELDS.find((key) => !(key in updated));
    const response = `That’s okay — I’ll use a typical value for ${LABELS[field].toLowerCase()} and mark it as estimated.${next ? `\n\n${QUESTIONS[next]}` : " I have a complete profile now."}`;
    setMessages((prior) => [...prior, { role: "assistant", text: response }]);
    if (!next) void score(updated);
  };

  const runQuickPrediction = async (event: FormEvent) => {
    event.preventDefault(); setError(""); setBusy(true);
    const unknownSet = new Set(quickUnknown);
    const formProfile: Profile = Object.fromEntries(FIELDS.map((field) => [field, unknownSet.has(field) ? null : Number(quickValues[field])]));
    setProfile(formProfile);
    try { setPrediction(await scoreProfile(formProfile, thresholds)); }
    catch (cause) { setError(cause instanceof Error ? cause.message : "Could not connect to the prediction service."); }
    finally { setBusy(false); }
  };

  const loadImportance = async () => {
    if (importance.length) return;
    try { const response = await fetch("/api/predict?view=stats"); const body = await response.json(); if (response.ok) setImportance(body.importance || []); }
    catch { /* The evaluation page remains useful without this optional chart. */ }
  };

  // --- Dataset actions ------------------------------------------------

  const analyzeCsv = async (text: string, name: string, isSample: boolean) => {
    setUploadBusy(true); setUploadError("");
    try {
      const response = await postJson<Record<string, unknown>>("/api/analyze", { csv: text, thresholds });
      if (response.rejected) {
        setUploadError(String(response.reason));
        return; // a rejected upload never replaces the working dataset
      }
      const columns = response.columns_rows as { columns: string[]; rows: (string | number | boolean | null)[][] };
      const rows: ScoredRow[] = columns.rows.map((cells) => Object.fromEntries(columns.columns.map((column, index) => [column, cells[index] ?? null])));
      setPortfolio({
        name, isSample,
        columns: columns.columns, rows,
        summary: response.summary as Portfolio["summary"],
        warnings: response.warnings as string[],
        notes: response.notes as string[],
        labelColumnPresent: Boolean(response.label_column_present),
        idSourceColumn: (response.id_source_column as string | null) ?? null,
        predictionThreshold: Number(response.prediction_threshold),
        thresholds: response.risk_thresholds as Thresholds,
        bandCounts: response.band_counts as Record<string, number>,
        groupRates: response.group_rates as Portfolio["groupRates"],
      });
      setFilters(DEFAULT_FILTERS);
      setSimBaseline(null);
    } catch (cause) {
      setUploadError(cause instanceof Error ? cause.message : "The dataset could not be analyzed.");
    } finally { setUploadBusy(false); }
  };

  const handleUpload = async (event: ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0]; if (!file) return;
    if (file.size > 4_200_000) { setUploadError("This file is over the 4 MB limit. Please split it into smaller CSV files."); return; }
    await analyzeCsv(await file.text(), file.name, false);
  };

  const loadSample = async () => {
    try {
      const response = await fetch("/sample_customers.csv");
      await analyzeCsv(await response.text(), "Built-in sample (public telecom benchmark, de-identified)", true);
    } catch {
      setUploadError("The built-in sample dataset could not be read from the application files.");
    }
  };

  const changeThresholds = (next: Thresholds) => {
    setThresholds(next);
    setPortfolio((current) => current ? {
      ...current,
      thresholds: next,
      rows: applyBands(current.rows, next),
      bandCounts: current.bandCounts,
    } : current);
  };

  const clearSession = () => {
    setPortfolio(null); setFilters(DEFAULT_FILTERS); setThresholds(DEFAULT_THRESHOLDS);
    setSimBaseline(null); setUploadError(""); setPrediction(null); setProfile({});
    setMessages(INITIAL); setComposer(""); setError("");
  };

  const exportSummary = async () => {
    if (!portfolio) return;
    setExportBusy(true);
    try {
      const rows = applyBands(portfolio.rows, thresholds);
      const response = await postJson<{ analytics: Record<string, unknown>[] }>("/api/analyze", {
        rows: rows.map((row) => ({ ...row })),
        thresholds,
      });
      saveCsvText("cram_portfolio_analytics_summary.csv", toCsv(response.analytics));
    } catch (cause) {
      setUploadError(cause instanceof Error ? cause.message : "The analytics summary could not be exported.");
    } finally { setExportBusy(false); }
  };

  const resetChat = () => { setProfile({}); setMessages(INITIAL); setPrediction(null); setError(""); setComposer(""); };
  const handleComposerKey = (event: KeyboardEvent<HTMLTextAreaElement>) => { if (event.key === "Enter" && !event.shiftKey) { event.preventDefault(); void submitChat(); } };

  const pageCopy: Record<View, [string, string, string]> = {
    Predict: ["CUSTOMER WORKSPACE", "Customer risk", "Describe a customer in your own words and explore an estimated churn score."],
    Portfolio: ["PORTFOLIO OVERVIEW", "Portfolio at a glance", "Upload or load a customer list to explore risk, patterns, and predictions across the portfolio."],
    Customers: ["CUSTOMER DATABASE", "Explore customers", "Search, filter, and open any customer to see its prediction, risk band, and explanation."],
    "Retention Strategies": ["RETENTION REVIEW", "Who to review first", "Transparent, rule-based review suggestions for the customers in the current view."],
    "Simulation Lab": ["WHAT-IF ANALYSIS", "Try hypothetical changes", "Compare a baseline customer with up to three hypothetical scenarios through the model."],
    "AI Analyst": ["RESTRICTED ANALYTICS", "Ask your data", "Plain-language questions, answered with statistics computed from your scored customers."],
    "Model performance": ["MODEL TRANSPARENCY", "How the model performs", "Review held-out results, missing-detail sensitivity, and the patterns the model uses."],
    "Project info": ["PROJECT NOTES", "About this research", "Explore the data, approach, intended limits, and your session settings."],
  };
  const [eyebrow, title, description] = pageCopy[view];

  return <div className="shell">
    <aside className="sidebar">
      <div className="brand"><div className="brand-icon">✦</div><div><div className="brand-title">Churn Compass</div><div className="brand-sub">Customer intelligence</div></div></div>
      <div className="side-label">WORKSPACE</div>
      <nav className="nav" aria-label="Main navigation">{NAV.map((item) => <button key={item.name} className={`nav-button ${view === item.name ? "active" : ""}`} onClick={() => { setView(item.name); if (item.name === "Model performance") void loadImportance(); }}><span className="nav-icon">{item.icon}</span>{item.name}</button>)}</nav>
      {portfolio && <div className="side-dataset">
        <div className="side-label" style={{ padding: "14px 10px 8px" }}>ACTIVE DATA</div>
        <div className="side-dataset-card">
          <strong>{portfolio.isSample ? "Sample data" : "Your upload"}</strong>
          <span>{portfolio.rows.length.toLocaleString()} customers · {portfolio.name.length > 26 ? `${portfolio.name.slice(0, 26)}…` : portfolio.name}</span>
          <span>Bands: Med ≥ {thresholds.medium_threshold.toFixed(2)} · High ≥ {thresholds.high_threshold.toFixed(2)}</span>
        </div>
        <button className="side-clear" onClick={clearSession}>Clear session data</button>
      </div>}
      <div className="side-bottom"><strong><span className="side-dot" />Research preview</strong>Telecom customer data<br />{metadata.dataset.rows.toLocaleString()} source records · 13 details</div>
    </aside>
    <main className="main">
      <div className="topline"><div className="crumb">CHURN COMPASS <span style={{ color: "#5f737d", padding: "0 6px" }}>/</span> {view.toUpperCase()}</div><span className="preview"><span className="online" /> MODEL ESTIMATES</span></div>
      <PageHeader eyebrow={eyebrow} title={title} description={description} />

      {view === "Predict" && <>
        <div className="mode-switch"><button className={`mode-choice ${entryMode === "chat" ? "selected" : ""}`} onClick={() => setEntryMode("chat")}><span>✦</span> Chat with Compass</button><button className={`mode-choice ${entryMode === "form" ? "selected" : ""}`} onClick={() => setEntryMode("form")}><span>▤</span> Quick form</button></div>
        {entryMode === "chat" ? <>
        <div className="split">
          <section className="surface chat-panel">
            <div className="chat-top"><div className="guide-id"><div className="guide-avatar">✦</div><div><div className="guide-name">Compass <span className="guide-role">· customer guide</span></div><div className="guide-status"><span className="online" />Ready to help map a customer profile</div></div></div><span className="chat-tag">GUIDED INTERVIEW</span></div>
            <div className="messages" aria-live="polite">{messages.map((message, index) => <div key={`${index}-${message.role}`} className={`message ${message.role}`}><div className="message-avatar">{message.role === "assistant" ? "✦" : "🙂"}</div><div className="message-body">{message.role === "assistant" && <span className="message-by">Compass guide</span>}{message.text}</div></div>)}{busy && <div className="message"><div className="message-avatar">✦</div><div className="message-body"><span className="message-by">Compass guide</span><span className="typing"><i /><i /><i /></span></div></div>}</div>
            {!complete && <div style={{ marginTop: 12 }}>
              <div className="quick-title">{messages.length === 1 ? "A few easy ways to start" : "You can also mention"}</div>
              <div className="quick-row">{QUICK_STARTS.map((quick) => <button className="quick-chip" key={quick.label} onClick={() => { setComposer(quick.text); composerRef.current?.focus(); }}>{quick.icon} &nbsp;{quick.label}</button>)}</div>
            </div>}
            {helpOpen && <div className="help-pop"><div className="help-head">{missing.length ? `A simple way to think about it: ${LABELS[missing[0]]}` : "Need to update something?"}</div>{missing.length ? <>{QUESTIONS[missing[0]]}<br /><span style={{ color: "#9fb6b5" }}>Typical values in the research data run from {RANGE[missing[0]].min} to {RANGE[missing[0]].max}.</span><div className="action-row"><button className="text-button" onClick={() => setComposer(EXAMPLES[missing[0]] || `They use about ${Math.round(RANGE[missing[0]].median)} ${missing[0].includes("minutes") ? "minutes" : "calls"} ${missing[0].includes("day") ? "during the day" : missing[0].includes("eve") ? "in the evening" : missing[0].includes("night") ? "at night" : "internationally"} each month.`)}>Show me an example</button><button className="text-button" onClick={() => void useTypical()}>Use a typical value</button></div></> : "Describe the customer detail you want to change in the chat."}</div>}
            {error && <div className="error">{error}</div>}
            <form className="composer" onSubmit={(event) => void submitChat(event)}><textarea ref={composerRef} value={composer} onChange={(event) => setComposer(event.target.value)} onKeyDown={handleComposerKey} placeholder="Describe the customer in your own words…" rows={1} aria-label="Describe a customer" /><button className="send-button" type="submit" disabled={!composer.trim() || busy} aria-label="Send message">↑</button></form>
            <div className="chat-under"><span>Share what you know; “not sure” is a valid answer.</span><button className="text-button" onClick={() => setHelpOpen(!helpOpen)}>{helpOpen ? "Close help" : "Need a hand?"}</button></div>
            {prediction && <ScoreCard result={prediction} profile={profile} thresholds={thresholds} />}
            {complete && <div className="chat-actions"><button className="outline-button" onClick={resetChat}>↺ &nbsp;Start a new customer</button><span className="success-line">Profile ready · {captured.length} known · {estimated.length} estimated</span></div>}
          </section>
          <aside className="surface snapshot"><div className="snapshot-head"><div><div className="section-heading">Customer snapshot</div><div className="snapshot-sub">A live read of the conversation</div></div><span className="snapshot-mark">{complete ? "✓" : "✦"}</span></div><div className="count">{Object.keys(profile).length} of {FIELDS.length} details gathered</div><div className="progress"><span style={{ width: `${Object.keys(profile).length / FIELDS.length * 100}%` }} /></div>
            {Object.keys(profile).length ? <div className="profile-list">{FIELDS.filter((field) => field in profile).map((field) => <div className="profile-item" key={field}><div className="profile-label">{LABELS[field]}</div><div className="profile-value">{formatValue(field, profile[field])}</div>{profile[field] === null && <div className="estimated-label">◌ estimated from typical data</div>}</div>)}</div> : <div className="snapshot-empty"><span className="empty-spark">✧</span><strong>Your profile will build as we talk.</strong><p>Start with a customer’s phone habits, plan, or time with their provider.</p></div>}
            {estimated.length > 0 && <div className="snapshot-note">{estimated.length} detail(s) will use typical values and be clearly marked in the result.</div>}
            <details className="helper"><summary>How the estimate works</summary><p>Compass maps the details you share to a research model. Missing details use typical values from its training data and are called out in your result.</p></details>
            {prediction && <button className="outline-button" style={{ marginTop: 13, width: "100%" }} onClick={() => setSimBaseline({ profile, source: "Predict page profile" })}>⇄ Use this profile in the Simulation Lab</button>}
          </aside>
        </div>
        </> : <div className="quick-split"><form className="surface panel quick-panel" onSubmit={runQuickPrediction}>
          <div className="section-heading">Customer details</div><p className="section-copy">A rough estimate is fine. Mark anything unknown and we’ll show what was estimated.</p>
          <div className="form-section">Account & plans</div><div className="form-grid">{FIELDS.slice(0, 4).map((field) => renderField(field))}</div>
          <div className="form-section">Monthly phone activity</div><div className="form-grid">{FIELDS.slice(4, 12).map((field) => renderField(field))}</div>
          <div className="form-section">Customer support</div><div className="form-grid">{renderField(FIELDS[12])}</div>
          <div className="action-row"><button className="primary-button" type="submit" disabled={busy}>{busy ? "Scoring…" : "Estimate customer risk"}</button><button type="button" className="outline-button" onClick={() => { setQuickValues(Object.fromEntries(FIELDS.map((field) => [field, String(RANGE[field].median)]))); setQuickUnknown([]); setPrediction(null); }}>Reset fields</button></div>
          {prediction && <ScoreCard result={prediction} profile={Object.fromEntries(FIELDS.map((field) => [field, quickUnknown.includes(field) ? null : Number(quickValues[field])]))} thresholds={thresholds} />}
        </form><aside className="surface panel form-aside"><div className="project-icon">✧</div><h2 className="subhead">Make it your own</h2><p className="muted">Enter a complete profile or mark a detail as unknown. Unknown fields use typical values from the model’s training data and are clearly labeled in the result.</p><div className="form-aside-stat"><span>Prediction cutoff</span><strong>50%</strong></div><div className="form-aside-stat"><span>Inputs used</span><strong>13 details</strong></div><div className="note-card" style={{ marginTop: 16 }}>A research estimate for exploring this dataset. Not a guarantee or a customer decision.</div></aside></div>}
      </>}

      {view === "Portfolio" && <PortfolioView portfolio={portfolio} filters={filters} onFilters={setFilters} uploadBusy={uploadBusy} uploadError={uploadError} onFile={(event) => void handleUpload(event)} onLoadSample={() => void loadSample()} onExportSummary={() => void exportSummary()} exportBusy={exportBusy} onNavigate={(next) => setView(next as View)} />}
      {view === "Customers" && <CustomersView portfolio={portfolio} filters={filters} onFilters={setFilters} onUseAsBaseline={(row) => { setSimBaseline({ profile: Object.fromEntries(FIELDS.map((field) => [field, row[field] ?? null])) as Profile, source: `Customer ${row["Customer ID"]}` }); setView("Simulation Lab"); }} />}
      {view === "Retention Strategies" && <RetentionView portfolio={portfolio} filters={filters} />}
      {view === "Simulation Lab" && <SimulationView portfolio={portfolio} baseline={simBaseline} onBaseline={setSimBaseline} />}
      {view === "AI Analyst" && <AnalystView portfolio={portfolio} filters={filters} />}
      {view === "Model performance" && <ModelView importance={importance} thresholds={thresholds} />}
      {view === "Project info" && <ProjectView portfolio={portfolio} thresholds={thresholds} onThresholds={changeThresholds} onClearSession={clearSession} onLoadSample={() => void loadSample()} uploadBusy={uploadBusy} uploadError={uploadError} onFile={(event) => void handleUpload(event)} />}
    </main>
  </div>;

  function renderField(field: string) {
    const binary = field === "International plan" || field === "Voice mail plan";
    const unknown = quickUnknown.includes(field);
    return <div className="field" key={field}><div className="field-head"><label htmlFor={`field-${field}`}>{LABELS[field]}</label><label className="unknown-check"><input type="checkbox" checked={unknown} onChange={(event) => setQuickUnknown((current) => event.target.checked ? [...current, field] : current.filter((key) => key !== field))} /> Not sure</label></div>
      {binary ? <select id={`field-${field}`} required={!unknown} value={quickValues[field]} disabled={unknown} onChange={(event) => setQuickValues((current) => ({ ...current, [field]: event.target.value }))}><option value="0">No</option><option value="1">Yes</option></select> : <input id={`field-${field}`} type="number" min={RANGE[field].min} max={RANGE[field].max} step={INTEGER_SET.has(field) ? 1 : 0.1} required={!unknown} value={quickValues[field]} disabled={unknown} onChange={(event) => setQuickValues((current) => ({ ...current, [field]: event.target.value }))} />}
      <small>{binary ? "Choose the plan that best describes this customer." : `Research data range: ${RANGE[field].min}–${RANGE[field].max}${field.includes("minutes") ? " min" : ""}.`}</small>
    </div>;
  }
}

function toCsv(rows: Record<string, unknown>[]): string {
  if (!rows.length) return "";
  const headers = Object.keys(rows[0]);
  return [headers.map(csvCell).join(","), ...rows.map((row) => headers.map((key) => csvCell(row[key])).join(","))].join("\r\n");
}

// ---------------------------------------------------------------------------
// Model performance — existing views + the V2 champion's evaluation record
// ---------------------------------------------------------------------------

const HELDOUT = v2Metadata.held_out_test_metrics as Record<string, Record<string, number | number[][]>>;
const cm = (name: string) => HELDOUT[name].confusion_matrix as unknown as number[][];
const CALIBRATION = v2Metadata.calibration as unknown as Record<string, { brier_score: number; expected_calibration_error: number }>;
const COST = v2Metadata.business_cost_framework as { cost_false_negative: number; cost_false_positive: number };

function ModelView({ importance, thresholds }: { importance: Importance; thresholds: Thresholds }) {
  const [globalImportance, setGlobalImportance] = useState<{ feature: string; mean_abs_contribution: number }[]>([]);
  useEffect(() => {
    fetch("/global_feature_importance.csv").then((response) => response.text()).then((text) => {
      const [, ...lines] = text.trim().split("\n");
      setGlobalImportance(lines.map((line) => { const [feature, value] = line.split(","); return { feature, mean_abs_contribution: Number(value) }; }).filter((item) => item.feature && !Number.isNaN(item.mean_abs_contribution)));
    }).catch(() => { /* The page remains useful without this optional chart. */ });
  }, []);
  const metrics: [string, number, string][] = [
    ["Accuracy", METRICS.accuracy, "Overall correct predictions"], ["Churn recall", METRICS.churn_recall, "Churners correctly identified"], ["Churn precision", METRICS.churn_precision, "Flagged customers who churned"],
    ["Churn F1", METRICS.churn_f1, "Precision and recall balance"], ["ROC-AUC", METRICS.roc_auc, "Ranking separation"], ["PR-AUC", METRICS.pr_auc_average_precision, "Imbalance-aware ranking"],
  ];
  const matrix = METRICS.confusion_matrix_labels_0_1;
  const operatingPoints = Object.entries(HELDOUT).map(([name, values]) => ({ name, values }));
  return <div className="model-layout"><div className="metric-grid">{metrics.map(([label, value, note]) => <div className="metric-card" key={label}><div className="metric-label">{label}</div><div className="metric-value">{(value * 100).toFixed(1)}%</div><div className="metric-note">{note}</div></div>)}</div>
    <div className="note-card">Churn is {(metadata.dataset.churn_rate * 100).toFixed(1)}% of this dataset. A majority-class baseline reaches {(metadata.baseline.test_accuracy * 100).toFixed(1)}% accuracy but catches no churners. Accuracy alone can hide that.</div>
    <div className="model-grid"><section className="surface panel"><h2 className="subhead">Held-out results</h2><p className="section-copy">The model trained on {metadata.dataset.rows.toLocaleString()} records. These results use a stratified {Math.round(metadata.split.test_size * 100)}% test split ({metadata.split.test_rows} customers) held aside from training.</p><div className="table-wrap" style={{ marginTop: 15 }}><table className="data-table confusion"><thead><tr><th>Actual \ Predicted</th><th>No churn</th><th>Churn</th></tr></thead><tbody><tr><th>No churn</th><td>{matrix[0][0]}</td><td>{matrix[0][1]}</td></tr><tr><th>Churn</th><td>{matrix[1][0]}</td><td>{matrix[1][1]}</td></tr></tbody></table></div></section>
      <section className="surface panel"><h2 className="subhead">When details are unknown</h2><p className="section-copy">Test profiles had details hidden and replaced with training-data medians.</p><div className="unknown-grid">{Object.entries(MEDIAN_CHECK).map(([count, result]) => <div className="unknown-card" key={count}><div className="metric-label">{count} unknown details</div><div className="mini-metric"><span>Churn recall</span><strong>{(result.churn_recall.mean * 100).toFixed(1)}%</strong></div><div className="mini-metric"><span>Churn F1</span><strong>{(result.churn_f1.mean * 100).toFixed(1)}%</strong></div><div className="mini-metric"><span>Accuracy</span><strong>{(result.accuracy.mean * 100).toFixed(1)}%</strong></div></div>)}</div></section></div>

    <section className="surface panel">
      <h2 className="subhead">V2 engine · champion model evaluation</h2>
      <p className="section-copy">
        The upgraded application scores customers with the V2 champion ({String(v2Metadata.champion_model_name)}),
        evaluated once on a held-out test split. The table compares the two documented operating points —
        the production <b>50% binary cutoff</b> used across this app, and the Phase 1 cost-optimal threshold
        of {Number(v2Metadata.optimal_threshold).toFixed(4)}, shown here only as an evaluated operating point
        (a false negative costs ${COST.cost_false_negative.toFixed(0)} in lost customer value; a false positive costs ${COST.cost_false_positive.toFixed(0)} of retention outreach).
      </p>
      <div className="table-wrap" style={{ marginTop: 12 }}>
        <table className="data-table">
          <thead><tr><th style={{ textAlign: "left" }}>Operating point</th><th>Threshold</th><th>Accuracy</th><th>Precision</th><th>Recall</th><th>F1</th><th>Brier</th><th>Business cost</th><th>Savings vs. contact-no-one</th></tr></thead>
          <tbody>
            {operatingPoints.map(({ name, values }) => <tr key={name}>
              <td style={{ textAlign: "left", fontWeight: 600 }}>{name === "at_default_0_50" ? "Default 50% cutoff (production)" : "Phase 1 cost-optimal (evaluated)"}</td>
              <td>{Number(values.threshold).toFixed(4)}</td>
              <td>{(Number(values.accuracy) * 100).toFixed(1)}%</td>
              <td>{(Number(values.precision) * 100).toFixed(1)}%</td>
              <td>{(Number(values.recall) * 100).toFixed(1)}%</td>
              <td>{(Number(values.f1) * 100).toFixed(1)}%</td>
              <td>{Number(values.brier_score).toFixed(4)}</td>
              <td>${Number(values.business_cost).toLocaleString()}</td>
              <td>${Number(values.cost_savings).toLocaleString()}</td>
            </tr>)}
          </tbody>
        </table>
      </div>
      <div className="score-caption" style={{ marginTop: 10 }}>
        Trade-off at the cost-optimal point: recall rises from {(Number(HELDOUT.at_default_0_50.recall) * 100).toFixed(1)}% to {(Number(HELDOUT.at_optimal_threshold.recall) * 100).toFixed(1)}% and modeled cost falls, but far more customers are flagged
        ({cm("at_optimal_threshold")[0][1] + cm("at_optimal_threshold")[1][1]} vs. {cm("at_default_0_50")[0][1] + cm("at_default_0_50")[1][1]}). Production keeps the 50% label; this page documents the alternative.
      </div>
    </section>

    <section className="surface panel">
      <h2 className="subhead">Probability calibration</h2>
      <p className="section-copy">How closely predicted probabilities track observed churn on the held-out split (lower is better).</p>
      <div className="table-wrap" style={{ marginTop: 12 }}>
        <table className="data-table">
          <thead><tr><th style={{ textAlign: "left" }}>Variant</th><th>Brier score</th><th>Expected calibration error</th></tr></thead>
          <tbody>{Object.entries(CALIBRATION).map(([variant, values]) => <tr key={variant}>
            <td style={{ textAlign: "left" }}>{variant.replace("_", " ")}</td>
            <td>{Number(values.brier_score).toFixed(4)}</td>
            <td>{Number(values.expected_calibration_error).toFixed(4)}</td>
          </tr>)}</tbody>
        </table>
      </div>
      <div className="score-caption" style={{ marginTop: 10 }}>The served model is the uncalibrated champion; calibration variants were evaluated but not deployed. Probabilities are model estimates, not calibrated guarantees.</div>
    </section>

    <div className="model-grid">
      <section className="surface panel">
        <h2 className="subhead">Risk bands & input schema</h2>
        <p className="section-copy">Defaults from the V2 metadata; sessions can adjust bands in Project info.</p>
        <div className="form-aside-stat"><span>Medium risk starts at</span><strong>{thresholds.medium_threshold.toFixed(4)}</strong></div>
        <div className="form-aside-stat"><span>High risk starts at</span><strong>{thresholds.high_threshold.toFixed(2)}</strong></div>
        <div className="form-section" style={{ marginTop: 14 }}>Accepted input ranges</div>
        <div className="table-wrap" style={{ maxHeight: 260 }}>
          <table className="data-table"><thead><tr><th style={{ textAlign: "left" }}>Input</th><th>Min</th><th>Max</th><th>Typical (median)</th></tr></thead>
            <tbody>{FIELDS.map((field) => <tr key={field}><td style={{ textAlign: "left" }}>{field}</td><td>{V2_RANGE[field].min}</td><td>{V2_RANGE[field].max}</td><td>{V2_RANGE[field].median}</td></tr>)}</tbody>
          </table>
        </div>
        <p className="section-copy" style={{ marginTop: 10 }}>Plans accept Yes/No variants. Optional columns: <b>Customer ID</b> (synthetic IDs are generated when absent) and <b>Churn</b> (true/false labels for observed-rate comparisons).</p>
      </section>
      <section className="surface panel">
        <h2 className="subhead">Model-wide feature importance</h2>
        <p className="section-copy">Training-sample global SHAP results from Phase 2; this chart describes the model, not any particular upload.</p>
        {globalImportance.length ? <>
          <div className="bar-chart">{globalImportance.slice(0, 10).map((item) => <div className="bar-item" key={item.feature}><span>{item.feature}</span><div className="bar-track"><span style={{ width: `${item.mean_abs_contribution / globalImportance[0].mean_abs_contribution * 100}%` }} /></div><span>{(item.mean_abs_contribution * 100).toFixed(1)} pt</span></div>)}</div>
          <div className="action-row"><a className="outline-button" href="/global_feature_importance.csv" download="cram_global_feature_importance.csv">↓ Export global feature importance</a></div>
        </> : <p className="loading">Loading model signals…</p>}
      </section>
    </div>

    <section className="surface panel importance"><h2 className="subhead">Signals the original Random Forest used most</h2><p className="section-copy">Relative importance within the original serving model, kept for comparison with the upgraded engine. These patterns are not proof that a detail causes churn.</p>{importance.length ? <div className="bar-chart">{importance.slice(0, 10).map((item) => <div className="bar-item" key={item.name}><span>{LABELS[item.name] || item.name}</span><div className="bar-track"><span style={{ width: `${item.value / importance[0].value * 100}%` }} /></div><span>{(item.value * 100).toFixed(1)}%</span></div>)}</div> : <p className="loading">Loading model signals…</p>}</section>
    <p className="disclaimer">This model was evaluated on this telecom dataset only. It is an educational research prototype; don’t use its estimate as the sole basis for contacting, pricing, or treating real customers.</p>
  </div>;
}

// ---------------------------------------------------------------------------
// Project info + Settings
// ---------------------------------------------------------------------------

function ProjectView({
  portfolio,
  thresholds,
  onThresholds,
  onClearSession,
  onLoadSample,
  uploadBusy,
  uploadError,
  onFile,
}: {
  portfolio: Portfolio | null;
  thresholds: Thresholds;
  onThresholds: (thresholds: Thresholds) => void;
  onClearSession: () => void;
  onLoadSample: () => void;
  uploadBusy: boolean;
  uploadError: string;
  onFile: (event: ChangeEvent<HTMLInputElement>) => void;
}) {
  const [providerConfigured, setProviderConfigured] = useState<boolean | null>(null);
  useEffect(() => {
    fetch("/api/analyst").then((response) => response.json()).then((body) => setProviderConfigured(Boolean(body?.provider?.configured))).catch(() => setProviderConfigured(false));
  }, []);
  const [medium, setMedium] = useState(String(thresholds.medium_threshold));
  const [high, setHigh] = useState(String(thresholds.high_threshold));
  useEffect(() => { setMedium(String(thresholds.medium_threshold)); setHigh(String(thresholds.high_threshold)); }, [thresholds]);

  const applyThresholds = () => {
    const nextMedium = Number(medium), nextHigh = Number(high);
    if (Number.isNaN(nextMedium) || Number.isNaN(nextHigh) || nextMedium < 0 || nextHigh <= nextMedium || nextHigh > 1) return;
    onThresholds({ medium_threshold: nextMedium, high_threshold: nextHigh });
  };
  const thresholdValid = !Number.isNaN(Number(medium)) && !Number.isNaN(Number(high)) && Number(high) > Number(medium) && Number(medium) >= 0 && Number(high) <= 1;

  return <div className="project-layout">
    <section className="surface project-intro"><div className="eyebrow">THE RESEARCH</div><h2 className="project-title">A practical experiment in customer retention.</h2><p className="project-lead">This project explores whether account history, plan choices, calling habits, and support contacts can help identify telecom customers who may be at risk of leaving. The upgraded application runs the C.R.A.M. V2 engine: portfolio analytics, risk bands, SHAP explanations, rule-based retention reviews, what-if simulation, and a restricted AI analyst.</p></section>
    <div className="metric-grid"><div className="metric-card"><div className="metric-label">Customer records</div><div className="metric-value">{metadata.dataset.rows.toLocaleString()}</div><div className="metric-note">Public telecom dataset</div></div><div className="metric-card"><div className="metric-label">Customers who churned</div><div className="metric-value">{metadata.dataset.churn_rows.toLocaleString()}</div><div className="metric-note">{(metadata.dataset.churn_rate * 100).toFixed(1)}% of the source rows</div></div><div className="metric-card"><div className="metric-label">Customer details</div><div className="metric-value">13</div><div className="metric-note">Plans, usage, tenure, support</div></div></div>
    <div className="project-grid">
      <section className="surface panel"><div className="project-icon">⌕</div><h2 className="subhead">How it works</h2><p className="muted">The app gathers a customer profile or a CSV portfolio, validates every value, derives usage charges from minutes, fills unknown values with training-data medians, and scores with the V2 champion model. Explanations use SHAP against a fixed reference population.</p><div className="step-list"><div><b>01</b><span>Describe or upload customers</span></div><div><b>02</b><span>Review validated, scored results</span></div><div><b>03</b><span>Explore explanations and scenarios</span></div></div></section>
      <section className="surface panel"><div className="project-icon">◎</div><h2 className="subhead">Research limits</h2><p className="muted">The model reflects this telecom dataset and its customer population. It may not generalize to another provider or industry. A score is an estimate, not a guarantee or a causal explanation. Recommendations are review prompts, not proven interventions.</p><div className="note-card" style={{ marginTop: 15 }}>Use this application to explore model behavior and research questions, not to make decisions about real customers.</div><a className="paper-link" href="/Customer_Churn_Project_Paper.pdf" target="_blank" rel="noreferrer">Read the project paper ↗</a></section>
    </div>

    <section className="surface panel" id="settings">
      <h2 className="subhead">Session settings</h2>
      <p className="section-copy">Your dataset, thresholds, and results live only in this browser session.</p>
      <div className="settings-grid">
        <div>
          <div className="form-section">Active dataset</div>
          {portfolio ? <div className="profile-list">
            <div className="profile-item"><div className="profile-label">Source</div><div className="profile-value">{portfolio.isSample ? "Built-in sample (de-identified benchmark)" : "Your upload"}</div></div>
            <div className="profile-item"><div className="profile-label">File</div><div className="profile-value">{portfolio.name}</div></div>
            <div className="profile-item"><div className="profile-label">Customers scored</div><div className="profile-value">{portfolio.rows.length.toLocaleString()}</div></div>
          </div> : <p className="muted">No dataset is loaded. Upload a CSV or load the built-in 80-customer sample.</p>}
          <div className="action-row">
            {portfolio && <button className="outline-button" onClick={onClearSession}>↺ Clear session data</button>}
            {!portfolio && <button className="primary-button" onClick={onLoadSample} disabled={uploadBusy}>✦ Load the built-in sample</button>}
          </div>
          <div className="upload-box" style={{ marginTop: 12 }}>
            <strong style={{ fontSize: 12 }}>Upload a customer CSV</strong>
            <p className="section-copy">Up to 4 MB and 10,000 customers; all 13 model input columns required. A rejected file never replaces a working dataset.</p>
            <input className="file-input" type="file" accept=".csv,text/csv" onChange={onFile} disabled={uploadBusy} />
          </div>
          {uploadBusy && <div className="loading" style={{ marginTop: 8 }}>Validating and scoring…</div>}
          {uploadError && <div className="error">{uploadError}</div>}
        </div>
        <div>
          <div className="form-section">Risk bands (V2 classification)</div>
          <p className="section-copy">Descriptive segments separate from the 50% binary decision. Defaults come from the V2 metadata.</p>
          <div className="range-inputs" style={{ marginTop: 10 }}>
            <div className="field"><label htmlFor="band-medium">Medium risk ≥</label><input id="band-medium" type="number" min={0} max={1} step={0.0001} value={medium} onChange={(event) => setMedium(event.target.value)} /></div>
            <div className="field"><label htmlFor="band-high">High risk ≥</label><input id="band-high" type="number" min={0} max={1} step={0.01} value={high} onChange={(event) => setHigh(event.target.value)} /></div>
          </div>
          <div className="action-row">
            <button className="primary-button" disabled={!thresholdValid} onClick={applyThresholds}>Apply bands</button>
            <button className="outline-button" onClick={() => onThresholds(DEFAULT_THRESHOLDS)}>Reset to defaults</button>
          </div>
          {!thresholdValid && <div className="error">Bands must satisfy 0 ≤ medium &lt; high ≤ 1.</div>}
          <div className="form-section" style={{ marginTop: 16 }}>AI provider status</div>
          <p className="section-copy">{providerConfigured === null ? "Checking…" : providerConfigured
            ? "A language-model key is configured on the server. The analyst narrates computed facts with grounding validation."
            : "No language-model key is configured. The analyst runs in facts-only mode and still returns verified statistics."} The key itself is never shown here or sent to the browser.</p>
          <div className="note-card" style={{ marginTop: 10 }}><b>Privacy:</b> uploads are processed in memory for this session and never stored server-side. When AI narration is on, the request sends only column names, dataset metadata, and aggregate computed facts — not raw rows or identifiers.</div>
        </div>
      </div>
    </section>
  </div>;
}
