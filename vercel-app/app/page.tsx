"use client";

import { ChangeEvent, FormEvent, KeyboardEvent, useRef, useState } from "react";
import metadata from "../model_metadata.json";
import { FIELDS, LABELS, Profile, parseCustomerText, parseSingleAnswer } from "@/lib/chat";

type View = "Predict" | "Batch scoring" | "Model performance" | "Project info";
type Message = { role: "assistant" | "user"; text: string };
type Prediction = { score: number; label: string; estimatedFields: string[]; signals?: { name: string; change: number }[] };
type BatchRecord = Record<string, string> & { "Churn risk score"?: string; Prediction?: string; "Estimated input count"?: string };
type Importance = { name: string; value: number }[];

const NAV: { name: View; icon: string }[] = [
  { name: "Predict", icon: "◉" },
  { name: "Batch scoring", icon: "▤" },
  { name: "Model performance", icon: "⌁" },
  { name: "Project info", icon: "ⓘ" },
];
const RANGE = metadata.input_ranges as Record<string, { min: number; max: number; median: number }>;
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

async function scoreProfile(profile: Profile): Promise<Prediction> {
  const response = await fetch("/api/predict", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ profile }) });
  const body = await response.json();
  if (!response.ok) throw new Error(body.error || "The model could not score that customer.");
  return body as Prediction;
}

function parseCsv(text: string): string[][] {
  const rows: string[][] = [];
  let row: string[] = [], cell = "", quoted = false;
  for (let i = 0; i < text.length; i++) {
    const char = text[i];
    if (char === '"' && quoted && text[i + 1] === '"') { cell += '"'; i++; }
    else if (char === '"') quoted = !quoted;
    else if (char === "," && !quoted) { row.push(cell); cell = ""; }
    else if ((char === "\n" || char === "\r") && !quoted) {
      if (char === "\r" && text[i + 1] === "\n") i++;
      row.push(cell); if (row.some((v) => v.trim())) rows.push(row); row = []; cell = "";
    } else cell += char;
  }
  row.push(cell); if (row.some((v) => v.trim())) rows.push(row);
  return rows;
}

function csvCell(value: unknown) {
  const text = String(value ?? "");
  return /[",\r\n]/.test(text) ? `"${text.replaceAll('"', '""')}"` : text;
}

function saveCsv(filename: string, rows: Record<string, unknown>[]) {
  if (!rows.length) return;
  const headers = Object.keys(rows[0]);
  const content = [headers.map(csvCell).join(","), ...rows.map((row) => headers.map((key) => csvCell(row[key])).join(","))].join("\r\n");
  const url = URL.createObjectURL(new Blob([content], { type: "text/csv;charset=utf-8" }));
  const anchor = document.createElement("a"); anchor.href = url; anchor.download = filename; anchor.click(); URL.revokeObjectURL(url);
}

function ScoreCard({ result }: { result: Prediction }) {
  const [open, setOpen] = useState(false);
  return <div className="score-card">
    <div className="score-top"><div><div className="score-label">ESTIMATED CHURN SCORE</div><div className="score-number">{(result.score * 100).toFixed(1)}%</div></div><span className={`risk-pill ${result.score < .5 ? "low" : ""}`}>{result.label}</span></div>
    <div className="score-track"><span style={{ width: `${Math.max(1, result.score * 100)}%` }} /></div>
    <div className="score-caption">This is a model estimate, not a guarantee. {result.estimatedFields.length ? `${result.estimatedFields.length} detail(s) used typical values.` : "No details needed estimating."}</div>
    {result.estimatedFields.length > 0 && <div className="alert">Estimated details: {result.estimatedFields.map((field) => LABELS[field]).join(", ")}. Predictions with more unknown details are less dependable.</div>}
    {result.signals?.length ? <>
      <button className="text-button" onClick={() => setOpen(!open)}>{open ? "Hide" : "Explore"} which details shifted this score</button>
      {open && <div className="signal-list">{result.signals.map((signal) => <div className="signal-row" key={signal.name}><span>{LABELS[signal.name]}</span><div className="signal-track"><span style={{ width: `${Math.max(5, Math.min(100, Math.abs(signal.change) * 500))}%` }} /></div><span>{signal.change > 0 ? "+" : ""}{(signal.change * 100).toFixed(1)} pt</span></div>)}<div className="score-caption">Comparison with the model’s typical value for one detail at a time. These are model patterns, not causes of churn.</div></div>}
    </> : null}
  </div>;
}

function PageHeader({ eyebrow, title, description }: { eyebrow: string; title: string; description: string }) {
  return <div className="page-head"><div><p className="eyebrow">{eyebrow}</p><h1 className="page-title">{title}</h1><p className="page-description">{description}</p></div><span className="preview"><span className="online" /> RESEARCH PREVIEW</span></div>;
}

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
  const [batchRows, setBatchRows] = useState<BatchRecord[]>([]);
  const [batchBusy, setBatchBusy] = useState(false);
  const [batchProgress, setBatchProgress] = useState(0);
  const [batchError, setBatchError] = useState("");
  const [importance, setImportance] = useState<Importance>([]);
  const [fileName, setFileName] = useState("");
  const composerRef = useRef<HTMLTextAreaElement>(null);

  const missing = FIELDS.filter((field) => !(field in profile));
  const captured = FIELDS.filter((field) => field in profile && profile[field] !== null);
  const estimated = FIELDS.filter((field) => field in profile && profile[field] === null);
  const complete = missing.length === 0;

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
      setBusy(true);
      try { setPrediction(await scoreProfile(updated)); }
      catch (cause) { setError(cause instanceof Error ? cause.message : "Could not connect to the prediction service."); }
      finally { setBusy(false); }
    }
  };

  const useTypical = async () => {
    const field = missing[0]; if (!field) return;
    const updated = { ...profile, [field]: null };
    setProfile(updated); setHelpOpen(false);
    const next = FIELDS.find((key) => !(key in updated));
    const response = `That’s okay — I’ll use a typical value for ${LABELS[field].toLowerCase()} and mark it as estimated.${next ? `\n\n${QUESTIONS[next]}` : " I have a complete profile now."}`;
    setMessages((prior) => [...prior, { role: "assistant", text: response }]);
    if (!next) {
      setBusy(true);
      try { setPrediction(await scoreProfile(updated)); }
      catch (cause) { setError(cause instanceof Error ? cause.message : "Could not connect to the prediction service."); }
      finally { setBusy(false); }
    }
  };

  const runQuickPrediction = async (event: FormEvent) => {
    event.preventDefault(); setError(""); setBusy(true);
    const unknownSet = new Set(quickUnknown);
    const formProfile: Profile = Object.fromEntries(FIELDS.map((field) => [field, unknownSet.has(field) ? null : Number(quickValues[field])]));
    setProfile(formProfile);
    try { setPrediction(await scoreProfile(formProfile)); }
    catch (cause) { setError(cause instanceof Error ? cause.message : "Could not connect to the prediction service."); }
    finally { setBusy(false); }
  };

  const handleBatchUpload = async (event: ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0]; if (!file) return;
    setFileName(file.name); setBatchRows([]); setBatchError(""); setBatchProgress(0);
    if (file.size > 4_000_000) { setBatchError("This file is over 4 MB. Please split it into smaller CSV files for upload."); return; }
    const raw = parseCsv(await file.text());
    if (raw.length < 2) { setBatchError("The CSV needs a header row and at least one customer."); return; }
    if (raw.length - 1 > 10000) { setBatchError("This research demo accepts up to 10,000 customers per file."); return; }
    const headers = raw[0].map((header, index) => (index === 0 ? header.replace(/^\uFEFF/, "") : header).trim());
    const missingColumns = FIELDS.filter((field) => !headers.includes(field));
    if (missingColumns.length === FIELDS.length) { setBatchError("I could not find the expected customer detail columns. Download the template below to see the right headers."); return; }
    const sourceRows = raw.slice(1).map((cells) => Object.fromEntries(headers.map((header, index) => [header, cells[index] ?? ""])));
    const predictions: BatchRecord[] = [];
    setBatchBusy(true);
    try {
      for (let offset = 0; offset < sourceRows.length; offset += 250) {
        const chunk = sourceRows.slice(offset, offset + 250);
        const profiles = chunk.map((row) => Object.fromEntries(FIELDS.map((field) => [field, row[field] ?? null])));
        const response = await fetch("/api/predict", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ profiles }) });
        const body = await response.json(); if (!response.ok) throw new Error(body.error || "The batch could not be scored.");
        chunk.forEach((row, index) => {
          const result = body.predictions[index] as Prediction;
          predictions.push({ ...row, "Churn risk score": `${(result.score * 100).toFixed(1)}%`, Prediction: result.label, "Estimated input count": String(result.estimatedFields.length) });
        });
        setBatchProgress(Math.min(100, Math.round((predictions.length / sourceRows.length) * 100)));
      }
      setBatchRows(predictions);
    } catch (cause) { setBatchError(cause instanceof Error ? cause.message : "Could not score this file."); }
    finally { setBatchBusy(false); }
  };

  const loadImportance = async () => {
    if (importance.length) return;
    try { const response = await fetch("/api/predict?view=stats"); const body = await response.json(); if (response.ok) setImportance(body.importance || []); }
    catch { /* The evaluation page remains useful without this optional chart. */ }
  };

  const resetChat = () => { setProfile({}); setMessages(INITIAL); setPrediction(null); setError(""); setComposer(""); };
  const handleComposerKey = (event: KeyboardEvent<HTMLTextAreaElement>) => { if (event.key === "Enter" && !event.shiftKey) { event.preventDefault(); void submitChat(); } };

  const pageCopy: Record<View, [string, string, string]> = {
    Predict: ["CUSTOMER WORKSPACE", "Customer risk", "Describe a customer in your own words and explore an estimated churn score."],
    "Batch scoring": ["BATCH WORKSPACE", "Score a customer list", "Upload a CSV to explore predictions across multiple customer profiles."],
    "Model performance": ["MODEL TRANSPARENCY", "How the model performs", "Review held-out results, missing-detail sensitivity, and the patterns the model uses."],
    "Project info": ["PROJECT NOTES", "About this research", "Explore the data, approach, and intended limits of this project."],
  };
  const [eyebrow, title, description] = pageCopy[view];

  return <div className="shell">
    <aside className="sidebar">
      <div className="brand"><div className="brand-icon">✦</div><div><div className="brand-title">Churn Compass</div><div className="brand-sub">Customer intelligence</div></div></div>
      <div className="side-label">WORKSPACE</div>
      <nav className="nav" aria-label="Main navigation">{NAV.map((item) => <button key={item.name} className={`nav-button ${view === item.name ? "active" : ""}`} onClick={() => { setView(item.name); if (item.name === "Model performance") void loadImportance(); }}><span className="nav-icon">{item.icon}</span>{item.name}</button>)}</nav>
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
            {prediction && <ScoreCard result={prediction} />}
            {complete && <div className="chat-actions"><button className="outline-button" onClick={resetChat}>↺ &nbsp;Start a new customer</button><span className="success-line">Profile ready · {captured.length} known · {estimated.length} estimated</span></div>}
          </section>
          <aside className="surface snapshot"><div className="snapshot-head"><div><div className="section-heading">Customer snapshot</div><div className="snapshot-sub">A live read of the conversation</div></div><span className="snapshot-mark">{complete ? "✓" : "✦"}</span></div><div className="count">{Object.keys(profile).length} of {FIELDS.length} details gathered</div><div className="progress"><span style={{ width: `${Object.keys(profile).length / FIELDS.length * 100}%` }} /></div>
            {Object.keys(profile).length ? <div className="profile-list">{FIELDS.filter((field) => field in profile).map((field) => <div className="profile-item" key={field}><div className="profile-label">{LABELS[field]}</div><div className="profile-value">{formatValue(field, profile[field])}</div>{profile[field] === null && <div className="estimated-label">◌ estimated from typical data</div>}</div>)}</div> : <div className="snapshot-empty"><span className="empty-spark">✧</span><strong>Your profile will build as we talk.</strong><p>Start with a customer’s phone habits, plan, or time with their provider.</p></div>}
            {estimated.length > 0 && <div className="snapshot-note">{estimated.length} detail(s) will use typical values and be clearly marked in the result.</div>}
            <details className="helper"><summary>How the estimate works</summary><p>Compass maps the details you share to a research model. Missing details use typical values from its training data and are called out in your result.</p></details>
          </aside>
        </div>
        </> : <div className="quick-split"><form className="surface panel quick-panel" onSubmit={runQuickPrediction}>
          <div className="section-heading">Customer details</div><p className="section-copy">A rough estimate is fine. Mark anything unknown and we’ll show what was estimated.</p>
          <div className="form-section">Account & plans</div><div className="form-grid">{FIELDS.slice(0, 4).map((field) => renderField(field))}</div>
          <div className="form-section">Monthly phone activity</div><div className="form-grid">{FIELDS.slice(4, 12).map((field) => renderField(field))}</div>
          <div className="form-section">Customer support</div><div className="form-grid">{renderField(FIELDS[12])}</div>
          <div className="action-row"><button className="primary-button" type="submit" disabled={busy}>{busy ? "Scoring…" : "Estimate customer risk"}</button><button type="button" className="outline-button" onClick={() => { setQuickValues(Object.fromEntries(FIELDS.map((field) => [field, String(RANGE[field].median)]))); setQuickUnknown([]); setPrediction(null); }}>Reset fields</button></div>
          {prediction && <ScoreCard result={prediction} />}
        </form><aside className="surface panel form-aside"><div className="project-icon">✧</div><h2 className="subhead">Make it your own</h2><p className="muted">Enter a complete profile or mark a detail as unknown. Unknown fields use typical values from the model’s training data and are clearly labeled in the result.</p><div className="form-aside-stat"><span>Prediction cutoff</span><strong>50%</strong></div><div className="form-aside-stat"><span>Inputs used</span><strong>13 details</strong></div><div className="note-card" style={{ marginTop: 16 }}>A research estimate for exploring this dataset. Not a guarantee or a customer decision.</div></aside></div>}
      </>}

      {view === "Batch scoring" && <BatchView fileName={fileName} busy={batchBusy} progress={batchProgress} error={batchError} rows={batchRows} onFile={handleBatchUpload} />}
      {view === "Model performance" && <ModelView importance={importance} />}
      {view === "Project info" && <ProjectView />}
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

const INTEGER_SET = new Set(["Account length", "Number vmail messages", "Total day calls", "Total eve calls", "Total night calls", "Total intl calls", "Customer service calls"]);

function BatchView({ fileName, busy, progress, error, rows, onFile }: { fileName: string; busy: boolean; progress: number; error: string; rows: BatchRecord[]; onFile: (event: ChangeEvent<HTMLInputElement>) => void }) {
  const template = `${FIELDS.join(",")}\n`;
  return <div className="batch-layout"><div className="surface panel"><h2 className="subhead">Score a customer list</h2><p className="muted">Upload a CSV with the customer details you have. Blank cells are filled with typical values and marked in the results.</p><div className="upload-box"><div style={{ fontSize: 22, marginBottom: 8, color: "var(--teal)" }}>⇧</div><strong style={{ fontSize: 12 }}>Choose your customer file</strong><p className="section-copy">CSV file · up to 10,000 customers</p><input className="file-input" type="file" accept=".csv,text/csv" onChange={onFile} /></div><div className="action-row"><a className="outline-button" href={`data:text/csv;charset=utf-8,${encodeURIComponent(template)}`} download="churn_customer_template.csv">↓ &nbsp;Download CSV template</a></div>{fileName && <p className="section-copy">Selected: {fileName}</p>}{busy && <><div className="progress"><span style={{ width: `${progress}%` }} /></div><div className="loading">Scoring customer profiles… {progress}%</div></>}{error && <div className="error">{error}</div>}</div>
    <div className="surface panel"><h2 className="subhead">What you’ll get</h2><div className="profile-list"><div className="profile-item"><div className="profile-label">Estimated churn score</div><div className="profile-value">A percentage score for each row</div></div><div className="profile-item"><div className="profile-label">Plain-language prediction</div><div className="profile-value">Likely / less likely to churn</div></div><div className="profile-item"><div className="profile-label">Unknown detail count</div><div className="profile-value">Shows where typical values were used</div></div></div><p className="section-copy" style={{ marginTop: 13 }}>Your file is processed for this request and is not added to the research dataset.</p></div>
    {rows.length > 0 && <div className="surface panel batch-results"><div className="results-head"><div><h2 className="subhead">Results are ready</h2><p className="section-copy">{rows.length.toLocaleString()} customer profiles scored.</p></div><button className="primary-button" onClick={() => saveCsv("churn_predictions.csv", rows)}>↓ &nbsp;Download results</button></div><div className="table-wrap"><table className="data-table"><thead><tr>{Object.keys(rows[0]).map((header) => <th key={header}>{header}</th>)}</tr></thead><tbody>{rows.slice(0, 30).map((row, index) => <tr key={index}>{Object.keys(rows[0]).map((header) => <td key={header}>{row[header]}</td>)}</tr>)}</tbody></table></div>{rows.length > 30 && <p className="section-copy">Showing the first 30 rows. Download includes all results.</p>}</div>}
  </div>;
}

function ModelView({ importance }: { importance: Importance }) {
  const metrics: [string, number, string][] = [
    ["Accuracy", METRICS.accuracy, "Overall correct predictions"], ["Churn recall", METRICS.churn_recall, "Churners correctly identified"], ["Churn precision", METRICS.churn_precision, "Flagged customers who churned"],
    ["Churn F1", METRICS.churn_f1, "Precision and recall balance"], ["ROC-AUC", METRICS.roc_auc, "Ranking separation"], ["PR-AUC", METRICS.pr_auc_average_precision, "Imbalance-aware ranking"],
  ];
  const matrix = METRICS.confusion_matrix_labels_0_1;
  return <div className="model-layout"><div className="metric-grid">{metrics.map(([label, value, note]) => <div className="metric-card" key={label}><div className="metric-label">{label}</div><div className="metric-value">{(value * 100).toFixed(1)}%</div><div className="metric-note">{note}</div></div>)}</div>
    <div className="note-card">Churn is {(metadata.dataset.churn_rate * 100).toFixed(1)}% of this dataset. A majority-class baseline reaches {(metadata.baseline.test_accuracy * 100).toFixed(1)}% accuracy but catches no churners. Accuracy alone can hide that.</div>
    <div className="model-grid"><section className="surface panel"><h2 className="subhead">Held-out results</h2><p className="section-copy">The model trained on {metadata.dataset.rows.toLocaleString()} records. These results use a stratified {Math.round(metadata.split.test_size * 100)}% test split ({metadata.split.test_rows} customers) held aside from training.</p><div className="table-wrap" style={{ marginTop: 15 }}><table className="data-table confusion"><thead><tr><th>Actual \ Predicted</th><th>No churn</th><th>Churn</th></tr></thead><tbody><tr><th>No churn</th><td>{matrix[0][0]}</td><td>{matrix[0][1]}</td></tr><tr><th>Churn</th><td>{matrix[1][0]}</td><td>{matrix[1][1]}</td></tr></tbody></table></div></section>
      <section className="surface panel"><h2 className="subhead">When details are unknown</h2><p className="section-copy">Test profiles had details hidden and replaced with training-data medians.</p><div className="unknown-grid">{Object.entries(MEDIAN_CHECK).map(([count, result]) => <div className="unknown-card" key={count}><div className="metric-label">{count} unknown details</div><div className="mini-metric"><span>Churn recall</span><strong>{(result.churn_recall.mean * 100).toFixed(1)}%</strong></div><div className="mini-metric"><span>Churn F1</span><strong>{(result.churn_f1.mean * 100).toFixed(1)}%</strong></div><div className="mini-metric"><span>Accuracy</span><strong>{(result.accuracy.mean * 100).toFixed(1)}%</strong></div></div>)}</div></section></div>
    <section className="surface panel importance"><h2 className="subhead">Signals the model uses most</h2><p className="section-copy">Relative importance within this trained model. These patterns are not proof that a detail causes churn.</p>{importance.length ? <div className="bar-chart">{importance.slice(0, 10).map((item) => <div className="bar-item" key={item.name}><span>{LABELS[item.name] || item.name}</span><div className="bar-track"><span style={{ width: `${item.value / importance[0].value * 100}%` }} /></div><span>{(item.value * 100).toFixed(1)}%</span></div>)}</div> : <p className="loading">Loading model signals…</p>}</section>
    <p className="disclaimer">This model was evaluated on this telecom dataset only. It is an educational research prototype; don’t use its estimate as the sole basis for contacting, pricing, or treating real customers.</p>
  </div>;
}

function ProjectView() {
  return <div className="project-layout"><section className="surface project-intro"><div className="eyebrow">THE RESEARCH</div><h2 className="project-title">A practical experiment in customer retention.</h2><p className="project-lead">This project explores whether account history, plan choices, calling habits, and support contacts can help identify telecom customers who may be at risk of leaving.</p></section>
    <div className="metric-grid"><div className="metric-card"><div className="metric-label">Customer records</div><div className="metric-value">{metadata.dataset.rows.toLocaleString()}</div><div className="metric-note">Public telecom dataset</div></div><div className="metric-card"><div className="metric-label">Customers who churned</div><div className="metric-value">{metadata.dataset.churn_rows.toLocaleString()}</div><div className="metric-note">{(metadata.dataset.churn_rate * 100).toFixed(1)}% of the source rows</div></div><div className="metric-card"><div className="metric-label">Customer details</div><div className="metric-value">13</div><div className="metric-note">Plans, usage, tenure, support</div></div></div>
    <div className="project-grid"><section className="surface panel"><div className="project-icon">⌕</div><h2 className="subhead">How it works</h2><p className="muted">The app gathers a customer profile, derives usage charges from minutes, fills any unknown values with training-data medians, and scores the completed profile using a Random Forest model.</p><div className="step-list"><div><b>01</b><span>Describe what you know</span></div><div><b>02</b><span>Review captured details</span></div><div><b>03</b><span>Explore the estimate</span></div></div></section><section className="surface panel"><div className="project-icon">◎</div><h2 className="subhead">Research limits</h2><p className="muted">The model reflects this telecom dataset and its customer population. It may not generalize to another provider or industry. A score is an estimate, not a guarantee or a causal explanation.</p><div className="note-card" style={{ marginTop: 15 }}>Use this demo to explore model behavior and research questions, not to make decisions about real customers.</div><a className="paper-link" href="/Customer_Churn_Project_Paper.pdf" target="_blank" rel="noreferrer">Read the project paper ↗</a></section></div>
  </div>;
}
