"use client";

// AI Analyst — the Phase 6 restricted analytics layer in the Compass chat
// language. Python computes every number through the 8 whitelisted functions;
// the language model (when a key is configured on the server) only picks a
// function and narrates the computed facts, with grounding validation. Without
// a key the deterministic planner returns verified facts directly.

import { useEffect, useRef, useState } from "react";
import {
  AnalystAnswerPayload,
  Filters,
  Portfolio,
  filterRows,
  postJson,
} from "@/lib/portfolio";

const MAX_CHAT_ENTRIES = 10;

// F-4 provider privacy disclosure (same wording as the reference app).
const PRIVACY_DISCLOSURE =
  "When language-model narration is configured and used, the analyst request sends: the dataset's column names, dataset metadata and risk thresholds, and the aggregate analytical facts computed from your data. Raw customer rows and customer identifiers are not sent through the explanation path. The request does leave this application and is processed by the configured external AI provider (Google Gemini) under its own data policies.";

type Entry = { role: "assistant" | "user"; text: string; answer?: AnalystAnswerPayload };

function suggestedQuestions(portfolio: Portfolio): string[] {
  const questions = [
    "How many customers are in this dataset and how many are predicted to churn?",
    "What are the most common characteristics among high-risk customers?",
    "How is predicted churn probability distributed across the portfolio?",
  ];
  const hasLabels = portfolio.rows.some((row) => row.Observed_Churn === true || row.Observed_Churn === false);
  if (hasLabels) questions.push("How do observed churn outcomes compare with the model's predictions?");
  const dimensions = Object.keys(portfolio.groupRates);
  if (dimensions.length) questions.push(`Which ${dimensions[0].toLowerCase()} group has the highest churn rate?`);
  return questions;
}

export default function AnalystView({ portfolio, filters }: { portfolio: Portfolio | null; filters: Filters }) {
  const [entries, setEntries] = useState<Entry[]>([]);
  const [composer, setComposer] = useState("");
  const [stage, setStage] = useState<"idle" | "planning" | "computing" | "narrating">("idle");
  const [providerConfigured, setProviderConfigured] = useState<boolean | null>(null);
  const scroller = useRef<HTMLDivElement>(null);

  useEffect(() => {
    fetch("/api/analyst").then((response) => response.json()).then((body) => setProviderConfigured(Boolean(body?.provider?.configured))).catch(() => setProviderConfigured(false));
  }, []);

  useEffect(() => { scroller.current?.scrollTo({ top: scroller.current.scrollHeight }); }, [entries.length, stage]);

  if (!portfolio) {
    return <div className="surface panel empty-state">
      <div className="project-icon">✦</div>
      <h2 className="subhead">The analyst needs a dataset</h2>
      <p className="muted">Load the built-in sample or upload your CSV on the Portfolio page, then ask questions in plain language. Every number the analyst quotes is computed in Python from your scored customers.</p>
    </div>;
  }

  const ask = async (question: string) => {
    const text = question.trim();
    if (!text || stage !== "idle") return;
    setComposer("");
    setEntries((current) => [...current, { role: "user", text }]);
    setStage("planning");
    try {
      setStage("computing");
      const rows = filterRows(portfolio.rows, filters);
      const answer = await postJson<AnalystAnswerPayload>("/api/analyst", { rows, thresholds: portfolio.thresholds, question: text });
      setStage("narrating");
      setEntries((current) => [...current, { role: "assistant", text: answer.explanation || "", answer }]);
    } catch (cause) {
      setEntries((current) => [...current, { role: "assistant", text: "", answer: { status: "error", question: text, function: null, facts: null, caveats: [], explanation: null, note: cause instanceof Error ? cause.message : "The analysis could not be completed.", elapsed_seconds: 0 } }]);
    } finally {
      setStage("idle");
      setEntries((current) => current.slice(-MAX_CHAT_ENTRIES));
    }
  };

  return <div className="split">
    <section className="surface chat-panel">
      <div className="chat-top">
        <div className="guide-id">
          <div className="guide-avatar">✦</div>
          <div>
            <div className="guide-name">Analyst <span className="guide-role">· data analyst</span></div>
            <div className="guide-status"><span className="online" />{stage === "idle" ? "Answers from approved analytics functions only" : stageLabel(stage)}</div>
          </div>
        </div>
        <span className="chat-tag">{providerConfigured === null ? "CHECKING PROVIDER…" : providerConfigured ? "AI NARRATION ON" : "FACTS-ONLY MODE"}</span>
      </div>
      <div className="messages" aria-live="polite">
        {!entries.length && <div className="message"><div className="message-avatar">✦</div><div className="message-body"><span className="message-by">Analyst</span>Ask a question about your customers and I’ll compute the answer from the approved analytics functions — overview, risk bands, group comparisons, distributions, and more. Try one of the suggestions below.</div></div>}
        {entries.map((entry, index) => entry.role === "user"
          ? <div className="message user" key={index}><div className="message-avatar">🙂</div><div className="message-body">{entry.text}</div></div>
          : <AssistantEntry key={index} entry={entry} />)}
        {stage !== "idle" && <div className="message"><div className="message-avatar">✦</div><div className="message-body"><span className="message-by">Analyst</span><span className="typing"><i /><i /><i /></span> <span style={{ fontSize: 10, color: "var(--muted)" }}>{stageLabel(stage)}</span></div></div>}
      </div>
      {!entries.length && <div className="quick-row" style={{ marginTop: 12 }}>
        {suggestedQuestions(portfolio).map((question) => <button className="quick-chip" key={question} onClick={() => void ask(question)}>{question}</button>)}
      </div>}
      <form className="composer" onSubmit={(event) => { event.preventDefault(); void ask(composer); }}>
        <textarea value={composer} onChange={(event) => setComposer(event.target.value)} placeholder="Ask about your customers…" rows={1} aria-label="Ask the analyst" onKeyDown={(event) => { if (event.key === "Enter" && !event.shiftKey) { event.preventDefault(); void ask(composer); } }} />
        <button className="send-button" type="submit" disabled={!composer.trim() || stage !== "idle"} aria-label="Send question">↑</button>
      </form>
      <div className="chat-under">
        <span>Python computes every number; narration only describes verified facts.</span>
        <button className="text-button" onClick={() => { setEntries([]); setComposer(""); }}>Clear chat</button>
      </div>
    </section>
    <aside className="surface snapshot">
      <div className="snapshot-head"><div><div className="section-heading">How this works</div><div className="snapshot-sub">Restricted by design</div></div><span className="snapshot-mark">✓</span></div>
      <div className="profile-list" style={{ marginTop: 12 }}>
        {[
          ["Approved functions only", "8 analytics functions; no arbitrary code, SQL, or file access"],
          ["Computed in Python", "Statistics come from your scored dataset, never from the language model"],
          ["Grounded narration", "AI text is validated against the computed facts; ungrounded text is withheld"],
          ["Facts-only fallback", "Without a configured key, verified facts are still returned"],
          ["Bounded history", `The last ${MAX_CHAT_ENTRIES} exchanges are kept; Clear chat removes them`],
        ].map(([label, note]) => <div className="profile-item" key={label}><div className="profile-label">{label}</div><div className="profile-value" style={{ fontWeight: 400, fontSize: 11, color: "#c3d2d6" }}>{note}</div></div>)}
      </div>
      <div className="snapshot-note">{PRIVACY_DISCLOSURE}</div>
    </aside>
  </div>;
}

function stageLabel(stage: string) {
  if (stage === "planning") return "Choosing an approved function…";
  if (stage === "computing") return "Computing from your data…";
  if (stage === "narrating") return "Checking the explanation…";
  return "Ready";
}

function AssistantEntry({ entry }: { entry: Entry }) {
  const answer = entry.answer;
  const statusLabels: Record<string, string> = {
    answered: "ANSWERED · GROUNDED",
    facts_only: "COMPUTED FACTS",
    cannot_answer: "CANNOT ANSWER FROM THIS DATA",
    error: "ANALYSIS ERROR",
  };
  return <div className="message"><div className="message-avatar">✦</div>
    <div className="message-body" style={{ maxWidth: "100%" }}>
      <span className="message-by">Analyst · {answer ? statusLabels[answer.status] : ""}{answer?.elapsed_seconds ? ` · ${answer.elapsed_seconds}s` : ""}</span>
      {answer?.explanation && <div style={{ marginBottom: 8 }}>{answer.explanation}</div>}
      {answer?.facts && <FactsTable facts={answer.facts} />}
      {answer?.caveats?.map((caveat) => <div className="score-caption" key={caveat} style={{ marginTop: 6 }}>Note: {caveat}</div>)}
      {answer?.note && <div className="score-caption" style={{ marginTop: 6, color: "var(--amber)" }}>{answer.note}</div>}
      {!answer?.explanation && !answer?.facts && answer?.note && <div>{entry.text}</div>}
    </div>
  </div>;
}

function FactsTable({ facts }: { facts: Record<string, unknown> }) {
  const rows: [string, unknown][] = [];
  const walk = (prefix: string, value: unknown) => {
    if (Array.isArray(value)) {
      value.forEach((item, index) => walk(`${prefix}[${index + 1}]`, item));
    } else if (value && typeof value === "object") {
      for (const [key, item] of Object.entries(value)) walk(prefix ? `${prefix} · ${key}` : key, item);
    } else {
      rows.push([prefix, value]);
    }
  };
  walk("", facts);
  const display = (value: unknown) => {
    if (value === null || value === undefined) return "—";
    if (typeof value === "boolean") return value ? "true" : "false";
    if (typeof value === "number") return Number.isInteger(value) ? value.toLocaleString() : value.toFixed(4).replace(/\.?0+$/, "");
    return String(value);
  };
  return <div className="table-wrap" style={{ marginTop: 8, maxHeight: 320 }}>
    <table className="data-table">
      <thead><tr><th style={{ textAlign: "left" }}>Computed fact</th><th>Value</th></tr></thead>
      <tbody>{rows.slice(0, 60).map(([key, value]) => <tr key={key}><td style={{ textAlign: "left", color: "#b8c7cc" }}>{key}</td><td>{display(value)}</td></tr>)}</tbody>
    </table>
  </div>;
}
