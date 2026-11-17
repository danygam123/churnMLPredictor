// Shared portfolio state, V2 semantics ports, CSV utilities, and exports.
//
// The V2 ML engine (ml/) remains the source of truth for scoring, band
// defaults, explanations, recommendations, simulation, and analytics. These
// helpers mirror small, deterministic pieces of that behavior in the browser —
// risk-band classification (ml.risk_intelligence.classify_risk), portfolio
// headline arithmetic (portfolio_summary), and the Phase 7 filter semantics —
// so the interface stays instant while the model work stays server-side.

import { FIELDS } from "./chat";

export type ScoredRow = Record<string, string | number | boolean | null>;

export type Thresholds = { medium_threshold: number; high_threshold: number };

export type Summary = {
  customers_analyzed: number;
  predicted_churners: number;
  average_predicted_probability: number;
  high_risk_customers: number;
  observed_churn_rate: number | null;
  observed_label_count: number;
};

export type GroupRate = { group: string; rate: number; customers: number };

export type Filters = {
  risks: string[];
  search: string;
  internationalPlan: string[];
  voiceMailPlan: string[];
  accountLength: [number | null, number | null];
  charge: [number | null, number | null];
};

export type Portfolio = {
  name: string;
  isSample: boolean;
  columns: string[];
  rows: ScoredRow[];
  summary: Summary;
  warnings: string[];
  notes: string[];
  labelColumnPresent: boolean;
  idSourceColumn: string | null;
  predictionThreshold: number;
  thresholds: Thresholds;
  bandCounts: Record<string, number>;
  groupRates: Record<string, GroupRate[]>;
};

export const ALL_BANDS = ["High Risk", "Medium Risk", "Low Risk"];
export const DEFAULT_FILTERS: Filters = {
  risks: [...ALL_BANDS],
  search: "",
  internationalPlan: [],
  voiceMailPlan: [],
  accountLength: [null, null],
  charge: [null, null],
};

export const BAND_COLORS: Record<string, string> = {
  "High Risk": "var(--red)",
  "Medium Risk": "var(--amber)",
  "Low Risk": "var(--green)",
};
export const BAND_CLASS: Record<string, string> = {
  "High Risk": "high",
  "Medium Risk": "medium",
  "Low Risk": "low",
};

// ---------------------------------------------------------------------------
// V2 semantics ports (display layer; the engine remains authoritative)
// ---------------------------------------------------------------------------

/** Mirrors ml.risk_intelligence.classify_risk exactly. */
export function classifyRisk(probability: number, thresholds: Thresholds): string {
  if (probability >= thresholds.high_threshold) return "High Risk";
  if (probability >= thresholds.medium_threshold) return "Medium Risk";
  return "Low Risk";
}

export function applyBands(rows: ScoredRow[], thresholds: Thresholds): ScoredRow[] {
  return rows.map((row) => ({
    ...row,
    Risk_Category: classifyRisk(Number(row.Churn_Probability), thresholds),
  }));
}

export function bandCounts(rows: ScoredRow[]): Record<string, number> {
  const counts: Record<string, number> = { "High Risk": 0, "Medium Risk": 0, "Low Risk": 0 };
  for (const row of rows) {
    const band = String(row.Risk_Category);
    if (band in counts) counts[band] += 1;
  }
  return counts;
}

/** Mirrors ml.risk_intelligence.portfolio_summary arithmetic. */
export function summarize(rows: ScoredRow[]): Summary {
  if (!rows.length) {
    return {
      customers_analyzed: 0,
      predicted_churners: 0,
      average_predicted_probability: 0,
      high_risk_customers: 0,
      observed_churn_rate: null,
      observed_label_count: 0,
    };
  }
  const probabilities = rows.map((row) => Number(row.Churn_Probability));
  const labels = rows
    .map((row) => row.Observed_Churn)
    .filter((value) => value === true || value === false) as boolean[];
  return {
    customers_analyzed: rows.length,
    predicted_churners: rows.filter((row) => row.Predicted_Churner === true).length,
    average_predicted_probability:
      probabilities.reduce((total, value) => total + value, 0) / rows.length,
    high_risk_customers: rows.filter((row) => row.Risk_Category === "High Risk").length,
    observed_churn_rate: labels.length ? labels.filter(Boolean).length / labels.length : null,
    observed_label_count: labels.length,
  };
}

/** Mirrors ml.risk_intelligence.churn_rate_by_group for plan/dimension columns. */
export function groupRates(rows: ScoredRow[], column: string): GroupRate[] {
  const buckets = new Map<string, { total: number; count: number }>();
  for (const row of rows) {
    const key = row[column];
    if (key === null || key === undefined || String(key).trim() === "") continue;
    const bucket = buckets.get(String(key)) ?? { total: 0, count: 0 };
    const labeled = row.Observed_Churn;
    if (typeof labeled === "boolean") {
      bucket.total += labeled ? 1 : 0;
      bucket.count += 1;
    } else {
      bucket.total += Number(row.Churn_Probability);
      bucket.count += 1;
    }
    buckets.set(String(key), bucket);
  }
  return [...buckets.entries()]
    .map(([group, bucket]) => ({ group, rate: bucket.total / bucket.count, customers: bucket.count }))
    .sort((a, b) => b.rate - a.rate)
    .slice(0, 25);
}

/** Mirrors the Phase 7 filtered_view semantics for the supported schema. */
export function filterRows(rows: ScoredRow[], filters: Filters): ScoredRow[] {
  let view = rows.filter((row) => filters.risks.includes(String(row.Risk_Category)));
  const search = filters.search.trim().toLowerCase();
  if (search) {
    view = view.filter((row) =>
      Object.entries(row).some(
        ([column, value]) =>
          (column === "Customer ID" || FIELDS.includes(column) || isBusinessColumn(column)) &&
          value !== null &&
          String(value).toLowerCase().includes(search),
      ),
    );
  }
  if (filters.internationalPlan.length) {
    view = view.filter((row) => filters.internationalPlan.includes(displayPlan(row["International plan"])));
  }
  if (filters.voiceMailPlan.length) {
    view = view.filter((row) => filters.voiceMailPlan.includes(displayPlan(row["Voice mail plan"])));
  }
  view = numericRange(view, "Account length", filters.accountLength);
  view = numericRange(view, "Derived Usage Charge Total", filters.charge);
  return view;
}

function isBusinessColumn(column: string) {
  return ["Churn", "Contract", "Contract Type", "Payment Method"].includes(column);
}

export function displayPlan(value: string | number | boolean | null | undefined): string {
  if (value === null || value === undefined) return "";
  if (typeof value === "boolean") return value ? "Yes" : "No";
  const text = String(value).trim().toLowerCase();
  if (["1", "1.0", "true", "yes", "y"].includes(text)) return "Yes";
  if (["0", "0.0", "false", "no", "n"].includes(text)) return "No";
  return String(value);
}

function numericRange(
  rows: ScoredRow[],
  column: string,
  [low, high]: [number | null, number | null],
): ScoredRow[] {
  if (low === null && high === null) return rows;
  return rows.filter((row) => {
    const value = Number(row[column]);
    if (Number.isNaN(value)) return true; // keep rows without the value, like the reference app
    if (low !== null && value < low) return false;
    if (high !== null && value > high) return false;
    return true;
  });
}

// ---------------------------------------------------------------------------
// CSV utilities (shared by every view)
// ---------------------------------------------------------------------------

export function parseCsv(text: string): string[][] {
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

export function csvCell(value: unknown) {
  const text = String(value ?? "");
  return /[",\r\n]/.test(text) ? `"${text.replaceAll('"', '""')}"` : text;
}

export function saveCsv(filename: string, rows: Record<string, unknown>[]) {
  if (!rows.length) return;
  const headers = Object.keys(rows[0]);
  const content = [headers.map(csvCell).join(","), ...rows.map((row) => headers.map((key) => csvCell(row[key])).join(","))].join("\r\n");
  const url = URL.createObjectURL(new Blob([content], { type: "text/csv;charset=utf-8" }));
  const anchor = document.createElement("a"); anchor.href = url; anchor.download = filename; anchor.click(); URL.revokeObjectURL(url);
}

export function saveCsvText(filename: string, text: string) {
  const url = URL.createObjectURL(new Blob([text], { type: "text/csv;charset=utf-8" }));
  const anchor = document.createElement("a"); anchor.href = url; anchor.download = filename; anchor.click(); URL.revokeObjectURL(url);
}

// ---------------------------------------------------------------------------
// Exports (the five required families)
// ---------------------------------------------------------------------------

export function customerExportRows(rows: ScoredRow[]): Record<string, unknown>[] {
  const columns = [
    "Customer ID",
    ...FIELDS,
    "Churn_Probability",
    "Prediction",
    "Risk_Category",
    "Estimated_Input_Count",
    "Derived Usage Charge Total",
    "Churn",
  ].filter((column) => rows.some((row) => column in row));
  return rows.map((row) => {
    const record: Record<string, unknown> = {};
    for (const column of columns) record[column] = row[column] ?? "";
    return record;
  });
}

export function shapExportRows(explanation: ShapExplanation, customerId: string): Record<string, unknown>[] {
  return explanation.all_contributions.map((item) => ({
    customer_id: customerId,
    feature: item.feature,
    value: item.value ?? "",
    input_status: item.input_status,
    contribution: item.contribution,
    direction: item.direction,
  }));
}

export function recommendationExportRows(recommendations: Recommendation[]): Record<string, unknown>[] {
  return recommendations.map((item) => ({
    customer_id: item.customer_id,
    rule_id: item.rule_id,
    rule_name: item.rule_name,
    title: item.title,
    priority: item.priority,
    priority_score: item.priority_score,
    churn_probability: item.churn_probability,
    risk_category: item.risk_category,
    action: item.action,
    why_triggered: item.why_triggered,
    triggering_characteristics: JSON.stringify(item.triggering_characteristics ?? {}),
    model_context: JSON.stringify(item.model_context ?? []),
    model_context_status: item.model_context_status,
    objective: item.objective,
    human_approval_required: item.human_approval_required,
    limitations: item.limitations,
  }));
}

// ---------------------------------------------------------------------------
// API response types
// ---------------------------------------------------------------------------

export type ShapContribution = {
  feature: string;
  value: string | number | boolean | null;
  input_status: string;
  contribution: number;
  direction: "increases" | "decreases" | "neutral";
  magnitude: number;
};

export type ShapExplanation = {
  churn_probability: number;
  base_value: number;
  contributions_sum: number;
  reconstructed_probability: number;
  max_evals: number;
  explanation_method: string;
  estimated_features: string[];
  genuinely_missing_features: string[];
  rejected_or_coerced_features: string[];
  top_contributions: ShapContribution[];
  increasing_risk_factors: ShapContribution[];
  decreasing_risk_factors: ShapContribution[];
  limited_influence_factors: ShapContribution[];
  all_contributions: ShapContribution[];
  summary: string;
  prediction: string;
  risk_category: string;
  decision_threshold_used: number;
};

export type Recommendation = {
  customer_id: string;
  rule_id: string;
  rule_name: string;
  title: string;
  priority: string;
  priority_score: number;
  churn_probability: number;
  risk_category: string;
  action: string;
  why_triggered: string;
  triggering_characteristics: Record<string, string | number>;
  related_shap_features: string[];
  model_context: { feature: string; direction: string; contribution: number; value: string | number | null }[];
  model_context_status: string;
  objective: string;
  human_approval_required: boolean;
  limitations: string;
};

export type SimulationResult = {
  name: string;
  original_probability: number;
  simulated_probability: number;
  difference: number;
  difference_percentage_points: number;
  original_prediction: string;
  simulated_prediction: string;
  baseline_risk_category: string;
  simulated_risk_category: string;
  decision_threshold: number;
  changed_features: Record<string, { original_value: unknown; simulated_value: unknown }>;
  scenario_inputs: Record<string, unknown>;
  estimated_features: string[];
  out_of_training_range_features: string[];
};

export type AnalystAnswerPayload = {
  status: "answered" | "cannot_answer" | "facts_only" | "error";
  question: string;
  function: string | null;
  facts: Record<string, unknown> | null;
  caveats: string[];
  explanation: string | null;
  note: string | null;
  elapsed_seconds: number;
};

export async function postJson<T>(url: string, body: unknown): Promise<T> {
  const response = await fetch(url, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
  const payload = await response.json();
  if (!response.ok) throw new Error((payload as { error?: string }).error || "The request could not be completed.");
  return payload as T;
}
