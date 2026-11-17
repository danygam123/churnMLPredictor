// Unit tests for the client-side ports of the V2 engine semantics in lib/portfolio.ts.
// The engine (ml/) stays the source of truth; these tests pin the mirrored
// display-layer arithmetic so it cannot silently drift from the reference app.

import { describe, expect, it } from "vitest";
import {
  ALL_BANDS,
  DEFAULT_FILTERS,
  ScoredRow,
  applyBands,
  bandCounts,
  classifyRisk,
  csvCell,
  customerExportRows,
  displayPlan,
  filterRows,
  groupRates,
  parseCsv,
  saveCsv,
  shapExportRows,
  summarize,
} from "./portfolio";
import { FIELDS } from "./chat";

const T = { medium_threshold: 0.07727272727272727, high_threshold: 0.5 };

function row(overrides: Partial<ScoredRow> = {}): ScoredRow {
  return {
    "Customer ID": "C-0001",
    "Account length": 100,
    "International plan": 0,
    "Voice mail plan": 1,
    "Number vmail messages": 5,
    "Total day minutes": 180,
    "Total day calls": 100,
    "Total eve minutes": 180,
    "Total eve calls": 100,
    "Total night minutes": 180,
    "Total night calls": 100,
    "Total intl minutes": 10,
    "Total intl calls": 4,
    "Customer service calls": 1,
    "Derived Usage Charge Total": 50,
    Churn_Probability: 0.3,
    Prediction: "Less likely to churn",
    Predicted_Churner: false,
    Risk_Category: "Low Risk",
    Estimated_Input_Count: 0,
    Observed_Churn: false,
    ...overrides,
  };
}

describe("classifyRisk (mirror of ml.risk_intelligence.classify_risk)", () => {
  it("classifies the three bands with >= boundaries", () => {
    expect(classifyRisk(0.0, T)).toBe("Low Risk");
    expect(classifyRisk(T.medium_threshold - 0.0001, T)).toBe("Low Risk");
    expect(classifyRisk(T.medium_threshold, T)).toBe("Medium Risk");
    expect(classifyRisk(0.4999, T)).toBe("Medium Risk");
    expect(classifyRisk(0.5, T)).toBe("High Risk");
    expect(classifyRisk(1.0, T)).toBe("High Risk");
  });
});

describe("applyBands / bandCounts", () => {
  it("re-classifies rows and counts every row exactly once", () => {
    const rows = [row({ Churn_Probability: 0.01 }), row({ Churn_Probability: 0.2 }), row({ Churn_Probability: 0.9 })];
    const banded = applyBands(rows, T);
    expect(banded.map((r) => r.Risk_Category)).toEqual(["Low Risk", "Medium Risk", "High Risk"]);
    const counts = bandCounts(banded);
    expect(Object.keys(counts).sort()).toEqual([...ALL_BANDS].sort());
    expect(Object.values(counts).reduce((a, b) => a + b, 0)).toBe(3);
  });
});

describe("summarize (mirror of portfolio_summary)", () => {
  it("returns zeros for an empty view", () => {
    const empty = summarize([]);
    expect(empty.customers_analyzed).toBe(0);
    expect(empty.observed_churn_rate).toBeNull();
  });
  it("computes churners, average probability, high-risk count, observed rate", () => {
    const rows = [
      row({ Churn_Probability: 0.6, Predicted_Churner: true, Risk_Category: "High Risk", Observed_Churn: true }),
      row({ Churn_Probability: 0.2, Observed_Churn: false }),
      row({ Churn_Probability: 0.4, Observed_Churn: null }),
    ];
    const summary = summarize(rows);
    expect(summary.customers_analyzed).toBe(3);
    expect(summary.predicted_churners).toBe(1);
    expect(summary.average_predicted_probability).toBeCloseTo(0.4);
    expect(summary.high_risk_customers).toBe(1);
    expect(summary.observed_label_count).toBe(2);
    expect(summary.observed_churn_rate).toBeCloseTo(0.5);
  });
});

describe("groupRates (mirror of churn_rate_by_group)", () => {
  it("uses observed labels when present and predicted probability otherwise", () => {
    const rows = [
      row({ "International plan": 1, Observed_Churn: true }),
      row({ "International plan": 1, Observed_Churn: false }),
      row({ "International plan": 0, Observed_Churn: null, Churn_Probability: 0.8 }),
    ];
    const rates = groupRates(rows, "International plan");
    const byGroup = Object.fromEntries(rates.map((r) => [r.group, r]));
    expect(byGroup["1"]).toMatchObject({ rate: 0.5, customers: 2 });
    expect(byGroup["0"]).toMatchObject({ rate: 0.8, customers: 1 });
  });
  it("sorts descending and caps at 25 groups", () => {
    const rows = Array.from({ length: 30 }, (_, i) => row({ "Customer service calls": i, Observed_Churn: i % 2 === 0 }));
    const rates = groupRates(rows, "Customer service calls");
    expect(rates.length).toBeLessThanOrEqual(25);
    for (let i = 1; i < rates.length; i++) expect(rates[i - 1].rate).toBeGreaterThanOrEqual(rates[i].rate);
  });
});

describe("filterRows (Phase 7 filter semantics)", () => {
  const rows = [
    row({ "Customer ID": "C-1", "International plan": 1, Risk_Category: "High Risk", "Account length": 200, Observed_Churn: true }),
    row({ "Customer ID": "C-2", "International plan": 0, Risk_Category: "Low Risk", "Account length": 50 }),
    row({ "Customer ID": "C-3", "International plan": 1, Risk_Category: "Medium Risk", "Account length": 100 }),
  ];
  it("filters by risk band, plan, numeric range, and search", () => {
    expect(filterRows(rows, { ...DEFAULT_FILTERS, risks: ["High Risk"] }).map((r) => r["Customer ID"])).toEqual(["C-1"]);
    expect(filterRows(rows, { ...DEFAULT_FILTERS, internationalPlan: ["Yes"] }).length).toBe(2);
    expect(filterRows(rows, { ...DEFAULT_FILTERS, accountLength: [60, null] }).map((r) => r["Customer ID"])).toEqual(["C-1", "C-3"]);
    expect(filterRows(rows, { ...DEFAULT_FILTERS, search: "c-2" }).map((r) => r["Customer ID"])).toEqual(["C-2"]);
  });
  it("returns an empty view for an empty band selection (the UI chip guard re-selects all bands first)", () => {
    expect(filterRows(rows, { ...DEFAULT_FILTERS, risks: [] })).toHaveLength(0);
  });
});

describe("displayPlan", () => {
  it("normalizes the supported Yes/No representations", () => {
    expect(displayPlan(1)).toBe("Yes");
    expect(displayPlan(0)).toBe("No");
    expect(displayPlan("yes")).toBe("Yes");
    expect(displayPlan("false")).toBe("No");
    expect(displayPlan(null)).toBe("");
  });
});

describe("CSV utilities", () => {
  it("parses quoted CSV including embedded separators", () => {
    const table = parseCsv('a,b\n"x,y",2\n"say ""hi""",3\n');
    expect(table[0]).toEqual(["a", "b"]);
    expect(table[1]).toEqual(["x,y", "2"]);
    expect(table[2]).toEqual(['say "hi"', "3"]);
  });
  it("quotes cells that contain commas or quotes", () => {
    expect(csvCell("plain")).toBe("plain");
    expect(csvCell("a,b")).toBe('"a,b"');
    expect(csvCell('say "hi"')).toBe('"say ""hi"""');
  });
  it("builds the customer prediction export with the documented filename inputs", () => {
    const rows = customerExportRows([row()]);
    expect(Object.keys(rows[0])).toContain("Churn_Probability");
    expect(Object.keys(rows[0])).toContain("Risk_Category");
    expect(Object.keys(rows[0])).toContain("Prediction");
  });
  it("builds the SHAP export rows", () => {
    const explanation = {
      all_contributions: [{ feature: "Customer service calls", value: 4, input_status: "provided", contribution: 0.05, direction: "increases" }],
    } as never;
    const rows = shapExportRows(explanation, "C-9");
    expect(rows[0]).toMatchObject({ customer_id: "C-9", feature: "Customer service calls", contribution: 0.05 });
  });
  it("saveCsv is a no-op for an empty result set", () => {
    expect(() => saveCsv("x.csv", [])).not.toThrow();
  });
});

describe("schema contract", () => {
  it("exposes exactly the 13 engine input fields in order", () => {
    expect(FIELDS.length).toBe(13);
    expect(FIELDS[0]).toBe("Account length");
    expect(FIELDS[12]).toBe("Customer service calls");
  });
});
