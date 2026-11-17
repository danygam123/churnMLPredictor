// Playwright E2E — the 13 production-release journeys for Churn Compass.
// The browser talks to the real built Next.js app with /api/* proxied to the
// real Python endpoint handlers, so these journeys exercise the actual
// production code paths (client shell + serverless engine).

import { expect, test } from "@playwright/test";

const SAMPLE_NOTE = "Sample data";
const COMPLETE_MARK = "Start a new customer";

function nav(page: import("@playwright/test").Page, name: string) {
  return page.locator(".nav-button", { hasText: name });
}

async function loadSample(page: import("@playwright/test").Page) {
  await page.goto("/");
  await nav(page, "Portfolio").click();
  await page.getByRole("button", { name: /Load the built-in sample/ }).click();
  await expect(page.locator(".dataset-banner")).toContainText(SAMPLE_NOTE, { timeout: 60_000 });
}

// ---------------------------------------------------------------------------
// 1. Guided interview completes and produces a score card
// ---------------------------------------------------------------------------
test("01 chat interview gathers a profile and scores it", async ({ page }) => {
  await page.goto("/");
  const composer = page.getByPlaceholder("Describe the customer in your own words…");
  await composer.fill(
    "They have been with the provider for about 5 months, no international plan, voicemail plan yes, " +
      "200 daytime minutes, 100 daytime calls, 150 evening minutes, 80 evening calls, " +
      "150 night minutes, 80 night calls, 10 international minutes, 4 international calls, 3 customer support calls.",
  );
  await page.keyboard.press("Enter");
  // Whatever the parser could not capture is filled with disclosed typical values.
  while (!(await page.getByText(COMPLETE_MARK).isVisible().catch(() => false))) {
    const helpButton = page.getByRole("button", { name: "Need a hand?" });
    if (await helpButton.isVisible().catch(() => false)) {
      await helpButton.click();
      await page.getByRole("button", { name: "Use a typical value" }).click();
    } else {
      await composer.fill("not sure");
      await page.keyboard.press("Enter");
    }
    await expect(page.getByText(COMPLETE_MARK).or(page.locator(".help-pop")).first()).toBeVisible({ timeout: 15_000 });
  }
  await expect(page.locator(".score-card")).toBeVisible({ timeout: 30_000 });
  await expect(page.locator(".score-card")).toContainText(/Likely to churn|Less likely to churn/);
  await expect(page.locator(".score-card")).toContainText(/risk/i);
});

// ---------------------------------------------------------------------------
// 2. “Not sure” is a valid answer and the estimate discloses imputation
// ---------------------------------------------------------------------------
test("02 unknown details are marked estimated and disclosed", async ({ page }) => {
  await page.goto("/");
  const composer = page.getByPlaceholder("Describe the customer in your own words…");
  await composer.fill("They have called customer support 5 times.");
  await page.keyboard.press("Enter");
  for (let i = 0; i < 13; i++) {
    if (await page.getByText(COMPLETE_MARK).isVisible().catch(() => false)) break;
    await page.getByRole("button", { name: "Need a hand?" }).click();
    await page.getByRole("button", { name: "Use a typical value" }).click();
  }
  await expect(page.getByText(COMPLETE_MARK)).toBeVisible();
  await expect(page.locator(".score-card")).toBeVisible({ timeout: 30_000 });
  await expect(page.locator(".score-card")).toContainText(/used typical values/i);
  await expect(page.locator(".score-card")).toContainText("Estimated details:");
});

// ---------------------------------------------------------------------------
// 3. Quick form produces a prediction
// ---------------------------------------------------------------------------
test("03 quick form predicts with an unknown detail marked", async ({ page }) => {
  await page.goto("/");
  await page.getByRole("button", { name: /Quick form/ }).click();
  await page.locator(".unknown-check").last().click(); // mark "Customer service calls" as unknown
  await page.getByRole("button", { name: "Estimate customer risk" }).click();
  await expect(page.locator(".score-card")).toBeVisible({ timeout: 30_000 });
  await expect(page.locator(".score-card")).toContainText(/Estimated details:/);
});

// ---------------------------------------------------------------------------
// 4. SHAP explanation computes on demand
// ---------------------------------------------------------------------------
test("04 SHAP explanation loads from the engine", async ({ page }) => {
  await page.goto("/");
  await page.getByRole("button", { name: /Quick form/ }).click();
  await page.getByRole("button", { name: "Estimate customer risk" }).click();
  await expect(page.locator(".score-card")).toBeVisible({ timeout: 30_000 });
  await page.getByRole("button", { name: /Compute the model explanation \(SHAP\)/ }).click();
  await expect(page.getByText(/Signed SHAP contributions/i)).toBeVisible({ timeout: 60_000 });
});

// ---------------------------------------------------------------------------
// 5. Built-in sample dataset loads and reports headline metrics
// ---------------------------------------------------------------------------
test("05 sample dataset loads with headline metrics", async ({ page }) => {
  await loadSample(page);
  await expect(page.locator(".metric-card", { hasText: "Customers analyzed" })).toContainText("80");
  await expect(page.getByText("Risk-band distribution")).toBeVisible();
  await expect(page.getByText("Customer patterns")).toBeVisible();
});

// ---------------------------------------------------------------------------
// 6. A rejected upload never replaces the working dataset
// ---------------------------------------------------------------------------
test("06 invalid upload is rejected and the working dataset survives", async ({ page }) => {
  await loadSample(page);
  await nav(page, "Project info").click();
  const badCsv = Buffer.from("Customer ID,Wrong Column\nC-1,1\n", "utf-8");
  await page.locator("#settings input[type=file]").setInputFiles({
    name: "bad-upload.csv",
    mimeType: "text/csv",
    buffer: badCsv,
  });
  await expect(page.getByText(/missing required model input columns/i)).toBeVisible({ timeout: 30_000 });
  await expect(page.locator(".side-dataset-card")).toContainText(SAMPLE_NOTE);
});

// ---------------------------------------------------------------------------
// 7. Customer table: search, filter, and CSV export of the filtered view
// ---------------------------------------------------------------------------
test("07 customers view filters and exports the selection", async ({ page }) => {
  await loadSample(page);
  await nav(page, "Customers").click();
  await expect(page.locator(".data-table tbody tr").first()).toBeVisible();
  await expect(page.getByText(/of 80 customers shown/)).toBeVisible();

  await page.locator("#customer-search").fill("C-0666");
  await expect(page.getByText(/1 of 80 customers shown/)).toBeVisible();
  await page.locator("#customer-search").fill("");
  await page.getByRole("button", { name: "High", exact: true }).click();
  await expect(page.getByText(/filters active/)).toBeVisible();

  const [download] = await Promise.all([
    page.waitForEvent("download"),
    page.getByRole("button", { name: /Export these customers/ }).click(),
  ]);
  expect(download.suggestedFilename()).toBe("churn_customer_predictions.csv");
});

// ---------------------------------------------------------------------------
// 8. Customer detail with SHAP, suggestions, and the Simulation Lab hand-off
// ---------------------------------------------------------------------------
test("08 customer detail explains and hands off to the Simulation Lab", async ({ page }) => {
  await loadSample(page);
  await nav(page, "Customers").click();
  await page.locator(".data-table tbody tr").first().click();
  await expect(page.locator(".customer-detail")).toBeVisible();
  await expect(page.locator(".customer-detail")).toContainText(/SHAP/i, { timeout: 60_000 });
  await page.getByRole("button", { name: /Use in Simulation Lab/ }).click();
  await expect(page.getByText(/Baseline · Customer/)).toBeVisible();
});

// ---------------------------------------------------------------------------
// 9. Retention strategies render rule-based suggestions
// ---------------------------------------------------------------------------
test("09 retention view shows rule groups for the sample data", async ({ page }) => {
  await loadSample(page);
  await nav(page, "Retention Strategies").click();
  await expect(page.getByText(/review suggestions/i).first()).toBeVisible({ timeout: 60_000 });
  await expect(page.locator(".retention-layout").getByText(/rule match/i)).toBeVisible();
  await expect(page.locator(".retention-layout").getByText(/not proven interventions/i)).toBeVisible();
});

// ---------------------------------------------------------------------------
// 10. Simulation Lab compares a baseline with a scenario
// ---------------------------------------------------------------------------
test("10 simulation compares baseline with a what-if scenario", async ({ page }) => {
  await loadSample(page);
  await nav(page, "Simulation Lab").click();
  await page.locator("#sim-customer").selectOption({ index: 1 });
  await expect(page.getByText(/Baseline · Customer/)).toBeVisible();
  await page.locator('[id="sim-value-Scenario A"]').fill("300");
  await page.getByRole("button", { name: "Run comparison" }).click();
  await expect(page.locator(".data-table th", { hasText: "Churn probability" })).toBeVisible({ timeout: 60_000 });
  await expect(page.locator(".data-table tbody tr").first()).toContainText("Baseline");
  await expect(page.locator(".data-table tbody tr", { hasText: "Scenario A" })).toBeVisible();
});

// ---------------------------------------------------------------------------
// 11. AI Analyst answers with computed facts (facts-only mode without a key)
// ---------------------------------------------------------------------------
test("11 AI analyst returns grounded computed facts", async ({ page }) => {
  await loadSample(page);
  await nav(page, "AI Analyst").click();
  await page.locator(".quick-chip").first().click();
  await expect(page.locator(".message").last()).toContainText(/ANSWERED · GROUNDED|COMPUTED FACTS/, { timeout: 60_000 });
  await expect(page.getByText("Computed fact").first()).toBeVisible();
  await expect(page.locator(".snapshot")).toContainText("Restricted by design");
  await expect(page.locator(".snapshot")).toContainText(/Raw customer rows and customer identifiers are not sent/);
});

// ---------------------------------------------------------------------------
// 12. Model performance documents both operating points; 0.0773 is labeled
//     as an evaluated operating point, never as the production cutoff
// ---------------------------------------------------------------------------
test("12 model performance shows the V2 evaluation record", async ({ page }) => {
  await page.goto("/");
  await nav(page, "Model performance").click();
  await expect(page.getByText("V2 engine · champion model evaluation")).toBeVisible();
  await expect(page.getByText("Default 50% cutoff (production)")).toBeVisible();
  await expect(page.getByText("Phase 1 cost-optimal (evaluated)")).toBeVisible();
  await expect(page.getByText("Probability calibration")).toBeVisible();
  await expect(page.getByText(/Training-sample global SHAP results/i)).toBeVisible();
  await expect(page.locator(".page-title")).toContainText(/How the model performs/);
});

// ---------------------------------------------------------------------------
// 13. Settings: risk bands are session-adjustable; clearing resets everything
// ---------------------------------------------------------------------------
test("13 settings adjust bands and clear the session", async ({ page }) => {
  await loadSample(page);
  await nav(page, "Project info").click();
  await expect(page.locator("#settings")).toBeVisible();
  await expect(page.getByText(/facts-only mode|language-model key is configured/i)).toBeVisible();

  await page.locator("#band-medium").fill("0.2");
  await page.locator("#band-high").fill("0.6");
  await page.getByRole("button", { name: "Apply bands" }).click();
  await expect(page.locator(".side-dataset-card")).toContainText("Med ≥ 0.20");
  await expect(page.locator(".side-dataset-card")).toContainText("High ≥ 0.60");

  await page.getByRole("button", { name: "Reset to defaults" }).click();
  await expect(page.locator(".side-dataset-card")).toContainText("Med ≥ 0.08");

  await page.locator(".side-clear").click();
  await expect(page.getByText("No dataset is loaded.")).toBeVisible();
  await expect(page.locator(".side-dataset-card")).toHaveCount(0);
});
