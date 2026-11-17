// Unit tests for the guided-interview language parser (lib/chat.ts).

import { describe, expect, it } from "vitest";
import { FIELDS, LABELS, parseCustomerText, parseSingleAnswer } from "./chat";

const RANGES = Object.fromEntries(
  FIELDS.map((field) => [field, { min: field.includes("calls") || field.includes("messages") ? 0 : 0, max: field === "Account length" ? 243 : 9999 }]),
) as Record<string, { min: number; max: number }>;

describe("parseCustomerText", () => {
  it("extracts plans, tenure, usage, and support calls from one message", () => {
    const result = parseCustomerText(
      "They have been with the provider for about 6 months, no international plan, voicemail plan yes, " +
        "200 daytime minutes, 100 daytime calls, 3 customer support calls.",
      {},
      RANGES,
    );
    const profile = result.profile;
    expect(profile["Account length"]).toBe(180);
    expect(profile["International plan"]).toBe(0);
    expect(profile["Voice mail plan"]).toBe(1);
    expect(profile["Total day minutes"]).toBe(200);
    expect(profile["Total day calls"]).toBe(100);
    expect(profile["Customer service calls"]).toBe(3);
    expect(result.changed.length).toBeGreaterThanOrEqual(6);
  });

  it("understands number words", () => {
    const result = parseCustomerText("they make about ten international calls", {}, RANGES);
    expect(result.profile["Total intl calls"]).toBe(10);
  });

  it("marks mentioned unknowns as estimated (null)", () => {
    const result = parseCustomerText("not sure about their voicemail plan", {}, RANGES);
    expect(result.profile["Voice mail plan"]).toBeNull();
    expect(result.changed).toContain("Voice mail plan");
  });

  it("ignores values outside the accepted range", () => {
    const result = parseCustomerText("they use 50000 daytime minutes", {}, RANGES);
    expect("Total day minutes" in result.profile ? result.profile["Total day minutes"] : undefined).not.toBe(50000);
  });
});

describe("parseSingleAnswer", () => {
  it("maps yes/no wording for plan questions", () => {
    expect(parseSingleAnswer("International plan", "yes", RANGES)).toBe(1);
    expect(parseSingleAnswer("International plan", "no", RANGES)).toBe(0);
    expect(parseSingleAnswer("Voice mail plan", "they have it", RANGES)).toBe(1);
  });

  it("converts months to days for tenure and rejects out-of-range years", () => {
    expect(parseSingleAnswer("Account length", "about 6 months", RANGES)).toBe(180);
    expect(parseSingleAnswer("Account length", "2 years", RANGES)).toBeUndefined(); // 730 days exceeds the accepted 243
  });

  it("treats explicit uncertainty as estimated (null)", () => {
    expect(parseSingleAnswer("Customer service calls", "not sure", RANGES)).toBeNull();
  });

  it("rejects out-of-range numbers as undefined (not estimated)", () => {
    expect(parseSingleAnswer("Total day minutes", "99999", RANGES)).toBeUndefined();
  });
});

describe("labels", () => {
  it("provides a human label for every field", () => {
    for (const field of FIELDS) expect(LABELS[field]).toBeTruthy();
  });
});
