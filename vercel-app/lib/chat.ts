export type Profile = Record<string, number | null>;

export const FIELDS = [
  "Account length", "International plan", "Voice mail plan", "Number vmail messages",
  "Total day minutes", "Total day calls", "Total eve minutes", "Total eve calls",
  "Total night minutes", "Total night calls", "Total intl minutes", "Total intl calls",
  "Customer service calls",
];

export const LABELS: Record<string, string> = {
  "Account length": "Time with provider",
  "International plan": "International plan",
  "Voice mail plan": "Voicemail plan",
  "Number vmail messages": "Voicemail messages",
  "Total day minutes": "Daytime phone use",
  "Total day calls": "Daytime calls",
  "Total eve minutes": "Evening phone use",
  "Total eve calls": "Evening calls",
  "Total night minutes": "Nighttime phone use",
  "Total night calls": "Nighttime calls",
  "Total intl minutes": "International phone use",
  "Total intl calls": "International calls",
  "Customer service calls": "Calls to customer support",
};

const WORDS: Record<string, string> = {
  zero: "0", one: "1", two: "2", three: "3", four: "4", five: "5", six: "6", seven: "7",
  eight: "8", nine: "9", ten: "10", eleven: "11", twelve: "12", thirteen: "13", fourteen: "14",
  fifteen: "15", sixteen: "16", seventeen: "17", eighteen: "18", nineteen: "19", twenty: "20", dozen: "12", couple: "2",
};
const INTEGER = new Set(["Account length", "Number vmail messages", "Total day calls", "Total eve calls", "Total night calls", "Total intl calls", "Customer service calls"]);

export function parseCustomerText(input: string, current: Profile, ranges: Record<string, { min: number; max: number }>) {
  const text = input.toLowerCase().replace(/\b(zero|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty|dozen|couple)\b/g, (w) => WORDS[w]);
  const profile = { ...current };
  const changed = new Set<string>();
  const set = (field: string, raw: number) => {
    const range = ranges[field];
    if (!range || !Number.isFinite(raw) || raw < range.min || raw > range.max) return;
    profile[field] = INTEGER.has(field) ? Math.round(raw) : raw;
    changed.add(field);
  };

  for (const [field, pattern] of [
    ["International plan", "international(?: calling)?(?: plan)?"],
    ["Voice mail plan", "voice[- ]?mail plan"],
  ]) {
    const re = new RegExp(`(?:no|without|doesn['’]?t have|not on)\\s+(?:an?\\s+)?${pattern}|${pattern}\\s*(?:is|:)?\\s*(?:no|false)`, "i");
    const yes = new RegExp(`(?:has|have|with|yes|on)\\s+(?:an?\\s+)?${pattern}|${pattern}\\s*(?:is|:)?\\s*(?:yes|true)`, "i");
    if (re.test(text)) set(field, 0);
    else if (yes.test(text)) set(field, 1);
  }

  const tenureContext = /account|customer|tenure|with (?:the )?(?:provider|company)|been (?:a |an )?(?:customer|subscriber)/;
  const tenure = text.match(/(\d+(?:\.\d+)?)\s*(days?|weeks?|months?|mos?|years?|yrs?)\b/);
  if (tenure && tenureContext.test(text)) {
    const n = Number(tenure[1]);
    const unit = tenure[2];
    set("Account length", n * (unit.startsWith("week") ? 7 : /^(month|mo)/.test(unit) ? 30 : /^(year|yr)/.test(unit) ? 365 : 1));
  }

  const nums = [...text.matchAll(/\b\d+(?:,\d{3})*(?:\.\d+)?\b/g)];
  for (const match of nums) {
    const value = Number(match[0].replaceAll(",", ""));
    const start = match.index ?? 0;
    const sentenceStart = Math.max(text.lastIndexOf(".", start), text.lastIndexOf(",", start), text.lastIndexOf(";", start), text.lastIndexOf("!", start), text.lastIndexOf("?", start)) + 1;
    const sentenceEndCandidates = [".", ",", ";", "!", "?"].map((mark) => text.indexOf(mark, start + match[0].length)).filter((index) => index >= 0);
    const sentenceEnd = sentenceEndCandidates.length ? Math.min(...sentenceEndCandidates) : text.length;
    const segment = text.slice(sentenceStart, sentenceEnd);
    const relativeStart = start - sentenceStart;
    const conjunctions = [...segment.matchAll(/\band\b/gi)].map((item) => ({ start: item.index ?? 0, end: (item.index ?? 0) + item[0].length }));
    const beforeBoundary = conjunctions.filter((item) => item.end <= relativeStart).at(-1);
    const afterBoundary = conjunctions.find((item) => item.start >= relativeStart + match[0].length);
    const context = segment.slice(beforeBoundary?.end ?? 0, afterBoundary?.start ?? segment.length);
    if (/voicemail|voice mail|voice-mail/.test(context) && /message|msg/.test(context)) { set("Number vmail messages", value); continue; }
    if (/customer\s*(?:service|support)|support team|help desk/.test(context) && /call|contact|time|reach/.test(context)) { set("Customer service calls", value); continue; }
    const periodMatch = context.match(/international|intl|evening|\beve\b|night|overnight|daytime|during the day|\bday\b/);
    const unitMatch = context.match(/minutes?|mins?|calls?/);
    if (!periodMatch || !unitMatch) continue;
    const period = /intl|international/.test(periodMatch[0]) ? "intl" : /evening|eve/.test(periodMatch[0]) ? "eve" : /night|overnight/.test(periodMatch[0]) ? "night" : "day";
    const unit = /^call/.test(unitMatch[0]) ? "calls" : "minutes";
    set(`Total ${period} ${unit}`, value);
  }

  const unknown = [...text.matchAll(/not sure|don't know|dont know|unsure|unknown|skip|\bidk\b|\bn\/?a\b/g)];
  for (const u of unknown) {
    const context = text.slice(Math.max(0, (u.index ?? 0) - 35), Math.min(text.length, (u.index ?? 0) + u[0].length + 30));
    let field: string | undefined;
    if (/international.{0,25}(?:plan|calling)|(?:plan|calling).{0,25}international/.test(context)) field = "International plan";
    else if (/voicemail|voice mail/.test(context)) field = /message|msg/.test(context) ? "Number vmail messages" : "Voice mail plan";
    else if (/support|customer service|help desk/.test(context)) field = "Customer service calls";
    else if (/tenure|account|provider|how long/.test(context)) field = "Account length";
    else {
      const period = /international|intl/.test(context) ? "intl" : /evening|\beve\b/.test(context) ? "eve" : /night/.test(context) ? "night" : /daytime|during the day/.test(context) ? "day" : undefined;
      if (period) field = `Total ${period} ${/call/.test(context) ? "calls" : "minutes"}`;
    }
    if (field && FIELDS.includes(field)) { profile[field] = null; changed.add(field); }
  }

  return { profile, changed: [...changed] };
}

export function parseSingleAnswer(field: string, answer: string, ranges: Record<string, { min: number; max: number }>): number | null | undefined {
  const numberWords = { zero: "0", one: "1", two: "2", three: "3", four: "4", five: "5", six: "6", seven: "7", eight: "8", nine: "9", ten: "10", eleven: "11", twelve: "12", thirteen: "13", fourteen: "14", fifteen: "15", sixteen: "16", seventeen: "17", eighteen: "18", nineteen: "19", twenty: "20", dozen: "12", couple: "2" } as Record<string, string>;
  const text = answer.trim().toLowerCase().replace(/\b(zero|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty|dozen|couple)\b/g, (word) => numberWords[word]);
  if (/^(i'?m )?(not sure|unsure|unknown|skip|idk|n\/?a|i don't know|dont know|no idea)[.! ]*$/.test(text)) return null;
  if (field === "International plan" || field === "Voice mail plan") {
    if (/\b(no|nope|nah|not|without|don't|dont)\b/.test(text)) return 0;
    if (/\b(yes|yeah|yep|have|has|included|subscribed)\b/.test(text)) return 1;
  }
  const duration = field === "Account length" ? text.match(/(\d+(?:\.\d+)?)\s*(years?|yrs?|months?|mos?|days?)\b/) : null;
  let value: number | undefined;
  if (duration) value = Number(duration[1]) * (/^(year|yr)/.test(duration[2]) ? 365 : /^(month|mo)/.test(duration[2]) ? 30 : 1);
  else {
    const range = text.match(/(\d+(?:\.\d+)?)\s*(?:-|to)\s*(\d+(?:\.\d+)?)/);
    const number = text.match(/\d+(?:\.\d+)?/);
    if (range) value = (Number(range[1]) + Number(range[2])) / 2;
    else if (number) value = Number(number[0]);
  }
  if (value === undefined || value < ranges[field].min || value > ranges[field].max) return undefined;
  return INTEGER.has(field) ? Math.round(value) : value;
}
