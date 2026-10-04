export const labels = ["Explicit", "Subtle", "Neutral"] as const;
export const classColors = ["#ae3d45", "#9c6b17", "#287767"];
export type Vote = {
  id: string;
  name: string;
  label: number;
  probs: number[];
  mode: string;
  has_features: boolean;
};
export type Prediction = {
  script: { id: string; name: string; note: string };
  models: Vote[];
  tokens: { text: string; weight: number }[];
  explanation: string;
  explanation_model: string;
  explanation_truncated: boolean;
  latency_ms: number;
  warning: string | null;
};
export type Health = {
  status: string;
  models: { id: string; name: string; mode: string }[];
  private_cases: boolean;
  transformer_available: boolean;
  stores_user_text: boolean;
};
export type Summary = {
  macro_f1: number;
  per_class_f1: number[];
  confusion: number[][];
  support: number[];
  rows: number;
};
export type Model = Summary & {
  id: string;
  name: string;
  family: string;
  specialist: boolean;
  by_script: Record<string, Summary>;
  decision_rule: string;
};
export type Results = {
  models: Model[];
  rows: number;
  warning: string;
  source: string;
  script_definition: Record<string, string>;
};
export type CalibrationModel = {
  bins: {
    lower: number;
    upper: number;
    count: number;
    confidence: number | null;
    accuracy: number | null;
  }[];
  ece: number;
  brier: number;
  macro_f1_grid: number[][];
  rows: number;
};
export type CalibrationData = {
  offsets: number[];
  models: Record<string, CalibrationModel>;
};
export type Example = {
  id: string;
  language: string;
  label: number;
  text: string;
};
export type HardCase = {
  id: number;
  category: string;
  text: string;
  label: number;
  script: string;
  votes: { id: string; label: number | null }[];
};
export const score = (value: number) => value.toFixed(3);
export const pct = (value: number) => `${(value * 100).toFixed(1)}%`;
export async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(path, init);
  if (!response.ok) {
    if (response.status === 429)
      throw new Error("Another prediction is running. Try again in a moment.");
    if (response.status === 403)
      throw new Error(
        "Private cases are disabled. Enable local mode to view competition posts.",
      );
    throw new Error(
      `The service could not complete the request (${response.status}). Please try again.`,
    );
  }
  return response.json() as Promise<T>;
}
export function validPrediction(value: Prediction) {
  return (
    value &&
    typeof value.script?.name === "string" &&
    Array.isArray(value.models) &&
    value.models.length > 0 &&
    value.models.every(
      (m) =>
        Number.isInteger(m.label) &&
        m.label >= 0 &&
        m.label <= 2 &&
        Array.isArray(m.probs) &&
        m.probs.length === 3 &&
        m.probs.every((p) => Number.isFinite(p) && p >= 0 && p <= 1) &&
        Math.abs(m.probs.reduce((a, b) => a + b, 0) - 1) < 1e-4,
    ) &&
    Array.isArray(value.tokens) &&
    value.tokens.every(
      (t) => typeof t.text === "string" && Number.isFinite(t.weight),
    )
  );
}
