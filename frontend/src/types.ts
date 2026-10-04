export const labels = ["Explicit", "Subtle", "Neutral"] as const;
export const classColors = ["#ae3d45", "#9c6b17", "#287767"];
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
