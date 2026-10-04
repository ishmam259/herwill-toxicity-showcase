// Mirrors backend/app/main.py PredictOut (the Phase 0 contract in TASKS.md).

export type Label = 0 | 1 | 2
export type Script = 'bangla' | 'latin' | 'mixed' | 'none'

export interface ModelVote {
  id: string
  name: string
  label: Label
  probs: [number, number, number]
  mode: 'local-trained' | 'illustrative'
  has_features: boolean
}

export interface Token {
  text: string
  start: number
  end: number
  weight: number
}

export interface Prediction {
  text: string
  script: Script
  script_name: string
  primary: string
  label: Label
  label_name: string
  models: ModelVote[]
  tokens: Token[]
  explanation: string
  explanation_truncated: boolean
  latency_ms: number
  warning: string | null
}

export interface Example {
  id: string
  language: string
  label: number
  text: string
}

export const CLASSES: { label: Label; key: 'explicit' | 'subtle' | 'neutral'; verdict: string; short: string }[] = [
  { label: 0, key: 'explicit', verdict: 'Explicitly toxic', short: 'Explicit' },
  { label: 1, key: 'subtle', verdict: 'Subtly toxic', short: 'Subtle' },
  { label: 2, key: 'neutral', verdict: 'Not toxic', short: 'Neutral' },
]

export const SCRIPT_NAMES: Record<Script, string> = {
  bangla: 'Bangla script',
  latin: 'Latin script (English or Banglish)',
  mixed: 'Mixed Bangla and Latin script',
  none: 'No letters found',
}
