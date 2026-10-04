import type { Prediction } from './types'

const API_BASE = import.meta.env.VITE_API_BASE ?? ''

export class PredictError extends Error {}

export async function predict(text: string, signal?: AbortSignal): Promise<Prediction> {
  let res: Response
  try {
    res = await fetch(`${API_BASE}/api/predict`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text }),
      signal,
    })
  } catch (err) {
    if ((err as Error).name === 'AbortError') throw err
    throw new PredictError('The model server did not answer. Check that the back end is running, then try again.')
  }
  if (res.status === 429) throw new PredictError('Another post is being checked. Try again in a moment.')
  if (res.status === 422) throw new PredictError('That post has no text to check. Type a few words first.')
  if (!res.ok) throw new PredictError(`The model server returned an error (${res.status}). Try again in a moment.`)
  const body = await res.json()
  if (!isPrediction(body)) throw new PredictError('The model server sent an answer this page cannot read. Try again.')
  return body
}

// Never render a malformed answer as a verdict.
function isPrediction(v: Prediction): v is Prediction {
  const probsOk = (p: number[]) =>
    Array.isArray(p) && p.length === 3 && p.every((x) => Number.isFinite(x) && x >= 0 && x <= 1) &&
    Math.abs(p[0] + p[1] + p[2] - 1) < 1e-3
  return (
    !!v && typeof v.text === 'string' && [0, 1, 2].includes(v.label) &&
    Array.isArray(v.models) && v.models.length > 0 &&
    v.models.every((m) => [0, 1, 2].includes(m.label) && probsOk(m.probs)) &&
    v.models.some((m) => m.id === v.primary) &&
    Array.isArray(v.tokens) &&
    v.tokens.every((t) => Number.isFinite(t.weight) && t.start >= 0 && t.end <= v.text.length && t.start < t.end)
  )
}
