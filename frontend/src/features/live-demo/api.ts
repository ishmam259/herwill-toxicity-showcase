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
  if (res.status === 422) throw new PredictError('That post has no text to check. Type a few words first.')
  if (!res.ok) throw new PredictError(`The model server returned an error (${res.status}). Try again in a moment.`)
  return res.json()
}
