import { useRef, useState } from 'react'
import type { FormEvent, KeyboardEvent, ReactNode } from 'react'
import { predict, PredictError } from './api'
import { CLASSES, SCRIPT_NAMES } from './types'
import type { Example, Prediction, Token } from './types'
import './LiveDemo.css'

const MAX_CHARS = 2000
const LANGUAGES = ['Bangla', 'Banglish', 'English']
const pct = (p: number) => `${Math.round(p * 100)}%`
// "girl," -> "girl", so the list of strongest words reads cleanly
const bare = (w: string) => w.replace(/^[\p{P}\p{S}]+|[\p{P}\p{S}]+$/gu, '') || w

type State =
  | { kind: 'idle' }
  | { kind: 'loading' }
  | { kind: 'error'; message: string }
  | { kind: 'done'; result: Prediction }

// Mounted by the app shell on "/"; the shell supplies the page title (h1).
export default function LiveDemo({ examples }: { examples: Example[] }) {
  const [text, setText] = useState('')
  const [language, setLanguage] = useState('Bangla')
  const [state, setState] = useState<State>({ kind: 'idle' })
  const abortRef = useRef<AbortController | null>(null)

  async function run(post: string) {
    if (!post.trim()) return
    abortRef.current?.abort()
    const ctrl = new AbortController()
    abortRef.current = ctrl
    setState({ kind: 'loading' })
    try {
      const result = await predict(post, ctrl.signal)
      if (!ctrl.signal.aborted) setState({ kind: 'done', result })
    } catch (err) {
      if ((err as Error).name === 'AbortError') return
      setState({ kind: 'error', message: err instanceof PredictError ? err.message : 'Something went wrong. Try again.' })
    }
  }

  function edit(next: string) {
    // A verdict for different text would be misleading: drop it, and any answer still on its way.
    abortRef.current?.abort()
    setText(next)
    if (state.kind !== 'idle') setState({ kind: 'idle' })
  }

  function onSubmit(e: FormEvent) {
    e.preventDefault()
    run(text)
  }

  function onKeyDown(e: KeyboardEvent<HTMLTextAreaElement>) {
    if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) {
      e.preventDefault()
      run(text)
    }
  }

  function tryExample(example: string) {
    setText(example)
    run(example)
  }

  const shown = examples.filter((e) => e.language === language)

  return (
    <section className="lens" aria-label="Live demo">
      <form className="lens-composer" onSubmit={onSubmit}>
        <label htmlFor="lens-post">Post to check</label>
        <textarea
          id="lens-post"
          value={text}
          maxLength={MAX_CHARS}
          rows={4}
          placeholder="Type or paste a post in Bangla, Banglish or English…"
          onChange={(e) => edit(e.target.value)}
          onKeyDown={onKeyDown}
        />
        <div className="lens-composer-row">
          <button type="submit" disabled={!text.trim() || state.kind === 'loading'}>
            {state.kind === 'loading' ? 'Checking…' : 'Check post'}
          </button>
          <span className="lens-hint">
            <span>Ctrl + Enter also checks</span>
            <span>{text.length} / {MAX_CHARS}</span>
          </span>
        </div>
      </form>

      <div className="lens-examples">
        <div className="lens-examples-head">
          <p id="lens-examples-label">Or try a hand-written example</p>
          <div className="lens-langs" role="group" aria-label="Example language">
            {LANGUAGES.map((l) => (
              <button key={l} type="button" aria-pressed={language === l} onClick={() => setLanguage(l)}>
                {l}
              </button>
            ))}
          </div>
        </div>
        <ul aria-labelledby="lens-examples-label">
          {shown.map((ex) => (
            <li key={ex.id}>
              <button type="button" className="lens-example" onClick={() => tryExample(ex.text)}>
                {ex.text}
              </button>
            </li>
          ))}
        </ul>
      </div>

      <div aria-live="polite">
        {state.kind === 'error' && <p className="lens-error" role="alert">{state.message}</p>}
        {state.kind === 'done' && <Result result={state.result} />}
      </div>
    </section>
  )
}

function Result({ result }: { result: Prediction }) {
  const cls = CLASSES[result.label]
  const primary = result.models.find((m) => m.id === result.primary) ?? result.models[0]
  const pushers = [...result.tokens].filter((t) => t.weight > 0).sort((a, b) => b.weight - a.weight).slice(0, 3)

  return (
    <article className={`lens-result is-${cls.key}`}>
      <div className="lens-verdict-row">
        <h2 className="lens-verdict">{cls.verdict}</h2>
        <p className="lens-verdict-meta">
          <span>{pct(primary.probs[result.label])} confident</span>
          <span>{SCRIPT_NAMES[result.script]}</span>
        </p>
      </div>
      {result.warning && <p className="lens-notice">{result.warning}</p>}
      {!primary.has_features && (
        <p className="lens-notice">None of these words were seen in training, so this verdict has little to go on.</p>
      )}

      <Highlighted text={result.text} tokens={result.tokens} />
      <p className="lens-key">
        {pushers.length > 0 ? (
          <>
            Thicker underline, stronger push toward “{cls.verdict.toLowerCase()}”. Strongest:{' '}
            {pushers.map((t, i) => (
              <span key={t.start}>
                <q>{bare(t.text)}</q>
                {i < pushers.length - 1 ? ', ' : '.'}
              </span>
            ))}
            {result.explanation_truncated && ' Only the first 80 words were tested.'}
          </>
        ) : (
          'No single word moved the verdict much; the model read the post as a whole.'
        )}
      </p>

      <h3>How sure, per class</h3>
      <dl className="lens-bars">
        {CLASSES.map((c) => (
          <div key={c.key} className={`lens-bar-row is-${c.key}`}>
            <dt>{c.short}</dt>
            <dd>
              <span className="lens-bar" style={{ width: pct(primary.probs[c.label]) }} />
              <span className="lens-bar-value">{pct(primary.probs[c.label])}</span>
            </dd>
          </div>
        ))}
      </dl>

      <h3>What each model says</h3>
      <div className="lens-table-wrap" tabIndex={0} role="region" aria-label="Model votes">
        <table className="lens-votes">
          <thead>
            <tr>
              <th scope="col">Model</th>
              <th scope="col">Verdict</th>
              {CLASSES.map((c) => (
                <th scope="col" key={c.key} className="num">{c.short}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {result.models.map((m) => (
              <tr key={m.id}>
                <th scope="row">
                  {m.name}
                  {m.mode === 'illustrative' && <small> (demo model)</small>}
                </th>
                <td className={`is-${CLASSES[m.label].key}`}>{CLASSES[m.label].short}</td>
                {m.probs.map((p, i) => (
                  <td key={i} className="num">{pct(p)}</td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <p className="lens-foot">
        <span>{result.explanation}</span>
        <span>Answered in {result.latency_ms} ms. The text you enter is not stored.</span>
      </p>
    </article>
  )
}

function Highlighted({ text, tokens }: { text: string; tokens: Token[] }) {
  const max = Math.max(1e-6, ...tokens.map((t) => Math.abs(t.weight)))
  const parts: ReactNode[] = []
  let at = 0
  for (const t of tokens) {
    if (t.start > at) parts.push(text.slice(at, t.start))
    const strength = t.weight / max // -1..1
    const pushes = strength > 0.08
    parts.push(
      <span
        key={t.start}
        className={pushes ? 'lens-w is-push' : 'lens-w'}
        style={pushes ? { textDecorationThickness: `${1 + strength * 5}px` } : undefined}
      >
        {text.slice(t.start, t.end)}
      </span>,
    )
    at = t.end
  }
  if (at < text.length) parts.push(text.slice(at))
  return <blockquote className="lens-post">{parts}</blockquote>
}
