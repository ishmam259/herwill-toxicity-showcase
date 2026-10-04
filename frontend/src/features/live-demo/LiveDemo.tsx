import { useRef, useState } from 'react'
import type { FormEvent, KeyboardEvent, ReactNode } from 'react'
import { predict, PredictError } from './api'
import { EXAMPLES } from './examples'
import { CLASSES, SCRIPT_NAMES } from './types'
import type { Prediction, Token } from './types'
import './LiveDemo.css'

const MAX_CHARS = 2000
const pct = (p: number) => `${Math.round(p * 100)}%`
// "girl," -> "girl", so the list of strongest words reads cleanly
const bare = (w: string) => w.replace(/^[\p{P}\p{S}]+|[\p{P}\p{S}]+$/gu, '') || w

type State =
  | { kind: 'idle' }
  | { kind: 'loading' }
  | { kind: 'error'; message: string }
  | { kind: 'done'; result: Prediction }

export default function LiveDemo() {
  const [text, setText] = useState('')
  const [state, setState] = useState<State>({ kind: 'idle' })
  const abortRef = useRef<AbortController | null>(null)

  async function run(post: string) {
    if (!post.trim()) return
    abortRef.current?.abort()
    const ctrl = new AbortController()
    abortRef.current = ctrl
    setState({ kind: 'loading' })
    try {
      setState({ kind: 'done', result: await predict(post, ctrl.signal) })
    } catch (err) {
      if ((err as Error).name === 'AbortError') return
      setState({ kind: 'error', message: err instanceof PredictError ? err.message : 'Something went wrong. Try again.' })
    }
  }

  function onSubmit(e: FormEvent) {
    e.preventDefault()
    run(text)
  }

  function onKeyDown(e: KeyboardEvent<HTMLTextAreaElement>) {
    if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) run(text)
  }

  function tryExample(example: string) {
    setText(example)
    run(example)
  }

  return (
    <section className="lens" aria-labelledby="lens-title">
      <header className="lens-head">
        <h1 id="lens-title">Read a post the way our model does</h1>
        <p className="lens-sub">
          Paste a comment in Bangla, Banglish or English. The model says whether it is explicitly toxic, subtly
          toxic or not toxic, and which words moved it.
        </p>
      </header>

      <form className="composer" onSubmit={onSubmit}>
        <label htmlFor="post" className="sr-only">Post to check</label>
        <textarea
          id="post"
          value={text}
          maxLength={MAX_CHARS}
          rows={4}
          placeholder="Type or paste a post…"
          onChange={(e) => setText(e.target.value)}
          onKeyDown={onKeyDown}
        />
        <div className="composer-row">
          <button type="submit" disabled={!text.trim() || state.kind === 'loading'}>
            {state.kind === 'loading' ? 'Checking…' : 'Check post'}
          </button>
          <span className="hint">
            <span>Ctrl + Enter also checks</span>
            <span>{text.length} / {MAX_CHARS}</span>
          </span>
        </div>
      </form>

      <div className="examples">
        <p id="examples-label">Or try one of these</p>
        <ul aria-labelledby="examples-label">
          {EXAMPLES.map((ex) => (
            <li key={ex.text}>
              <button type="button" className="example" onClick={() => tryExample(ex.text)}>
                <span className="example-script">{ex.script}</span>
                <span className="example-text">{ex.text}</span>
              </button>
            </li>
          ))}
        </ul>
      </div>

      <div aria-live="polite">
        {state.kind === 'error' && <p className="error" role="alert">{state.message}</p>}
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
    <article className={`result is-${cls.key}`}>
      <div className="verdict-row">
        <h2 className="verdict">{cls.verdict}</h2>
        <p className="verdict-meta">
          <span>{pct(primary.probs[result.label])} confident</span>
          <span>{SCRIPT_NAMES[result.script]}</span>
        </p>
      </div>

      <Highlighted text={result.text} tokens={result.tokens} />
      <p className="reading-key">
        {pushers.length > 0 ? (
          <>
            Thicker underline, stronger push toward “{cls.verdict.toLowerCase()}”. Strongest:{' '}
            {pushers.map((t, i) => (
              <span key={t.start}>
                <q>{bare(t.text)}</q>
                {i < pushers.length - 1 ? ', ' : '.'}
              </span>
            ))}
          </>
        ) : (
          'No single word moved the verdict much; the model read the post as a whole.'
        )}
      </p>

      <h3>How sure, per class</h3>
      <dl className="bars">
        {CLASSES.map((c) => (
          <div key={c.key} className={`bar-row is-${c.key}`}>
            <dt>{c.short}</dt>
            <dd>
              <span className="bar" style={{ width: pct(primary.probs[c.label]) }} />
              <span className="bar-value">{pct(primary.probs[c.label])}</span>
            </dd>
          </div>
        ))}
      </dl>

      <h3>What each model says</h3>
      <div className="table-wrap">
        <table className="votes">
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
                <th scope="row">{m.name}</th>
                <td className={`is-${CLASSES[m.label].key}`}>{CLASSES[m.label].short}</td>
                {m.probs.map((p, i) => (
                  <td key={i} className="num">{pct(p)}</td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
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
    const style =
      strength > 0.08
        ? { textDecorationThickness: `${1 + strength * 5}px` }
        : undefined
    parts.push(
      <span key={t.start} className={strength > 0.08 ? 'w push' : 'w'} style={style}>
        {text.slice(t.start, t.end)}
      </span>,
    )
    at = t.end
  }
  if (at < text.length) parts.push(text.slice(at))
  return <blockquote className="post">{parts}</blockquote>
}
