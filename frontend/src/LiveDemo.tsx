import { useEffect, useRef, useState } from "react";
import {
  ArrowRight,
  ArrowUpRight,
  Check,
  ChevronDown,
  CircleHelp,
  CornerDownLeft,
  FlaskConical,
  Languages,
  LoaderCircle,
  RotateCcw,
  ScanText,
  ShieldCheck,
  Sparkles,
  Waves,
  X,
} from "lucide-react";
import { Badge, Card } from "./components";
import {
  labels,
  pct,
  request,
  validPrediction,
  type Example,
  type Health,
  type Prediction,
} from "./types";
export default function LiveDemo({ examples }: { examples: Example[] }) {
  const [text, setText] = useState(
    examples.find((e) => e.id === "en-neu-1")!.text,
  );
  const [language, setLanguage] = useState("English");
  const [exampleSet, setExampleSet] = useState(0);
  const [result, setResult] = useState<Prediction | null>(null);
  const [health, setHealth] = useState<Health | null>(null);
  const [offline, setOffline] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const controller = useRef<AbortController | null>(null);
  const editor = useRef<HTMLTextAreaElement | null>(null);
  useEffect(() => {
    const abort = new AbortController();
    request<Health>("/api/health", {
      signal: AbortSignal.any([abort.signal, AbortSignal.timeout(8000)]),
    })
      .then(setHealth)
      .catch(() => {
        if (!abort.signal.aborted) setOffline(true);
      });
    return () => {
      abort.abort();
      controller.current?.abort();
    };
  }, []);
  function edit(next: string) {
    controller.current?.abort();
    controller.current = null;
    setBusy(false);
    setText(next);
    setResult(null);
    setError("");
  }
  async function analyze() {
    if (!text.trim() || busy) return;
    const abort = new AbortController();
    controller.current = abort;
    setBusy(true);
    setError("");
    setResult(null);
    let timedOut = false;
    const timeout = window.setTimeout(() => {
      timedOut = true;
      abort.abort();
    }, 30000);
    try {
      const response = await request<Prediction>("/api/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text }),
        signal: abort.signal,
      });
      if (!validPrediction(response))
        throw new Error(
          "The service returned an invalid prediction. Please try again.",
        );
      if (!abort.signal.aborted) setResult(response);
    } catch (e) {
      if (timedOut && controller.current === abort)
        setError("The classifier took too long. Please try again.");
      else if (!abort.signal.aborted)
        setError(
          e instanceof TypeError
            ? "Cannot reach the classifier. Start the API server and try again."
            : (e as Error).message,
        );
    } finally {
      window.clearTimeout(timeout);
      if (controller.current === abort) {
        setBusy(false);
        controller.current = null;
      }
    }
  }
  const primary = result?.models[0];
  return (
    <>
      <div className="demo-meta">
        <div className={`status-pill ${health ? "online" : ""}`}>
          <span />
          {health
            ? "Classifier online"
            : offline
              ? "Classifier unavailable"
              : "Checking classifier"}
        </div>
        <span>
          <Languages size={15} /> Bangla, Banglish & English
        </span>
        <span>
          <ShieldCheck size={15} /> Your text is never saved
        </span>
      </div>
      <div className="demo-grid">
        <Card className="editor-card">
          <div className="card-heading">
            <div>
              <span className="section-number">01</span>
              <h2>Start with a post</h2>
            </div>
            <span className="subtle">Your words, our model</span>
          </div>
          <label className="field-label" htmlFor="post">
            What would you like to analyze?
          </label>
          <div className="textarea-shell">
            <textarea
              id="post"
              ref={editor}
              value={text}
              maxLength={2000}
              onChange={(e) => edit(e.target.value)}
              placeholder="Write or paste a post in Bangla, Banglish or English…"
              onKeyDown={(e) => {
                if ((e.ctrlKey || e.metaKey) && e.key === "Enter") {
                  e.preventDefault();
                  void analyze();
                }
              }}
            />
            <div className="editor-bottom">
              <span>
                <Languages size={14} /> Script detected automatically
              </span>
              <span>{text.length.toLocaleString()} / 2,000</span>
            </div>
          </div>
          <div className="editor-actions">
            <button
              className="text-button"
              onClick={() => {
                edit("");
                editor.current?.focus();
              }}
              disabled={!text}
            >
              <RotateCcw size={14} /> Clear text
            </button>
            <button
              className="primary-button"
              onClick={() => void analyze()}
              disabled={!text.trim() || busy}
            >
              {busy ? (
                <LoaderCircle className="spin" size={17} />
              ) : (
                <Sparkles size={17} />
              )}{" "}
              {busy ? "Analyzing…" : "Analyze post"}{" "}
              {!busy && <ArrowRight size={17} />}
            </button>
          </div>
          <p className="keyboard-tip">
            <CornerDownLeft size={13} /> Ctrl / ⌘ + Enter to analyze
          </p>
          {error && (
            <div className="error-message" role="alert">
              <X size={16} />
              <span>{error}</span>
            </div>
          )}
          <div className="examples-section">
            <div className="examples-heading">
              <h3>Need a starting point?</h3>
              <div className="select-shell">
                <label className="sr-only" htmlFor="example-language">
                  Example language
                </label>
                <select
                  id="example-language"
                  value={language}
                  onChange={(e) => setLanguage(e.target.value)}
                >
                  {["English", "Bangla", "Banglish"].map((l) => (
                    <option key={l}>{l}</option>
                  ))}
                </select>
                <ChevronDown size={13} />
              </div>
            </div>
            <p className="subtle">
              Try an original example. Then make it your own.
            </p>
            <div className="example-list">
              {examples
                .filter((e) => e.language === language)
                .filter((_, i) => i % 2 === exampleSet)
                .map((e) => (
                  <button
                    className="example-button"
                    key={e.id}
                    onClick={() => {
                      edit(e.text);
                      editor.current?.focus();
                    }}
                  >
                    <Badge label={e.label} />
                    <span lang={e.language === "Bangla" ? "bn" : undefined}>
                      {e.text}
                    </span>
                    <ArrowUpRight size={15} />
                  </button>
                ))}
            </div>
            <p className="example-caption">
              Hand-written examples · badges show intended classes
            </p>
            <button
              className="text-button"
              onClick={() => setExampleSet(1 - exampleSet)}
            >
              <RotateCcw size={13} /> Try more examples
            </button>
          </div>
        </Card>
        <Card className="prediction-card">
          <div className="card-heading">
            <div>
              <span className="section-number">02</span>
              <h2>The model’s reading</h2>
            </div>
            <ScanText size={19} className="muted-icon" />
          </div>
          <div aria-live="polite" aria-busy={busy}>
            {!primary ? (
              <div className="empty-prediction">
                <div className="empty-orbit">
                  <Waves size={32} />
                  <i />
                  <i />
                </div>
                <h3>
                  {busy
                    ? "Reading between the lines…"
                    : "Every post tells a story."}
                </h3>
                <p>
                  {busy
                    ? "The classifier is analyzing your words."
                    : "Analyze a post to explore its class, confidence and word-level signals."}
                </p>
                <div className="class-legend">
                  {labels.map((_, i) => (
                    <Badge key={i} label={i} />
                  ))}
                </div>
              </div>
            ) : (
              <>
                <div className={`prediction-hero prediction-${primary.label}`}>
                  <div className="prediction-caption">
                    PREDICTED CLASS <span>{result?.script.name}</span>
                  </div>
                  <div className="prediction-label">
                    <h3>{labels[primary.label]}</h3>
                    <span>
                      {pct(primary.probs[primary.label])}
                      <small>model probability</small>
                    </span>
                  </div>
                  <p>
                    {
                      [
                        "Directly hostile or insulting language.",
                        "Indirect hostility, insinuation or dismissive language.",
                        "No explicit or subtle toxicity predicted.",
                      ][primary.label]
                    }
                  </p>
                </div>
                <div className="probabilities">
                  <h3>
                    Class probabilities <CircleHelp size={14} />
                  </h3>
                  {labels.map((name, i) => (
                    <div className="probability-row" key={name}>
                      <div>
                        <span>
                          <i className={`class-dot class-${i}`} />
                          {name}
                        </span>
                        <strong>{pct(primary.probs[i])}</strong>
                      </div>
                      <div className="bar-track">
                        <div
                          className={`bar-fill class-${i}`}
                          style={{ width: pct(primary.probs[i]) }}
                        />
                      </div>
                    </div>
                  ))}
                </div>
                {!primary.has_features && (
                  <p className="notice">
                    No known features were found. This prediction has little
                    textual evidence.
                  </p>
                )}
                <div className="reading-footer">
                  <Check size={14} />
                  <span>{result?.script.name} detected</span>
                  <span>{result?.latency_ms} ms</span>
                </div>
              </>
            )}
          </div>
          <div className="model-status">
            <FlaskConical size={17} />
            <div>
              <strong>
                {primary?.name ??
                  health?.models[0]?.name ??
                  "CPU text classifier"}
              </strong>
              <p>
                {(primary?.mode ?? health?.models[0]?.mode) === "local-trained"
                  ? "Local trained model · CPU inference"
                  : "Illustrative TF-IDF model · 18 authored examples"}
              </p>
            </div>
          </div>
        </Card>
      </div>
      {result && (
        <div className="analysis-grid">
          <Card>
            <div className="card-heading">
              <div>
                <span className="section-number">03</span>
                <h2>Signals in the words</h2>
              </div>
              <span className="subtle">TF-IDF influence</span>
            </div>
            <p className="subtle">
              Highlighted words support the TF-IDF prediction. Stronger color
              means a larger probability change when removed.
            </p>
            <div className="highlighted-text">
              {result.tokens.map((t, i) => (
                <mark
                  key={i}
                  className={
                    t.weight > 0.002
                      ? "supports"
                      : t.weight < -0.002
                        ? "opposes"
                        : ""
                  }
                  style={{
                    backgroundColor:
                      t.weight > 0.002
                        ? `rgba(112,38,66,${Math.min(0.42, 0.07 + t.weight * 2)})`
                        : undefined,
                  }}
                  title={`Deletion influence: ${t.weight.toFixed(4)}`}
                >
                  {t.text}
                </mark>
              ))}
            </div>
            <p className="chart-caption">
              {result.explanation}
              {result.explanation_truncated
                ? " Only the first 80 tokens were tested."
                : ""}
            </p>
          </Card>
          <Card>
            <div className="card-heading">
              <div>
                <span className="section-number">04</span>
                <h2>Served model votes</h2>
              </div>
            </div>
            {result.models.map((m) => (
              <div className="vote-row" key={m.id}>
                <div>
                  <strong>{m.name}</strong>
                  <small>{pct(m.probs[m.label])} probability</small>
                </div>
                <Badge label={m.label} />
              </div>
            ))}
            {!health?.transformer_available && (
              <p className="chart-caption">
                Transformer and LLM results are available on the research pages.
                Live neural votes require saved checkpoints.
              </p>
            )}
          </Card>
        </div>
      )}
      <div className="principles-grid">
        {[
          {
            icon: ScanText,
            title: "Beyond a keyword",
            text: "Subtle toxicity can hide in sarcasm, exclusion and context.",
          },
          {
            icon: Languages,
            title: "Script ≠ language",
            text: "Banglish and English share Latin letters. Script detection is not language detection.",
          },
          {
            icon: ShieldCheck,
            title: "An aid to human judgment",
            text:
              result?.warning ??
              (health?.models[0]?.mode === "local-trained"
                ? "A local competition-trained model is serving this demo. Its probabilities deserve careful interpretation."
                : "The default model is trained on authored examples. Treat its probabilities as illustrative."),
          },
        ].map((p) => (
          <div className="principle" key={p.title}>
            <span className="principle-icon">
              <p.icon size={19} />
            </span>
            <div>
              <h3>{p.title}</h3>
              <p>{p.text}</p>
            </div>
          </div>
        ))}
      </div>
    </>
  );
}
