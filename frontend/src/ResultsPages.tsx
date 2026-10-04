import { useEffect, useState } from "react";
import {
  ArrowDownWideNarrow,
  ArrowRight,
  ChartNoAxesCombined,
  ChevronDown,
  Info,
  LockKeyhole,
  RotateCcw,
  ShieldCheck,
} from "lucide-react";
import { Badge, Card, EvidenceNote } from "./components";
import {
  labels,
  pct,
  score,
  request,
  type Results,
  type Model,
  type Summary,
  type CalibrationData,
  type Health,
  type HardCase,
} from "./types";
function Select({
  id,
  label,
  value,
  onChange,
  children,
}: {
  id: string;
  label: string;
  value: string;
  onChange: (s: string) => void;
  children: React.ReactNode;
}) {
  return (
    <div className="filter-field">
      <label htmlFor={id}>{label}</label>
      <div className="select-shell">
        <select
          id={id}
          value={value}
          onChange={(e) => onChange(e.target.value)}
        >
          {children}
        </select>
        <ChevronDown size={14} />
      </div>
    </div>
  );
}
function ModelSelect({
  data,
  id,
  onChange,
}: {
  data: Results;
  id: string;
  onChange: (s: string) => void;
}) {
  return (
    <Select id="model-select" label="Model" value={id} onChange={onChange}>
      {data.models.map((m) => (
        <option value={m.id} key={m.id}>
          {m.id} · {m.name}
        </option>
      ))}
    </Select>
  );
}
function sliceFor(m: Model, group: string): Summary | undefined {
  return group === "all" ? m : m.by_script[group];
}
export function Comparison({ data }: { data: Results }) {
  const [group, setGroup] = useState("all");
  const [family, setFamily] = useState("all");
  const [metric, setMetric] = useState("macro");
  const value = (m: Model) => {
    const s = sliceFor(m, group);
    return !s
      ? null
      : metric === "macro"
        ? s.macro_f1
        : s.per_class_f1[Number(metric)];
  };
  const models = data.models
    .filter((m) => family === "all" || m.family === family)
    .sort((a, b) => (value(b) ?? -1) - (value(a) ?? -1));
  const top = models.find((m) => value(m) !== null);
  return (
    <>
      <EvidenceNote />
      <div className="stats-grid">
        <Card>
          <span className="stat-label">TEXT MODELS</span>
          <strong>
            {data.models.length}
            <small>across 4 families</small>
          </strong>
          <ChartNoAxesCombined size={24} />
        </Card>
        <Card>
          <span className="stat-label">HISTORICAL OOF ROWS</span>
          <strong>
            {data.rows.toLocaleString()}
            <small>sorted-ID alignment verified</small>
          </strong>
          <ShieldCheck size={24} />
        </Card>
        <Card>
          <span className="stat-label">SELECTED SLICE LEADER</span>
          <strong>
            {top ? score(value(top)!) : "—"}
            <small>
              {top?.id ?? "No coverage"} ·{" "}
              {metric === "macro" ? "macro" : labels[Number(metric)]} F1
            </small>
          </strong>
          <ArrowDownWideNarrow size={24} />
        </Card>
      </div>
      <Card>
        <div className="card-heading">
          <div>
            <span className="section-number">01</span>
            <h2>A view across the models</h2>
          </div>
          <span className="subtle">Higher F1 is better</span>
        </div>
        <div className="filter-bar">
          <Select
            id="script-slice"
            label="Script group"
            value={group}
            onChange={setGroup}
          >
            <option value="all">All covered posts</option>
            <option value="B">Bangla-containing</option>
            <option value="L">Non-Bangla</option>
          </Select>
          <Select
            id="family-filter"
            label="Model family"
            value={family}
            onChange={setFamily}
          >
            <option value="all">All families</option>
            {["tfidf", "encoder", "llm", "ensemble"].map((f) => (
              <option key={f} value={f}>
                {f.toUpperCase()}
              </option>
            ))}
          </Select>
          <Select
            id="metric-filter"
            label="Metric"
            value={metric}
            onChange={setMetric}
          >
            <option value="macro">Macro F1</option>
            {labels.map((l, i) => (
              <option key={l} value={i}>
                {l} F1
              </option>
            ))}
          </Select>
        </div>
        <div className="ranking-chart" aria-label="Model F1 comparison">
          {models.map((m) => (
            <div className="ranking-row" key={m.id}>
              <div>
                <strong>{m.id}</strong>
                <span>{m.name}</span>
                <small>
                  {m.specialist ? "Bangla specialist" : m.family.toUpperCase()}
                </small>
              </div>
              <div className="ranking-track">
                <span style={{ width: `${(value(m) ?? 0) * 100}%` }} />
              </div>
              <strong>{value(m) === null ? "N/A" : score(value(m)!)}</strong>
            </div>
          ))}
        </div>
        <p className="chart-caption">
          {group === "all"
            ? "All covered posts: specialist rows differ from full-coverage models. Select Bangla-containing for the same-row comparison."
            : data.script_definition[group]}{" "}
          · F1 scale 0–1.
        </p>
      </Card>
      <Card className="table-card">
        <div className="card-heading">
          <div>
            <span className="section-number">02</span>
            <h2>Compare the details</h2>
          </div>
          <span className="subtle">Raw probability argmax</span>
        </div>
        <div
          className="table-scroll"
          tabIndex={0}
          role="region"
          aria-label="Scrollable results table"
        >
          <table>
            <caption className="sr-only">
              F1 scores and coverage for the selected script group and family
            </caption>
            <thead>
              <tr>
                <th scope="col">Model</th>
                <th scope="col">Macro F1</th>
                {labels.map((l, i) => (
                  <th key={l} scope="col">
                    <Badge label={i} />
                  </th>
                ))}
                <th scope="col">Covered rows</th>
              </tr>
            </thead>
            <tbody>
              {models.map((m) => {
                const s = sliceFor(m, group);
                return (
                  <tr key={m.id}>
                    <th scope="row">
                      <strong>{m.id}</strong>
                      <small>{m.name}</small>
                    </th>
                    <td className="emphasis">
                      {s ? score(s.macro_f1) : "N/A"}
                    </td>
                    {labels.map((l, i) => (
                      <td key={l}>{s ? score(s.per_class_f1[i]) : "N/A"}</td>
                    ))}
                    <td>{s ? s.rows.toLocaleString() : "No coverage"}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </Card>
    </>
  );
}
export function Confusion({ data }: { data: Results }) {
  const [id, setId] = useState("V3.11i");
  const [group, setGroup] = useState("all");
  const [normalized, setNormalized] = useState(true);
  const model = data.models.find((m) => m.id === id)!;
  const summary = sliceFor(model, group);
  const cm = summary?.confusion;
  const total = summary?.rows ?? 0;
  const correct = cm?.reduce((a, r, i) => a + r[i], 0) ?? 0;
  function changeModel(next: string) {
    setId(next);
    const m = data.models.find((m) => m.id === next)!;
    if (group !== "all" && !m.by_script[group]) setGroup("all");
  }
  return (
    <>
      <EvidenceNote />
      <Card>
        <div className="card-heading">
          <div>
            <span className="section-number">01</span>
            <h2>Where predictions land</h2>
          </div>
          <span className="subtle">
            Rows = true class · columns = predicted class
          </span>
        </div>
        <div className="filter-bar">
          <ModelSelect data={data} id={id} onChange={changeModel} />
          <Select
            id="matrix-script"
            label="Script group"
            value={group}
            onChange={setGroup}
          >
            <option value="all">All covered posts</option>
            {Object.keys(model.by_script).map((g) => (
              <option key={g} value={g}>
                {g === "B" ? "Bangla-containing" : "Non-Bangla"}
              </option>
            ))}
          </Select>
          <label className="checkbox-label">
            <input
              type="checkbox"
              checked={normalized}
              onChange={(e) => setNormalized(e.target.checked)}
            />{" "}
            Show row percentages
          </label>
        </div>
        {summary && cm ? (
          <div className="matrix-layout">
            <div>
              <div className="axis-caption">PREDICTED CLASS →</div>
              <div className="matrix-grid">
                <div className="matrix-corner">
                  TRUE
                  <br />
                  CLASS ↓
                </div>
                {labels.map((l, i) => (
                  <div className="matrix-label" key={l}>
                    <Badge label={i} />
                  </div>
                ))}
                {cm.map((row, i) => (
                  <div className="matrix-row" key={i}>
                    <div className="matrix-label">
                      <Badge label={i} />
                    </div>
                    {row.map((n, j) => {
                      const fraction = n / (summary.support[i] || 1);
                      return (
                        <div
                          key={j}
                          className={`matrix-cell ${i === j ? "diagonal" : ""} ${i !== j && i < 2 && j < 2 ? "subtle-explicit" : ""}`}
                          style={{
                            background:
                              i === j
                                ? `rgba(40,119,103,${0.07 + fraction * 0.42})`
                                : `rgba(174,61,69,${0.035 + fraction * 0.4})`,
                          }}
                        >
                          <strong>
                            {normalized ? pct(fraction) : n.toLocaleString()}
                          </strong>
                          <small>
                            {normalized
                              ? `${n.toLocaleString()} posts`
                              : `${pct(fraction)} of true ${labels[i]}`}
                          </small>
                        </div>
                      );
                    })}
                  </div>
                ))}
              </div>
              <p className="chart-caption">
                Rows sum to 100% when supported. Diagonal cells are correct
                predictions. Outlined cells show Explicit ↔ Subtle confusion.
              </p>
            </div>
            <div className="matrix-insights">
              <h3>The reading</h3>
              <div>
                <span>Covered posts</span>
                <strong>{total.toLocaleString()}</strong>
              </div>
              <div>
                <span>Correct predictions</span>
                <strong>{pct(correct / (total || 1))}</strong>
              </div>
              <div>
                <span>Macro F1</span>
                <strong>{score(summary.macro_f1)}</strong>
              </div>
              <div className="insight-callout">
                <Info size={17} />
                <p>
                  <strong>
                    {(cm[0][1] + cm[1][0]).toLocaleString()} posts
                  </strong>{" "}
                  cross the line between Explicit and Subtle. Indirect hostility
                  is easy to misread.
                </p>
              </div>
            </div>
          </div>
        ) : (
          <p>No predictions cover this script group.</p>
        )}
      </Card>
      <Card className="table-card">
        <div className="card-heading">
          <h2>Matrix as a table</h2>
          <span className="subtle">Exact post counts</span>
        </div>
        <div
          className="table-scroll"
          tabIndex={0}
          role="region"
          aria-label="Scrollable results table"
        >
          <table>
            <caption className="sr-only">
              True classes in rows, predicted classes in columns
            </caption>
            <thead>
              <tr>
                <th scope="col">True / predicted</th>
                {labels.map((l) => (
                  <th key={l} scope="col">
                    {l}
                  </th>
                ))}
                <th scope="col">Support</th>
              </tr>
            </thead>
            <tbody>
              {cm?.map((row, i) => (
                <tr key={i}>
                  <th scope="row">{labels[i]}</th>
                  {row.map((n, j) => (
                    <td key={j}>{n.toLocaleString()}</td>
                  ))}
                  <td>{summary?.support[i].toLocaleString()}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Card>
    </>
  );
}
export function Calibration({
  data,
  calibration,
}: {
  data: Results;
  calibration: CalibrationData;
}) {
  const [id, setId] = useState("V3.11i");
  const [subtle, setSubtle] = useState(10);
  const [neutral, setNeutral] = useState(10);
  const cal = calibration.models[id];
  const f1 = cal.macro_f1_grid[subtle][neutral];
  const base = cal.macro_f1_grid[10][10];
  const delta = f1 - base;
  const points = cal.bins.filter(
    (b) => b.count && b.confidence !== null && b.accuracy !== null,
  );
  return (
    <>
      <EvidenceNote />
      <div className="filter-bar standalone">
        <ModelSelect
          data={data}
          id={id}
          onChange={(s) => {
            setId(s);
            setSubtle(10);
            setNeutral(10);
          }}
        />
        <span className="subtle">
          {cal.rows.toLocaleString()} covered OOF posts · top-class confidence
        </span>
      </div>
      <div className="calibration-grid">
        <Card>
          <div className="card-heading">
            <div>
              <span className="section-number">01</span>
              <h2>Does confidence match reality?</h2>
            </div>
          </div>
          <p className="subtle">A reliable model falls near the diagonal.</p>
          <div className="reliability-chart">
            <svg
              viewBox="0 0 440 330"
              role="img"
              aria-label={`Reliability curve for ${id}. Expected calibration error ${pct(cal.ece)}. Exact bins are in the table below.`}
            >
              {[0, 0.25, 0.5, 0.75, 1].map((v) => (
                <g key={v}>
                  <line
                    x1="55"
                    y1={270 - v * 230}
                    x2="405"
                    y2={270 - v * 230}
                    stroke="#eee7e8"
                  />
                  <text x="42" y={275 - v * 230} textAnchor="end">
                    {Math.round(v * 100)}%
                  </text>
                  <text x={55 + v * 350} y="293" textAnchor="middle">
                    {Math.round(v * 100)}%
                  </text>
                </g>
              ))}
              <line
                x1="55"
                y1="270"
                x2="405"
                y2="40"
                stroke="#aaa2a7"
                strokeDasharray="5 6"
              />
              <polyline
                points={points
                  .map(
                    (b) =>
                      `${55 + b.confidence! * 350},${270 - b.accuracy! * 230}`,
                  )
                  .join(" ")}
                fill="none"
                stroke="#702642"
                strokeWidth="3"
              />
              {points.map((b) => (
                <circle
                  key={b.lower}
                  cx={55 + b.confidence! * 350}
                  cy={270 - b.accuracy! * 230}
                  r="5"
                  fill="#702642"
                >
                  <title>
                    {pct(b.confidence!)} confidence, {pct(b.accuracy!)}{" "}
                    accuracy, {b.count.toLocaleString()} posts
                  </title>
                </circle>
              ))}
              <text x="230" y="325" textAnchor="middle">
                Model confidence
              </text>
              <text
                transform="translate(15,160) rotate(-90)"
                textAnchor="middle"
              >
                Observed accuracy
              </text>
            </svg>
          </div>
          <div className="chart-legend">
            <span>
              <i /> Observed reliability
            </span>
            <span>
              <i className="dashed" /> Perfect calibration
            </span>
          </div>
          <div className="mini-stats">
            <div>
              <span>Calibration error (ECE)</span>
              <strong>{pct(cal.ece)}</strong>
            </div>
            <div>
              <span>Multiclass Brier score</span>
              <strong>{score(cal.brier)}</strong>
            </div>
          </div>
          <p className="chart-caption">
            10 equal-width confidence bins. Empty bins are omitted. Lower ECE
            and Brier scores indicate better calibration on these historical
            rows.
          </p>
        </Card>
        <Card>
          <div className="card-heading">
            <div>
              <span className="section-number">02</span>
              <h2>Adjust the decision boundary</h2>
            </div>
            <button
              className="icon-button"
              aria-label="Reset class offsets"
              onClick={() => {
                setSubtle(10);
                setNeutral(10);
              }}
            >
              <RotateCcw size={16} />
            </button>
          </div>
          <p className="subtle">
            Add offsets to log probabilities. Positive values favor that class;
            Explicit stays fixed at zero.
          </p>
          <div className="threshold-formula">
            argmax ( log p + class offset )
          </div>
          {[
            { label: "Subtle", index: 1, value: subtle, set: setSubtle },
            { label: "Neutral", index: 2, value: neutral, set: setNeutral },
          ].map((item) => (
            <div className="slider-field" key={item.label}>
              <div>
                <label htmlFor={`offset-${item.label}`}>
                  <Badge label={item.index} />
                  <span className="sr-only"> offset</span>
                </label>
                <output htmlFor={`offset-${item.label}`}>
                  {calibration.offsets[item.value] >= 0 ? "+" : ""}
                  {calibration.offsets[item.value].toFixed(2)}
                </output>
              </div>
              <input
                id={`offset-${item.label}`}
                aria-label={`${item.label} log-probability offset`}
                type="range"
                min="0"
                max="20"
                step="1"
                value={item.value}
                aria-valuetext={calibration.offsets[item.value].toFixed(2)}
                onChange={(e) => item.set(Number(e.target.value))}
              />
              <div className="range-labels">
                <span>−0.50</span>
                <span>0</span>
                <span>+0.50</span>
              </div>
            </div>
          ))}
          <div className="threshold-result" aria-live="polite">
            <span>HISTORICAL MACRO F1</span>
            <strong>
              {score(f1)}
              <small className={delta >= 0 ? "positive" : "negative"}>
                {delta >= 0 ? "+" : ""}
                {delta.toFixed(3)} vs zero offsets
              </small>
            </strong>
            <div className="bar-track">
              <div className="bar-fill plum" style={{ width: pct(f1) }} />
            </div>
          </div>
          <div className="info-note">
            <Info size={16} />
            <p>
              Every slider setting uses an exact aggregate computed from OOF
              probabilities. Tuning on these same labels can inflate the score;
              this is an exploration, not validation. Offsets do not change the
              live demo.
            </p>
          </div>
        </Card>
      </div>
      <Card className="table-card">
        <div className="card-heading">
          <h2>Reliability bins</h2>
          <span className="subtle">Unadjusted probabilities</span>
        </div>
        <div
          className="table-scroll"
          tabIndex={0}
          role="region"
          aria-label="Scrollable results table"
        >
          <table>
            <caption className="sr-only">
              Confidence bin support, mean confidence and observed accuracy
            </caption>
            <thead>
              <tr>
                <th scope="col">Confidence range</th>
                <th scope="col">Posts</th>
                <th scope="col">Mean confidence</th>
                <th scope="col">Observed accuracy</th>
              </tr>
            </thead>
            <tbody>
              {cal.bins.map((b) => (
                <tr key={b.lower}>
                  <th scope="row">
                    {Math.round(b.lower * 100)}–{Math.round(b.upper * 100)}%
                  </th>
                  <td>{b.count.toLocaleString()}</td>
                  <td>{b.confidence === null ? "—" : pct(b.confidence)}</td>
                  <td>{b.accuracy === null ? "—" : pct(b.accuracy)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Card>
    </>
  );
}
export function HardCases() {
  const [cases, setCases] = useState<HardCase[] | null>(null);
  const [category, setCategory] = useState("all-wrong");
  const [status, setStatus] = useState("loading");
  const [error, setError] = useState("");
  useEffect(() => {
    const controller = new AbortController();
    async function load() {
      try {
        const health = await request<Health>("/api/health", {
          signal: controller.signal,
        });
        if (!health.private_cases) {
          setStatus("locked");
          return;
        }
        const data = await request<{ cases: HardCase[] }>("/api/hard-cases", {
          signal: controller.signal,
        });
        setCases(data.cases);
        setStatus("ready");
      } catch (e) {
        if (!controller.signal.aborted) {
          setStatus("locked");
          setError((e as Error).message);
        }
      }
    }
    void load();
    return () => controller.abort();
  }, []);
  if (status !== "ready")
    return (
      <Card className="private-empty">
        <div className="private-icon">
          <LockKeyhole size={32} />
        </div>
        <span className="eyebrow">LOCAL ACCESS ONLY</span>
        <h2>
          {status === "loading"
            ? "Checking local access…"
            : "Some research belongs behind a closed door."}
        </h2>
        <p>
          Competition posts are excluded from public builds. This collection is
          available only when private mode is explicitly enabled on a local
          server.
        </p>
        <div className="private-details">
          <ShieldCheck size={20} />
          <div>
            <strong>Public showcase, private data</strong>
            <p>
              Use the local setup in the README to export private cases and
              start the API on 127.0.0.1. The app never bundles dataset posts
              into frontend assets.
            </p>
          </div>
        </div>
        {error && <p className="subtle">{error}</p>}
        <a
          className="text-link"
          href="https://github.com/ishmam259/herwill-toxicity-showcase#private-hard-cases"
          target="_blank"
          rel="noreferrer"
        >
          Local setup instructions <ArrowRight size={16} />
        </a>
      </Card>
    );
  const filtered = cases!.filter((c) => c.category === category);
  return (
    <>
      <div className="evidence-note">
        <LockKeyhole size={17} />
        <p>
          <strong>Private competition text.</strong> Do not share screenshots or
          deploy this collection. Specialist votes without script coverage
          appear as N/A.
        </p>
      </div>
      <div className="filter-bar standalone">
        <Select
          id="case-category"
          label="Case collection"
          value={category}
          onChange={setCategory}
        >
          <option value="all-wrong">Every covered model gets it wrong</option>
          <option value="llm-only">Only the LLM member gets it right</option>
        </Select>
        <span className="subtle">
          {filtered.length} posts · capped at 40 per collection
        </span>
      </div>
      <div className="hard-case-list">
        {filtered.map((c) => (
          <Card key={c.id}>
            <div className="case-heading">
              <span>
                Post #{c.id} ·{" "}
                {c.script === "B" ? "Bangla-containing" : "Non-Bangla"}
              </span>
              <div>
                <span>Supplied class</span>
                <Badge label={c.label} />
              </div>
            </div>
            <p className="case-text">{c.text}</p>
            <div className="case-votes">
              {c.votes.map((v) => (
                <div key={v.id}>
                  <strong>{v.id}</strong>
                  {v.label === null ? (
                    <span className="subtle">N/A</span>
                  ) : (
                    <Badge label={v.label} />
                  )}
                </div>
              ))}
            </div>
          </Card>
        ))}
        {!filtered.length && (
          <Card>
            <p>No cases match this collection.</p>
          </Card>
        )}
      </div>
    </>
  );
}
