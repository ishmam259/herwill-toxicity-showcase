import { useEffect, useState } from "react";
import {
  ArrowUpRight,
  FlaskConical,
  ChartNoAxesCombined,
  Grid2X2,
  SlidersHorizontal,
  LockKeyhole,
  PanelLeftClose,
  Menu,
  Github,
  ArrowRight,
  Sparkles,
} from "lucide-react";
import rawResults from "../../data/models.json";
import rawCalibration from "../../data/calibration.json";
import rawExamples from "../../data/examples.json";
import { type Results, type CalibrationData, type Example } from "./types";
import LiveDemo from "./LiveDemo";
import { Card } from "./components";
import { Comparison, Confusion, Calibration, HardCases } from "./ResultsPages";
const results = rawResults as Results;
const calibration = rawCalibration as CalibrationData;
const examples = rawExamples.examples as Example[];
const pages = [
  {
    path: "/",
    name: "Live demo",
    icon: FlaskConical,
    kicker: "THE INTERACTIVE LAB",
    title: "A little context. A lot of meaning.",
    description:
      "Explore how language shapes toxicity. Try a post in Bangla, Banglish or English and see how the model reads it.",
  },
  {
    path: "/compare",
    name: "Model comparison",
    icon: ChartNoAxesCombined,
    kicker: "THE MODEL BENCH",
    title: "Different models. Different strengths.",
    description:
      "Compare our historical text models across classes and script groups. Every number comes from a verified prediction artifact.",
  },
  {
    path: "/confusion",
    name: "Confusion matrices",
    icon: Grid2X2,
    kicker: "BEHIND THE SCORE",
    title: "See where the lines get blurred.",
    description:
      "A closer look at correct predictions and the mistakes between explicit, subtle and neutral content.",
  },
  {
    path: "/calibration",
    name: "Calibration & thresholds",
    icon: SlidersHorizontal,
    kicker: "THE CONFIDENCE CHECK",
    title: "Confidence is only part of the story.",
    description:
      "Explore reliability and see how changing class offsets shifts the historical macro F1 score.",
  },
  {
    path: "/hard-cases",
    name: "Hard cases",
    icon: LockKeyhole,
    kicker: "THE PRIVATE COLLECTION",
    title: "The posts that challenge the models.",
    description:
      "Inspect shared mistakes and cases that only the language model gets right. Competition posts stay on your local machine.",
  },
];
function currentPath() {
  return window.location.pathname.replace(/\/$/, "") || "/";
}
export default function App() {
  const [path, setPath] = useState(currentPath);
  const [menu, setMenu] = useState(false);
  const [mobile, setMobile] = useState(
    () => window.matchMedia("(max-width:700px)").matches,
  );
  useEffect(() => {
    const media = window.matchMedia("(max-width:700px)");
    const resize = () => setMobile(media.matches);
    media.addEventListener("change", resize);
    return () => media.removeEventListener("change", resize);
  }, []);
  useEffect(() => {
    const close = (e: KeyboardEvent) => {
      if (e.key === "Escape") setMenu(false);
    };
    window.addEventListener("keydown", close);
    return () => window.removeEventListener("keydown", close);
  }, []);
  const page = pages.find((p) => p.path === path);
  useEffect(() => {
    const onPop = () => {
      setPath(currentPath());
      setMenu(false);
    };
    window.addEventListener("popstate", onPop);
    return () => window.removeEventListener("popstate", onPop);
  }, []);
  useEffect(() => {
    document.title = `${page?.name ?? "Page not found"} · HerWILL Toxicity Lab`;
  }, [page]);
  function navigate(event: React.MouseEvent<HTMLAnchorElement>, next: string) {
    if (
      event.metaKey ||
      event.ctrlKey ||
      event.shiftKey ||
      event.altKey ||
      event.button !== 0
    )
      return;
    event.preventDefault();
    window.history.pushState({}, "", next);
    setPath(next);
    setMenu(false);
    window.scrollTo({ top: 0 });
    document.getElementById("main-content")?.focus();
  }
  return (
    <div className="app-shell">
      <a href="#main-content" className="skip-link">
        Skip to content
      </a>
      <aside
        inert={mobile && !menu}
        className={`sidebar ${menu ? "is-open" : ""}`}
      >
        <a
          className="brand"
          href="/"
          onClick={(e) => navigate(e, "/")}
          aria-label="HerWILL Toxicity Lab home"
        >
          <span className="brand-mark">
            h<span>✳</span>
          </span>
          <span>
            HerWILL<small>TOXICITY LAB</small>
          </span>
        </a>
        <div className="project-tag">
          <span /> event_horizon / research showcase
        </div>
        <div className="nav-label">EXPLORE THE RESEARCH</div>
        <nav aria-label="Main navigation">
          {pages.map((p) => (
            <a
              key={p.path}
              href={p.path}
              onClick={(e) => navigate(e, p.path)}
              className={path === p.path ? "active" : ""}
              aria-current={path === p.path ? "page" : undefined}
            >
              <p.icon size={19} />
              <span>{p.name}</span>
              {p.path === "/hard-cases" ? (
                <span className="local-tag">LOCAL</span>
              ) : path === p.path ? (
                <ArrowRight size={15} />
              ) : null}
            </a>
          ))}
        </nav>
        <div className="sidebar-bottom">
          <div className="research-card">
            <Sparkles size={21} />
            <p>
              Safer conversations,
              <br />
              <strong>one post at a time.</strong>
            </p>
            <small>
              HerWILL × UAP
              <br />
              Safe Social Media Datathon 2026
            </small>
          </div>
          <a
            className="repo-link"
            href="https://github.com/ishmam259/herwill-toxicity-showcase"
            target="_blank"
            rel="noreferrer"
          >
            <Github size={17} /> View the repository <ArrowUpRight size={14} />
          </a>
          <div className="sidebar-foot">
            <span className="team-avatar">eh</span>
            <span>
              Built by event_horizon<small>Bangladesh · 2026</small>
            </span>
          </div>
        </div>
      </aside>
      {menu && (
        <button
          className="menu-scrim"
          aria-label="Close navigation"
          onClick={() => setMenu(false)}
        />
      )}
      <div className="main-shell">
        <header className="topbar">
          <button
            className="menu-button"
            aria-label={menu ? "Close navigation" : "Open navigation"}
            aria-expanded={menu}
            onClick={() => setMenu(!menu)}
          >
            {menu ? <PanelLeftClose size={20} /> : <Menu size={20} />}
          </button>
          <div className="breadcrumb">
            Research showcase <span>/</span>{" "}
            <strong>{page?.name ?? "Page not found"}</strong>
          </div>
          <div className="topbar-note">
            <span /> Three classes. Many ways to say it.
          </div>
        </header>
        <main id="main-content" tabIndex={-1}>
          {page ? (
            <>
              <div className="page-heading">
                <div className="eyebrow">
                  <span />
                  {page.kicker}
                </div>
                <h1>{page.title}</h1>
                <p>{page.description}</p>
              </div>
              {path === "/" ? (
                <LiveDemo examples={examples} />
              ) : path === "/compare" ? (
                <Comparison data={results} />
              ) : path === "/confusion" ? (
                <Confusion data={results} />
              ) : path === "/calibration" ? (
                <Calibration data={results} calibration={calibration} />
              ) : (
                <HardCases />
              )}
            </>
          ) : (
            <Card>
              <h1>Page not found</h1>
              <p>That page doesn’t exist in the lab.</p>
              <a href="/" onClick={(e) => navigate(e, "/")}>
                Return to the live demo
              </a>
            </Card>
          )}
          <footer className="main-footer">
            <span>
              Language is nuanced. Model predictions deserve a human second
              look.
            </span>
            <span>
              HerWILL × UAP <span className="footer-dot">·</span> 2026
            </span>
          </footer>
        </main>
      </div>
    </div>
  );
}
