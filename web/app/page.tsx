"use client";

import { useCallback, useMemo, useRef, useState } from "react";
import {
  ArrowRight,
  BarChart3,
  Check,
  ChevronDown,
  FileSpreadsheet,
  LoaderCircle,
  RotateCcw,
  ShieldCheck,
  Sparkles,
  Upload,
  X,
} from "lucide-react";
import Papa from "papaparse";

type Recommendation = {
  rank: number;
  model: string;
  score: number;
  probe_score: number;
};

type Result = {
  action: "top_1" | "top_2" | "top_3";
  risk: number;
  recommendations: Recommendation[];
  dataset: {
    rows: number;
    features: number;
    classes: number;
  };
};

const MODEL_LABELS: Record<string, string> = {
  logistic: "Logistic Regression",
  decision_tree: "Decision Tree",
  random_forest: "Random Forest",
  extra_trees: "Extra Trees",
  hist_gradient_boosting: "Histogram Gradient Boosting",
};

const API_URL =
  process.env.NEXT_PUBLIC_API_URL ||
  (process.env.NODE_ENV === "production"
    ? "https://mlselector.vercel.app"
    : "http://localhost:8000");

function formatNumber(value: number) {
  return new Intl.NumberFormat("en-US").format(value);
}

export default function Home() {
  const inputRef = useRef<HTMLInputElement>(null);
  const [file, setFile] = useState<File | null>(null);
  const [columns, setColumns] = useState<string[]>([]);
  const [rowCount, setRowCount] = useState(0);
  const [excludedColumns, setExcludedColumns] = useState(0);
  const [target, setTarget] = useState("");
  const [dragging, setDragging] = useState(false);
  const [status, setStatus] = useState<
    "idle" | "reading" | "ready" | "analyzing" | "done" | "error"
  >("idle");
  const [error, setError] = useState("");
  const [result, setResult] = useState<Result | null>(null);

  const fileSize = useMemo(() => {
    if (!file) return "";
    const value = file.size / 1024;
    return value > 1024
      ? `${(value / 1024).toFixed(1)} MB`
      : `${Math.round(value)} KB`;
  }, [file]);

  const reset = useCallback(() => {
    setFile(null);
    setColumns([]);
    setRowCount(0);
    setExcludedColumns(0);
    setTarget("");
    setStatus("idle");
    setError("");
    setResult(null);
    if (inputRef.current) inputRef.current.value = "";
  }, []);

  const readFile = useCallback((nextFile: File) => {
    if (!nextFile.name.toLowerCase().endsWith(".csv")) {
      setError("Please choose a CSV file.");
      setStatus("error");
      return;
    }

    setFile(nextFile);
    setStatus("reading");
    setError("");
    setResult(null);

    Papa.parse<Record<string, string>>(nextFile, {
      header: true,
      skipEmptyLines: true,
      preview: 10000,
      complete: ({ meta, data, errors }) => {
        if (errors.length || !meta.fields?.length) {
          setError("We could not read this CSV. Check the file and try again.");
          setStatus("error");
          return;
        }

        const nextColumns = meta.fields.filter(Boolean);
        const viableColumns = nextColumns
          .map((column) => {
            const values = data
              .map((row) => String(row[column] ?? "").trim())
              .filter(Boolean);
            const counts = new Map<string, number>();
            values.forEach((value) => {
              counts.set(value, (counts.get(value) || 0) + 1);
            });
            const unique = counts.size;
            const smallestClass = Math.min(...counts.values());
            const nameScore = /^(target|label|class|outcome|type|status|rating|category|churn|survived)$/i.test(column)
              ? 2
              : /(target|label|class|outcome|type|status|rating|category|churn)/i.test(column)
                ? 1
                : 0;
            return {
              column,
              viable:
                unique >= 2 &&
                unique <= Math.min(100, Math.floor(values.length / 2)) &&
                smallestClass >= 2,
              nameScore,
              unique,
            };
          })
          .filter((profile) => profile.viable)
          .sort(
            (first, second) =>
              second.nameScore - first.nameScore ||
              first.unique - second.unique
          )
          .map((profile) => profile.column);

        setColumns(viableColumns);
        setExcludedColumns(nextColumns.length - viableColumns.length);
        setRowCount(data.length);
        setTarget(viableColumns[0] || "");
        setStatus(viableColumns.length ? "ready" : "error");
        if (!viableColumns.length) {
          setError(
            "No suitable classification target was found. A target needs repeated labels with at least two rows per class."
          );
        }
      },
      error: () => {
        setError("We could not read this CSV. Check the file and try again.");
        setStatus("error");
      },
    });
  }, []);

  const analyze = async () => {
    if (!file || !target) return;

    setStatus("analyzing");
    setError("");

    const body = new FormData();
    body.append("file", file);
    body.append("target", target);

    try {
      const response = await fetch(`${API_URL}/recommend`, {
        method: "POST",
        body,
      });
      const payload = await response.json();

      if (!response.ok) {
        throw new Error(payload.detail || "Analysis failed.");
      }

      setResult(payload);
      setStatus("done");
    } catch (requestError) {
      setError(
        requestError instanceof Error
          ? requestError.message
          : "The analysis service is unavailable."
      );
      setStatus("error");
    }
  };

  return (
    <main>
      <nav className="nav">
        <a className="brand" href="#">
          <span className="brand-mark">
            <Sparkles size={16} strokeWidth={2.4} />
          </span>
          Modelwise
        </a>
        <span className="github-link">Open-source portfolio</span>
      </nav>

      <section className="hero">
        <div className="eyebrow">
          <span />
          Uncertainty-aware model selection
        </div>
        <h1>
          Find the right model
          <br />
          <span>before training them all.</span>
        </h1>
        <p>
          Upload a classification dataset. Modelwise runs lightweight probes
          and returns a risk-aware shortlist in under a minute.
        </p>
      </section>

      <section className="workspace">
        <div className="step-heading">
          <span>01</span>
          <div>
            <h2>Upload your dataset</h2>
            <p>Your file stays local until you start the analysis.</p>
          </div>
        </div>

        {!file ? (
          <button
            className={`dropzone ${dragging ? "dragging" : ""}`}
            onClick={() => inputRef.current?.click()}
            onDragEnter={(event) => {
              event.preventDefault();
              setDragging(true);
            }}
            onDragOver={(event) => event.preventDefault()}
            onDragLeave={() => setDragging(false)}
            onDrop={(event) => {
              event.preventDefault();
              setDragging(false);
              const nextFile = event.dataTransfer.files[0];
              if (nextFile) readFile(nextFile);
            }}
          >
            <span className="upload-icon">
              <Upload size={24} />
            </span>
            <strong>Drop your CSV here</strong>
            <span>or click to browse · up to 25 MB</span>
          </button>
        ) : (
          <div className="file-card">
            <span className="file-icon">
              <FileSpreadsheet size={22} />
            </span>
            <div className="file-copy">
              <strong>{file.name}</strong>
              <span>
                {fileSize}
                {rowCount ? ` · ${formatNumber(rowCount)} rows previewed` : ""}
              </span>
            </div>
            <span className="file-ready">
              <Check size={14} />
              Ready
            </span>
            <button className="icon-button" onClick={reset} aria-label="Remove file">
              <X size={18} />
            </button>
          </div>
        )}

        <input
          ref={inputRef}
          type="file"
          accept=".csv,text/csv"
          hidden
          onChange={(event) => {
            const nextFile = event.target.files?.[0];
            if (nextFile) readFile(nextFile);
          }}
        />

        {file && columns.length > 0 && (
          <div className="target-row">
            <label htmlFor="target">Target column <span className="target-hint">· {excludedColumns} unsuitable hidden</span></label>
            <div className="select-wrap">
              <select
                id="target"
                value={target}
                onChange={(event) => {
                  setTarget(event.target.value);
                  setResult(null);
                  setStatus("ready");
                }}
              >
                {columns.map((column) => (
                  <option key={column} value={column}>
                    {column}
                  </option>
                ))}
              </select>
              <ChevronDown size={17} />
            </div>
            <button
              className="analyze-button"
              onClick={analyze}
              disabled={status === "analyzing" || status === "reading"}
            >
              {status === "analyzing" ? (
                <>
                  <LoaderCircle className="spin" size={17} />
                  Comparing models
                </>
              ) : (
                <>
                  Analyze dataset
                  <ArrowRight size={17} />
                </>
              )}
            </button>
          </div>
        )}

        {error && (
          <div className="error-message">
            <span>{error}</span>
            <button onClick={() => setStatus(file ? "ready" : "idle")}>
              Try again
            </button>
          </div>
        )}
      </section>

      {result ? (
        <section className="results">
          <div className="results-topline">
            <div>
              <span className="section-label">Recommendation</span>
              <h2>
                {result.action === "top_1"
                  ? "A clear first choice"
                  : result.action === "top_2"
                    ? "Two models worth testing"
                    : "A close three-model race"}
              </h2>
            </div>
            <div className={`risk-badge risk-${result.action}`}>
              <ShieldCheck size={17} />
              {(result.risk * 100).toFixed(0)}% selection risk
            </div>
          </div>

          <div className="recommendation-list">
            {result.recommendations.map((recommendation, index) => (
              <article
                className={`recommendation ${index === 0 ? "primary" : ""}`}
                key={recommendation.model}
              >
                <span className="rank">
                  {String(recommendation.rank).padStart(2, "0")}
                </span>
                <div className="model-name">
                  <strong>
                    {MODEL_LABELS[recommendation.model] ||
                      recommendation.model}
                  </strong>
                  <span>
                    {index === 0
                      ? "Recommended"
                      : "Strong alternative"}
                  </span>
                </div>
                <div className="score-block">
                  <div className="score-label">
                    <span>Suitability</span>
                    <strong>{Math.round(recommendation.score * 100)}%</strong>
                  </div>
                  <div className="score-track">
                    <span
                      style={{
                        width: `${Math.max(
                          recommendation.score * 100,
                          4
                        )}%`,
                      }}
                    />
                  </div>
                </div>
                <span className="probe">
                  Probe {Math.round(recommendation.probe_score * 100)}%
                </span>
              </article>
            ))}
          </div>

          <div className="dataset-strip">
            <div>
              <span>Rows</span>
              <strong>{formatNumber(result.dataset.rows)}</strong>
            </div>
            <div>
              <span>Features</span>
              <strong>{formatNumber(result.dataset.features)}</strong>
            </div>
            <div>
              <span>Classes</span>
              <strong>{formatNumber(result.dataset.classes)}</strong>
            </div>
            <button onClick={reset}>
              <RotateCcw size={15} />
              New dataset
            </button>
          </div>
        </section>
      ) : (
        <section className="proof">
          <div className="proof-item">
            <BarChart3 size={18} />
            <div>
              <strong>5 candidates</strong>
              <span>Compared with lightweight probes</span>
            </div>
          </div>
          <div className="proof-item">
            <ShieldCheck size={18} />
            <div>
              <strong>Calibrated risk</strong>
              <span>Know when to test more than one</span>
            </div>
          </div>
          <div className="proof-stat">
            <strong>0.90</strong>
            <span>mean regret, percentage points</span>
          </div>
        </section>
      )}

      <footer>
        <span>Built for thoughtful model selection.</span>
        <span>Validate recommendations on your own holdout data.</span>
      </footer>
    </main>
  );
}
