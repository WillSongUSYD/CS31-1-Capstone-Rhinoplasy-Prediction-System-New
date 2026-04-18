import { useEffect, useState } from "react";

const API_BASE = "http://127.0.0.1:8000";
const MODELS = ["pix2pix", "autoencoder", "cyclegan", "diffusion"];

function ImageSlider({ beforeUrl, afterUrl }) {
  const [position, setPosition] = useState(50);
  if (!beforeUrl || !afterUrl) return null;
  return (
    <div className="slider-card">
      <div className="slider-stage">
        <img src={beforeUrl} alt="before" className="slider-image" />
        <div className="slider-overlay" style={{ width: `${position}%` }}>
          <img src={afterUrl} alt="after" className="slider-image" />
        </div>
      </div>
      <input type="range" min="0" max="100" value={position} onChange={(e) => setPosition(Number(e.target.value))} />
    </div>
  );
}

function LossCurve({ modelName }) {
  const [history, setHistory] = useState([]);
  useEffect(() => {
    fetch(`${API_BASE}/api/training-history/${modelName}`)
      .then((r) => r.json())
      .then((data) => setHistory(data.history || []))
      .catch(() => {});
  }, [modelName]);

  if (!history.length) return <p>No training history for {modelName}.</p>;

  const maxLoss = Math.max(...history.map((h) => Math.max(h.train_loss, h.val_l1)));
  const h = 160;
  const w = 400;
  const pad = 40;

  return (
    <div className="card">
      <h4>{modelName} Loss Curve</h4>
      <svg viewBox={`0 0 ${w + pad * 2} ${h + pad * 2}`} className="loss-chart">
        {history.map((pt, i) => {
          const x = pad + (i / Math.max(1, history.length - 1)) * w;
          const yTrain = pad + (1 - pt.train_loss / maxLoss) * h;
          const yVal = pad + (1 - pt.val_l1 / maxLoss) * h;
          return (
            <g key={i}>
              {i > 0 && (
                <>
                  <line
                    x1={pad + ((i - 1) / Math.max(1, history.length - 1)) * w}
                    y1={pad + (1 - history[i - 1].train_loss / maxLoss) * h}
                    x2={x} y2={yTrain} stroke="#0d5c63" strokeWidth="2"
                  />
                  <line
                    x1={pad + ((i - 1) / Math.max(1, history.length - 1)) * w}
                    y1={pad + (1 - history[i - 1].val_l1 / maxLoss) * h}
                    x2={x} y2={yVal} stroke="#e07020" strokeWidth="2"
                  />
                </>
              )}
              <circle cx={x} cy={yTrain} r="3" fill="#0d5c63" />
              <circle cx={x} cy={yVal} r="3" fill="#e07020" />
            </g>
          );
        })}
        <text x={w + pad + 5} y={pad + 10} fontSize="10" fill="#0d5c63">Train</text>
        <text x={w + pad + 5} y={pad + 24} fontSize="10" fill="#e07020">Val L1</text>
      </svg>
      <p className="chart-caption">
        Epochs: {history.length} | Best Val L1: {Math.min(...history.map((h) => h.val_l1)).toFixed(4)}
      </p>
    </div>
  );
}

export default function App() {
  const [tab, setTab] = useState("upload");
  const [file, setFile] = useState(null);
  const [modelName, setModelName] = useState("pix2pix");
  const [pairedInput, setPairedInput] = useState(true);
  const [prediction, setPrediction] = useState(null);
  const [history, setHistory] = useState([]);
  const [benchmarks, setBenchmarks] = useState([]);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");

  async function loadHistory() {
    const r = await fetch(`${API_BASE}/api/history`);
    setHistory(await r.json());
  }
  async function loadBenchmarks() {
    const r = await fetch(`${API_BASE}/api/benchmarks`);
    setBenchmarks(await r.json());
  }

  useEffect(() => {
    loadHistory().catch(() => {});
    loadBenchmarks().catch(() => {});
  }, []);

  async function handleSubmit(e) {
    e.preventDefault();
    if (!file) { setError("Please choose an image first."); return; }
    setBusy(true);
    setError("");
    const form = new FormData();
    form.append("file", file);
    form.append("model_name", modelName);
    form.append("paired_input", String(pairedInput));
    try {
      const r = await fetch(`${API_BASE}/api/predict`, { method: "POST", body: form });
      const payload = await r.json();
      if (!r.ok) throw new Error(payload.detail || "Prediction failed.");
      setPrediction(payload);
      setTab("prediction");
      await loadHistory();
    } catch (err) {
      setError(err.message);
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="page-shell">
      <header className="hero">
        <div>
          <p className="eyebrow">CS31 Research Prototype</p>
          <h1>Profile-to-Profile Rhinoplasty Outcome Prediction</h1>
          <p className="subtitle">
            Upload a paired WhatsApp export or a standalone pre-op profile to inspect the current local model.
          </p>
        </div>
      </header>

      <nav className="tabs">
        {["upload", "prediction", "benchmark", "compare", "about"].map((item) => (
          <button key={item} className={tab === item ? "tab active" : "tab"} onClick={() => setTab(item)}>
            {item[0].toUpperCase() + item.slice(1)}
          </button>
        ))}
      </nav>

      {/* ===== UPLOAD TAB ===== */}
      {tab === "upload" && (
        <section className="panel">
          <h2>Upload</h2>
          <form onSubmit={handleSubmit} className="form-grid">
            <label>
              Model
              <select value={modelName} onChange={(e) => setModelName(e.target.value)}>
                {MODELS.map((m) => <option key={m} value={m}>{m}</option>)}
              </select>
            </label>
            <label>
              Input mode
              <select value={pairedInput ? "paired" : "pre_only"} onChange={(e) => setPairedInput(e.target.value === "paired")}>
                <option value="paired">Paired WhatsApp canvas</option>
                <option value="pre_only">Pre-op only</option>
              </select>
            </label>
            <label className="file-input">
              Image file
              <input type="file" accept="image/*" onChange={(e) => setFile(e.target.files?.[0] || null)} />
            </label>
            <button type="submit" className="primary-button" disabled={busy}>
              {busy ? "Running..." : "Run prediction"}
            </button>
          </form>
          {error && <p className="error">{error}</p>}
          <div className="hint-grid">
            <div className="card">
              <h3>Scope</h3>
              <p>Profile-to-profile outcome prediction with four trained models. Facial landmark detection and NLP-based surgery description are integrated.</p>
            </div>
            <div className="card">
              <h3>Accepted input</h3>
              <p>Side-profile rhinoplasty images. Paired mode expects left=pre-op, right=post-op. Frontal images are excluded from training.</p>
            </div>
          </div>
        </section>
      )}

      {/* ===== PREDICTION TAB ===== */}
      {tab === "prediction" && (
        <section className="panel">
          <h2>Prediction</h2>
          {prediction ? (
            <>
              <div className="image-grid">
                <figure className="card">
                  <img src={`${API_BASE}${prediction.pre_image_url}`} alt="pre-op" />
                  <figcaption>Pre-op input</figcaption>
                </figure>
                {prediction.reference_post_url && (
                  <figure className="card">
                    <img src={`${API_BASE}${prediction.reference_post_url}`} alt="reference post-op" />
                    <figcaption>Real post-op reference</figcaption>
                  </figure>
                )}
                <figure className="card">
                  <img src={`${API_BASE}${prediction.generated_post_url}`} alt="generated post-op" />
                  <figcaption>Generated post-op</figcaption>
                </figure>
              </div>

              <ImageSlider
                beforeUrl={`${API_BASE}${prediction.pre_image_url}`}
                afterUrl={`${API_BASE}${prediction.generated_post_url}`}
              />

              {/* Surgery Description */}
              {prediction.description && (
                <div className="card description-card">
                  <h3>Surgery Description (NLP)</h3>
                  <p className="desc-summary">{prediction.description.summary}</p>
                  <ul className="change-list">
                    {prediction.description.changes.map((c, i) => <li key={i}>{c}</li>)}
                  </ul>
                  {prediction.description.metrics && (
                    <details>
                      <summary>Detail metrics</summary>
                      <pre>{JSON.stringify(prediction.description.metrics, null, 2)}</pre>
                    </details>
                  )}
                </div>
              )}

              {/* Landmarks */}
              {prediction.landmarks && (
                <div className="card">
                  <h3>Facial Landmarks</h3>
                  <p>View type: <strong>{prediction.landmarks.view_type}</strong></p>
                  {prediction.landmarks.nose_features && (
                    <div className="feature-grid">
                      {Object.entries(prediction.landmarks.nose_features).map(([k, v]) => (
                        <div key={k} className="feature-chip">
                          <span className="feature-label">{k.replace(/_/g, " ")}</span>
                          <span className="feature-value">{typeof v === "number" ? v.toFixed(3) : v}</span>
                        </div>
                      ))}
                    </div>
                  )}
                </div>
              )}

              <div className="card">
                <h3>Metadata</h3>
                <p>Model: {prediction.model_name} | Input mode: {prediction.input_mode}</p>
                <p className="disclaimer">{prediction.disclaimer}</p>
                {Object.keys(prediction.metrics).length > 0 && (
                  <pre>{JSON.stringify(prediction.metrics, null, 2)}</pre>
                )}
              </div>
            </>
          ) : (
            <p>No prediction yet. Use the Upload tab to run one.</p>
          )}

          <div className="card">
            <h3>History</h3>
            <div className="history-list">
              {history.map((item) => (
                <article key={item.id} className="history-item">
                  <div>
                    <strong>{item.model_name}</strong>
                    <p>{item.created_at}</p>
                  </div>
                  <div className="history-links">
                    <a href={`${API_BASE}${item.pre_url}`} target="_blank" rel="noreferrer">Pre-op</a>
                    <a href={`${API_BASE}${item.generated_url}`} target="_blank" rel="noreferrer">Generated</a>
                    {item.reference_url && (
                      <a href={`${API_BASE}${item.reference_url}`} target="_blank" rel="noreferrer">Reference</a>
                    )}
                  </div>
                </article>
              ))}
            </div>
          </div>
        </section>
      )}

      {/* ===== BENCHMARK TAB ===== */}
      {tab === "benchmark" && (
        <section className="panel">
          <h2>Benchmark</h2>
          <div className="table-wrapper">
            <table>
              <thead>
                <tr>
                  <th>Model</th><th>Samples</th><th>SSIM</th><th>ROI SSIM</th>
                  <th>LPIPS</th><th>ROI LPIPS</th><th>FID</th>
                </tr>
              </thead>
              <tbody>
                {benchmarks.map((row, i) => (
                  <tr key={`${row.model}-${i}`}>
                    <td>{row.model}</td>
                    <td>{row.sample_count}</td>
                    <td>{Number(row.ssim).toFixed(4)}</td>
                    <td>{Number(row.roi_ssim).toFixed(4)}</td>
                    <td>{Number(row.lpips).toFixed(4)}</td>
                    <td>{Number(row.roi_lpips).toFixed(4)}</td>
                    <td>{Number(row.fid).toFixed(2)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
            {!benchmarks.length && <p>No benchmark data yet. Run evaluation first.</p>}
          </div>
          <h3 style={{ marginTop: 24 }}>Training Loss Curves</h3>
          <div className="hint-grid">
            {MODELS.map((m) => <LossCurve key={m} modelName={m} />)}
          </div>
        </section>
      )}

      {/* ===== COMPARE TAB ===== */}
      {tab === "compare" && (
        <section className="panel">
          <h2>Model Comparison</h2>
          <p>Upload an image in the Upload tab. After prediction, all four models will be compared here.</p>
          {prediction && (
            <div className="card">
              <p>Current prediction uses <strong>{prediction.model_name}</strong>. To compare all models, run predictions with each model from the Upload tab.</p>
            </div>
          )}
          <div className="hint-grid">
            {MODELS.map((m) => (
              <div key={m} className="card">
                <h4>{m}</h4>
                <LossCurve modelName={m} />
              </div>
            ))}
          </div>
        </section>
      )}

      {/* ===== ABOUT TAB ===== */}
      {tab === "about" && (
        <section className="panel">
          <h2>About</h2>
          <div className="hint-grid">
            <div className="card">
              <h3>Implemented</h3>
              <p>Dataset indexing with duplicate-aware dedup (Union-Find), four trained models (Autoencoder, Pix2Pix, CycleGAN, Diffusion), facial landmark detection (MediaPipe), NLP surgery description, data augmentation, FastAPI backend, SQLite history, React web app.</p>
            </div>
            <div className="card">
              <h3>Data pipeline</h3>
              <p>2568 images scanned, 833 unique after dedup, 252 frontal views excluded, ~581 profile images used for training. Landmark-based dynamic ROI for evaluation.</p>
            </div>
            <div className="card">
              <h3>Clinical notice</h3>
              <p>This application is a course research prototype. It must not be used as medical advice or surgical planning software.</p>
            </div>
          </div>
        </section>
      )}
    </div>
  );
}
