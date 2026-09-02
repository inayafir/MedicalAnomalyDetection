import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { listPredictions, getHealth } from "../services/api";
import { formatDateTime, formatPercent } from "../lib/format";
import ErrorState from "../components/ErrorState";

export default function Dashboard() {
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [stats, setStats] = useState({ total: 0, normal: 0 });
  const [recent, setRecent] = useState([]);
  const [health, setHealth] = useState(null);

  async function load() {
    setLoading(true);
    setError(null);
    try {
      const [all, normal, recentPage, healthRes] = await Promise.all([
        listPredictions({ limit: 1 }),
        listPredictions({ limit: 1, predictedClass: "Normal" }),
        listPredictions({ limit: 6 }),
        getHealth().catch(() => null),
      ]);
      setStats({ total: all.total, normal: normal.total });
      setRecent(recentPage.items);
      setHealth(healthRes);
    } catch (e) {
      setError(e);
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    load();
  }, []);

  const abnormal = stats.total - stats.normal;

  return (
    <div className="mx-auto max-w-6xl px-6 py-12">
      <div className="mb-10 flex flex-col gap-3 border-b border-line pb-10 sm:flex-row sm:items-end sm:justify-between">
        <div>
          <h1 className="text-2xl font-medium tracking-tight text-ink">
            Chest X-ray analysis
          </h1>
          <p className="mt-2 max-w-lg text-sm text-ink-dim">
            Upload a chest X-ray to run it through the classifier and
            detector, and review Grad-CAM regions behind each finding.
          </p>
        </div>
        <Link
          to="/upload"
          className="inline-flex items-center justify-center rounded-md bg-teal px-4 py-2.5 text-sm font-medium text-void transition-opacity hover:opacity-90"
        >
          Upload X-ray
        </Link>
      </div>

      {error && <ErrorState error={error} onRetry={load} />}

      {!error && (
        <>
          <div className="mb-12 grid grid-cols-1 divide-y divide-line border border-line sm:grid-cols-3 sm:divide-x sm:divide-y-0">
            <StatCell label="Total analyses" value={stats.total} loading={loading} />
            <StatCell
              label="Normal"
              value={stats.normal}
              loading={loading}
              tone="teal"
            />
            <StatCell
              label="Abnormal"
              value={loading ? null : Math.max(abnormal, 0)}
              loading={loading}
              tone="alert"
            />
          </div>

          <div className="mb-4 flex items-center justify-between">
            <h2 className="text-sm font-medium text-ink">Recent analyses</h2>
            {health && (
              <span className="font-data text-xs text-ink-faint">
                {health.model_loaded
                  ? "models loaded"
                  : "running without loaded model weights"}
              </span>
            )}
          </div>

          {!loading && recent.length === 0 && (
            <div className="rounded-lg border border-dashed border-line px-8 py-14 text-center">
              <p className="text-sm text-ink-dim">
                No analyses yet — upload an X-ray to get started.
              </p>
            </div>
          )}

          <div className="divide-y divide-line-soft border-t border-line">
            {recent.map((p) => (
              <Link
                key={p.id}
                to={`/analysis/${p.id}`}
                className="flex items-center justify-between px-1 py-3.5 text-sm transition-colors hover:bg-panel"
              >
                <div className="flex items-center gap-3">
                  <span
                    className={`h-1.5 w-1.5 rounded-full ${
                      p.predicted_class === "Normal" ? "bg-teal" : "bg-alert"
                    }`}
                  />
                  <span className="text-ink">{p.predicted_class}</span>
                  <span className="font-data text-xs text-ink-faint">
                    #{p.id}
                  </span>
                </div>
                <div className="flex items-center gap-6">
                  <span className="font-data text-xs text-ink-dim">
                    {formatPercent(p.confidence)}
                  </span>
                  <span className="font-data text-xs text-ink-faint">
                    {formatDateTime(p.created_at)}
                  </span>
                </div>
              </Link>
            ))}
          </div>
        </>
      )}
    </div>
  );
}

function StatCell({ label, value, loading, tone }) {
  const color =
    tone === "teal" ? "text-teal" : tone === "alert" ? "text-alert" : "text-ink";
  return (
    <div className="px-6 py-6">
      <p className={`font-data text-3xl font-medium ${color}`}>
        {loading || value === null ? "—" : value}
      </p>
      <p className="mt-1 text-xs text-ink-dim">{label}</p>
    </div>
  );
}
