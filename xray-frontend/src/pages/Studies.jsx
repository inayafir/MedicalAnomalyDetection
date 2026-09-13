import { useEffect, useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import { listImages, getImage, createPrediction, fileUrl } from "../services/api";
import { formatBytes, formatDateTime } from "../lib/format";
import ErrorState from "../components/ErrorState";
import EmptyState from "../components/EmptyState";

export default function Studies() {
  const [items, setItems] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [opening, setOpening] = useState(null);
  const navigate = useNavigate();

  async function load() {
    setLoading(true);
    setError(null);
    try {
      const res = await listImages({ limit: 50 });
      setItems(res.items);
    } catch (e) {
      setError(e);
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    load();
  }, []);

  async function openStudy(imageId) {
    setOpening(imageId);
    try {
      const detail = await getImage(imageId);
      if (detail.latest_prediction) {
        navigate(`/analysis/${detail.latest_prediction.id}`);
        return;
      }
      const prediction = await createPrediction(imageId);
      navigate(`/analysis/${prediction.id}`);
    } catch (e) {
      setError(e);
    } finally {
      setOpening(null);
    }
  }

  return (
    <div className="mx-auto max-w-6xl px-6 py-10">
      <div className="mb-8 flex items-center justify-between border-b border-line pb-6">
        <h1 className="text-2xl font-medium tracking-tight text-ink">
          Studies
        </h1>
        <Link
          to="/upload"
          className="rounded-md border border-line px-4 py-2 text-sm text-ink hover:border-teal"
        >
          Upload new
        </Link>
      </div>

      {error && <ErrorState error={error} onRetry={load} />}

      {!error && !loading && items.length === 0 && (
        <EmptyState
          title="No studies uploaded yet"
          description="Uploaded X-rays will appear here once analyzed."
        />
      )}

      {!loading && items.length > 0 && (
        <div className="grid grid-cols-2 gap-4 sm:grid-cols-3 lg:grid-cols-4">
          {items.map((img) => (
            <button
              key={img.id}
              onClick={() => openStudy(img.id)}
              disabled={opening === img.id}
              className="group overflow-hidden rounded-lg border border-line bg-panel text-left disabled:opacity-60"
            >
              <div className="aspect-square overflow-hidden bg-void">
                <img
                  src={fileUrl(img.file_path)}
                  alt={img.original_filename}
                  className="h-full w-full object-cover opacity-90 transition-opacity group-hover:opacity-100"
                />
              </div>
              <div className="px-3 py-2.5">
                <p className="truncate text-xs text-ink">
                  {opening === img.id ? "Opening…" : img.original_filename}
                </p>
                <p className="mt-0.5 font-data text-[11px] text-ink-faint">
                  {formatBytes(img.file_size_bytes)} · {formatDateTime(img.uploaded_at)}
                </p>
              </div>
            </button>
          ))}
        </div>
      )}
    </div>
  );
}
