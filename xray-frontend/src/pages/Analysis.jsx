import { useEffect, useState } from "react";
import { Link, useNavigate, useParams } from "react-router-dom";
import { getPrediction, getImage, fileUrl } from "../services/api";
import { formatBytes, formatDateTime, formatPercent } from "../lib/format";
import XRayViewer from "../components/XRayViewer";
import DetectionTable from "../components/DetectionTable";
import ConfidenceMeter from "../components/ConfidenceMeter";
import ErrorState from "../components/ErrorState";

export default function Analysis() {
  const { predictionId } = useParams();
  const navigate = useNavigate();
  const [prediction, setPrediction] = useState(null);
  const [image, setImage] = useState(null);
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(true);

  async function load() {
    setLoading(true);
    setError(null);
    try {
      const pred = await getPrediction(predictionId);
      setPrediction(pred);
      const img = await getImage(pred.image_id);
      setImage(img);
    } catch (e) {
      setError(e);
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    load();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [predictionId]);

  if (loading) {
    return (
      <div className="mx-auto max-w-6xl px-6 py-16 text-sm text-ink-dim">
        Loading analysis…
      </div>
    );
  }

  if (error) {
    return (
      <div className="mx-auto max-w-3xl px-6 py-16">
        <ErrorState error={error} onRetry={load} />
      </div>
    );
  }

  const isNormal = prediction.predicted_class === "Normal";

  return (
    <div className="mx-auto max-w-6xl px-6 py-10">
      <div className="mb-8 flex flex-wrap items-start justify-between gap-4 border-b border-line pb-6">
        <div>
          <p className="font-data text-xs text-ink-faint">
            Analysis #{prediction.id}
          </p>
          <h1 className="mt-1 flex items-center gap-3 text-2xl font-medium tracking-tight text-ink">
            {prediction.predicted_class}
            {isNormal ? (
              <span className="rounded-full bg-teal-dim px-2.5 py-0.5 text-xs font-normal text-teal">
                No abnormality
              </span>
            ) : (
              <span className="rounded-full bg-alert-dim px-2.5 py-0.5 text-xs font-normal text-alert">
                Abnormal
              </span>
            )}
          </h1>
        </div>
        <button
          onClick={() => navigate(`/report/${prediction.id}`)}
          className="rounded-md bg-teal px-4 py-2.5 text-sm font-medium text-void hover:opacity-90"
        >
          Generate report
        </button>
      </div>

      <div className="grid grid-cols-1 gap-8 lg:grid-cols-[1.6fr_1fr]">
        <div>
          <XRayViewer
            imageSrc={fileUrl(image.file_path)}
            heatmapSrc={prediction.heatmap_path ? fileUrl(prediction.heatmap_path) : null}
            bboxes={prediction.bboxes}
          />

          <div className="mt-8">
            <h2 className="mb-3 text-sm font-medium text-ink">
              Detected regions
            </h2>
            {isNormal ? (
              <p className="text-sm text-ink-dim">
                No abnormal bounding boxes were returned — the classifier's
                top finding is Normal.
              </p>
            ) : (
              <DetectionTable bboxes={prediction.bboxes} />
            )}
          </div>
        </div>

        <div className="flex flex-col gap-6">
          <Panel title="Classification">
            <Row label="Finding">{prediction.predicted_class}</Row>
            <div className="flex items-center justify-between py-2">
              <span className="text-sm text-ink-dim">Confidence</span>
              <ConfidenceMeter
                value={prediction.confidence}
                tone={isNormal ? "teal" : "alert"}
              />
            </div>
          </Panel>

          <Panel title="Image metadata">
            <Row label="Filename">{image.original_filename}</Row>
            <Row label="Content type" mono>{image.content_type}</Row>
            <Row label="Size" mono>{formatBytes(image.file_size_bytes)}</Row>
            <Row label="Uploaded" mono>{formatDateTime(image.uploaded_at)}</Row>
            <Row label="Patient" mono>{image.patient_id ?? "—"}</Row>
          </Panel>

          <Panel title="Analysis">
            <Row label="Analyzed" mono>{formatDateTime(prediction.created_at)}</Row>
            <Row label="Findings" mono>{prediction.bboxes.length}</Row>
          </Panel>

          <Link
            to="/upload"
            className="text-center text-sm text-ink-dim underline underline-offset-2 hover:text-ink"
          >
            Analyze another X-ray
          </Link>
        </div>
      </div>
    </div>
  );
}

function Panel({ title, children }) {
  return (
    <div className="rounded-lg border border-line bg-panel px-5 py-4">
      <h3 className="mb-2 text-xs font-medium text-ink-faint">{title}</h3>
      <div className="divide-y divide-line-soft">{children}</div>
    </div>
  );
}

function Row({ label, children, mono }) {
  return (
    <div className="flex items-center justify-between py-2 text-sm">
      <span className="text-ink-dim">{label}</span>
      <span className={mono ? "font-data text-ink" : "text-ink"}>{children}</span>
    </div>
  );
}
