import { useEffect, useState } from "react";
import { useParams } from "react-router-dom";
import {
  getPrediction,
  getImage,
  createReport,
  fileUrl,
} from "../services/api";
import { formatDateTime, formatPercent } from "../lib/format";
import DetectionTable from "../components/DetectionTable";
import ErrorState from "../components/ErrorState";

const DISCLAIMER =
  "AI-generated preliminary analysis. This system is intended for research and educational purposes and does not replace interpretation by a qualified medical professional.";

export default function Report() {
  const { predictionId } = useParams();
  const [prediction, setPrediction] = useState(null);
  const [image, setImage] = useState(null);
  const [report, setReport] = useState(null);
  const [loading, setLoading] = useState(true);
  const [generating, setGenerating] = useState(false);
  const [error, setError] = useState(null);

  async function loadContext() {
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

  async function generate() {
    setGenerating(true);
    setError(null);
    try {
      const r = await createReport(predictionId);
      setReport(r);
    } catch (e) {
      setError(e);
    } finally {
      setGenerating(false);
    }
  }

  useEffect(() => {
    loadContext();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [predictionId]);

  if (loading) {
    return (
      <div className="mx-auto max-w-3xl px-6 py-16 text-sm text-ink-dim">
        Loading…
      </div>
    );
  }

  if (error && !prediction) {
    return (
      <div className="mx-auto max-w-3xl px-6 py-16">
        <ErrorState error={error} onRetry={loadContext} />
      </div>
    );
  }

  const isNormal = prediction.predicted_class === "Normal";
  const summary = buildSummary(prediction);

  return (
    <div className="mx-auto max-w-3xl px-6 py-10">
      <div className="no-print mb-8 flex items-center justify-between border-b border-line pb-6">
        <div>
          <p className="font-data text-xs text-ink-faint">
            Report · Analysis #{prediction.id}
          </p>
          <h1 className="mt-1 text-2xl font-medium tracking-tight text-ink">
            Preliminary report
          </h1>
        </div>
        <div className="flex gap-3">
          {!report && (
            <button
              onClick={generate}
              disabled={generating}
              className="rounded-md bg-teal px-4 py-2.5 text-sm font-medium text-void hover:opacity-90 disabled:opacity-50"
            >
              {generating ? "Generating…" : "Generate report"}
            </button>
          )}
          {report && (
            <button
              onClick={() => window.print()}
              className="rounded-md border border-line px-4 py-2.5 text-sm text-ink hover:border-teal"
            >
              Download / print
            </button>
          )}
        </div>
      </div>

      {error && (
        <div className="no-print mb-6">
          <ErrorState error={error} onRetry={generate} />
        </div>
      )}

      {!report && !error && (
        <p className="no-print text-sm text-ink-dim">
          Click "Generate report" to create a report record for this
          analysis.
        </p>
      )}

      {report && (
        <div className="rounded-lg border border-line bg-panel px-8 py-8 text-sm">
          <div className="mb-6 flex items-center justify-between border-b border-line pb-4">
            <span className="text-ink">Chest X-Ray AI Analysis Report</span>
            <span className="font-data text-xs text-ink-faint">
              Report #{report.id}
            </span>
          </div>

          <dl className="mb-6 grid grid-cols-2 gap-y-2">
            <ReportRow label="Patient">{image.patient_id ?? "Not assigned"}</ReportRow>
            <ReportRow label="Analysis ID">{prediction.id}</ReportRow>
            <ReportRow label="Analyzed">{formatDateTime(prediction.created_at)}</ReportRow>
            <ReportRow label="Generated">{formatDateTime(report.generated_at)}</ReportRow>
          </dl>

          <Section title="Primary finding">
            <p className="flex items-center gap-3">
              <span className="text-base text-ink">{prediction.predicted_class}</span>
              <span className="font-data text-ink-dim">
                {formatPercent(prediction.confidence)} confidence
              </span>
            </p>
          </Section>

          <Section title="Detected regions">
            {isNormal ? (
              <p className="text-ink-dim">No abnormal regions were detected.</p>
            ) : (
              <DetectionTable bboxes={prediction.bboxes} />
            )}
          </Section>

          <Section title="Summary">
            <p className="leading-relaxed text-ink-dim">{summary}</p>
          </Section>

          {prediction.heatmap_path && (
            <Section title="Grad-CAM visualization">
              <img
                src={fileUrl(prediction.heatmap_path)}
                alt="Grad-CAM heatmap"
                className="max-h-72 rounded border border-line-soft"
              />
            </Section>
          )}

          <div className="mt-6 border-t border-line pt-4 text-xs text-ink-faint">
            {DISCLAIMER}
          </div>
        </div>
      )}
    </div>
  );
}

function buildSummary(prediction) {
  const { predicted_class: cls, confidence, bboxes } = prediction;
  if (cls === "Normal") {
    return `The classifier's top finding was Normal, with ${formatPercent(
      confidence
    )} confidence. No localized abnormalities were flagged by the detector.`;
  }
  const regionText =
    bboxes.length === 0
      ? "no specific regions were localized by the detector"
      : `${bboxes.length} region${bboxes.length === 1 ? "" : "s"} ${
          bboxes.length === 1 ? "was" : "were"
        } flagged, most notably ${bboxes[0].class ?? bboxes[0].class_}`;
  return `The classifier's top finding was ${cls}, with ${formatPercent(
    confidence
  )} confidence. On detection, ${regionText}. This summary is generated directly from the structured model output, not an independent interpretation.`;
}

function Section({ title, children }) {
  return (
    <div className="mb-6">
      <h3 className="mb-2 text-xs font-medium text-ink-faint">{title}</h3>
      {children}
    </div>
  );
}

function ReportRow({ label, children }) {
  return (
    <>
      <dt className="text-ink-faint">{label}</dt>
      <dd className="text-right font-data text-ink">{children}</dd>
    </>
  );
}
