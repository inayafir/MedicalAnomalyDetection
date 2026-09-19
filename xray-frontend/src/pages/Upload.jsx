import { useState } from "react";
import { useNavigate } from "react-router-dom";
import UploadDropzone from "../components/UploadDropzone";
import ProcessingStages from "../components/ProcessingStages";
import ErrorState from "../components/ErrorState";
import { uploadImage, createPrediction } from "../services/api";

export default function Upload() {
  const navigate = useNavigate();
  const [file, setFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [patientId, setPatientId] = useState("");
  const [stage, setStage] = useState(null); // null | uploading | analyzing
  const [error, setError] = useState(null);

  function handleFile(f) {
    setFile(f);
    setError(null);
    setPreviewUrl(URL.createObjectURL(f));
  }

  function reset() {
    setFile(null);
    setPreviewUrl(null);
    setError(null);
    setStage(null);
  }

  async function runAnalysis() {
    setError(null);
    setStage("uploading");
    try {
      const image = await uploadImage(file, patientId ? Number(patientId) : undefined);
      setStage("analyzing");
      const prediction = await createPrediction(image.id);
      setStage("done");
      navigate(`/analysis/${prediction.id}`);
    } catch (e) {
      setStage(null);
      setError(
        e.status === 503
          ? {
              message:
                "The model weights aren't loaded on the backend yet, so analysis can't run. This is expected until trained checkpoints are in place — check GET /health for model status.",
            }
          : e
      );
    }
  }

  return (
    <div className="mx-auto max-w-3xl px-6 py-12">
      <h1 className="text-2xl font-medium tracking-tight text-ink">
        Upload an X-ray
      </h1>
      <p className="mt-2 text-sm text-ink-dim">
        Accepts PNG or JPEG, up to 10 MB. The image is analyzed immediately
        after upload.
      </p>

      <div className="mt-8">
        {!file && <UploadDropzone onFileSelected={handleFile} />}

        {file && !stage && (
          <div className="flex flex-col gap-6">
            <div className="overflow-hidden rounded-lg border border-line bg-panel">
              <img
                src={previewUrl}
                alt="Preview"
                className="mx-auto block max-h-96 w-auto"
              />
            </div>

            <div className="flex items-end justify-between gap-4">
              <label className="flex flex-col gap-1.5 text-sm">
                <span className="text-ink-dim">Patient ID (optional)</span>
                <input
                  type="number"
                  min="1"
                  value={patientId}
                  onChange={(e) => setPatientId(e.target.value)}
                  placeholder="Leave blank if none"
                  className="w-48 rounded-md border border-line bg-panel px-3 py-2 font-data text-sm text-ink placeholder:text-ink-faint focus:border-teal"
                />
              </label>

              <div className="flex gap-3">
                <button
                  onClick={reset}
                  className="rounded-md border border-line px-4 py-2.5 text-sm text-ink-dim hover:text-ink"
                >
                  Choose different file
                </button>
                <button
                  onClick={runAnalysis}
                  className="rounded-md bg-teal px-5 py-2.5 text-sm font-medium text-void hover:opacity-90"
                >
                  Run analysis
                </button>
              </div>
            </div>
          </div>
        )}

        {stage && stage !== "done" && <ProcessingStages current={stage} />}

        {error && (
          <div className="mt-6">
            <ErrorState error={error} onRetry={runAnalysis} />
          </div>
        )}
      </div>
    </div>
  );
}
