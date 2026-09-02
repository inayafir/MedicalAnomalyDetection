import { useCallback, useRef, useState } from "react";

const ACCEPTED_TYPES = ["image/png", "image/jpeg"];
const MAX_SIZE_MB = 10;

export default function UploadDropzone({ onFileSelected }) {
  const [dragging, setDragging] = useState(false);
  const [validationError, setValidationError] = useState(null);
  const inputRef = useRef(null);

  const validateAndEmit = useCallback(
    (file) => {
      if (!file) return;
      if (!ACCEPTED_TYPES.includes(file.type)) {
        setValidationError("Only PNG or JPEG images are accepted.");
        return;
      }
      if (file.size > MAX_SIZE_MB * 1024 * 1024) {
        setValidationError(`File exceeds the ${MAX_SIZE_MB} MB limit.`);
        return;
      }
      setValidationError(null);
      onFileSelected(file);
    },
    [onFileSelected]
  );

  return (
    <div>
      <div
        onDragOver={(e) => {
          e.preventDefault();
          setDragging(true);
        }}
        onDragLeave={() => setDragging(false)}
        onDrop={(e) => {
          e.preventDefault();
          setDragging(false);
          validateAndEmit(e.dataTransfer.files?.[0]);
        }}
        className={`flex flex-col items-center justify-center gap-3 rounded-lg border px-8 py-16 text-center transition-colors ${
          dragging
            ? "border-teal bg-teal-dim/20"
            : "border-dashed border-line hover:border-ink-faint"
        }`}
      >
        <svg width="34" height="34" viewBox="0 0 24 24" fill="none">
          <path
            d="M12 15V4M12 4L7.5 8.5M12 4l4.5 4.5"
            stroke="#3fb6ad"
            strokeWidth="1.5"
            strokeLinecap="round"
            strokeLinejoin="round"
          />
          <path
            d="M4 15v3a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2v-3"
            stroke="#5d6d78"
            strokeWidth="1.5"
            strokeLinecap="round"
          />
        </svg>
        <p className="text-sm text-ink">
          Drag a chest X-ray here, or{" "}
          <button
            type="button"
            onClick={() => inputRef.current?.click()}
            className="text-teal underline underline-offset-2"
          >
            browse files
          </button>
        </p>
        <p className="text-xs text-ink-faint">PNG or JPEG, up to {MAX_SIZE_MB} MB</p>
        <input
          ref={inputRef}
          type="file"
          accept="image/png,image/jpeg"
          className="hidden"
          onChange={(e) => validateAndEmit(e.target.files?.[0])}
        />
      </div>
      {validationError && (
        <p className="mt-3 text-sm text-alert">{validationError}</p>
      )}
    </div>
  );
}
