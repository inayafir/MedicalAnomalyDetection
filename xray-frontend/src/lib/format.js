export function formatPercent(value) {
  if (value === null || value === undefined) return "—";
  return `${(value * 100).toFixed(1)}%`;
}

export function formatDateTime(iso) {
  if (!iso) return "—";
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return iso;
  return d.toLocaleString(undefined, {
    year: "numeric",
    month: "short",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  });
}

export function formatBytes(bytes) {
  if (!bytes && bytes !== 0) return "—";
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(2)} MB`;
}

// The classifier's whole-image label — 5 classes, sourced from the
// ResNet-50 checkpoint at load time (backend/app/models.py).
export const CLASSIFIER_CLASSES = [
  "Normal",
  "Cardiomegaly",
  "Pleural effusion",
  "Lung Opacity",
  "Pulmonary fibrosis",
];

// Per-box detector labels — 14 disease classes, no "Normal" (YOLOv8m).
export const DETECTOR_CLASSES = [
  "Aortic enlargement",
  "Atelectasis",
  "Calcification",
  "Cardiomegaly",
  "Consolidation",
  "ILD",
  "Infiltration",
  "Lung Opacity",
  "Nodule/Mass",
  "Other lesion",
  "Pleural effusion",
  "Pleural thickening",
  "Pneumothorax",
  "Pulmonary fibrosis",
];

const BOX_PALETTE = [
  "#3fb6ad",
  "#d6a349",
  "#d1543f",
  "#7a9cc6",
  "#a97fc9",
  "#6fbf8b",
  "#c97fa0",
  "#9aa6b2",
];

export function colorForClass(className) {
  let hash = 0;
  for (let i = 0; i < className.length; i += 1) {
    hash = (hash * 31 + className.charCodeAt(i)) >>> 0;
  }
  return BOX_PALETTE[hash % BOX_PALETTE.length];
}
