const STAGES = [
  { key: "uploading", label: "Uploading image" },
  { key: "analyzing", label: "Running classification and detection" },
  { key: "done", label: "Preparing results" },
];

export default function ProcessingStages({ current }) {
  const currentIndex = STAGES.findIndex((s) => s.key === current);

  return (
    <div className="flex flex-col gap-4 rounded-lg border border-line bg-panel px-6 py-8">
      {STAGES.map((stage, i) => {
        const state =
          i < currentIndex ? "done" : i === currentIndex ? "active" : "pending";
        return (
          <div key={stage.key} className="flex items-center gap-3">
            <span
              className={`flex h-5 w-5 items-center justify-center rounded-full border text-[10px] font-data ${
                state === "done"
                  ? "border-teal bg-teal text-void"
                  : state === "active"
                  ? "border-teal text-teal animate-pulse"
                  : "border-line text-ink-faint"
              }`}
            >
              {state === "done" ? "✓" : i + 1}
            </span>
            <span
              className={`text-sm ${
                state === "pending" ? "text-ink-faint" : "text-ink"
              }`}
            >
              {stage.label}
            </span>
          </div>
        );
      })}
    </div>
  );
}
