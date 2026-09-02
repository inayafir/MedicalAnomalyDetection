export default function ErrorState({ error, onRetry }) {
  const message =
    typeof error === "string" ? error : error?.message || "Something went wrong.";

  return (
    <div className="flex flex-col items-center gap-3 rounded-lg border border-alert-dim bg-alert-dim/30 px-8 py-10 text-center">
      <p className="text-sm font-medium text-ink">{message}</p>
      {onRetry && (
        <button
          onClick={onRetry}
          className="rounded-md border border-line px-3 py-1.5 text-xs text-ink-dim hover:border-teal hover:text-ink"
        >
          Try again
        </button>
      )}
    </div>
  );
}
