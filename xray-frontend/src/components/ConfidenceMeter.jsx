import { formatPercent } from "../lib/format";

export default function ConfidenceMeter({ value, tone = "teal" }) {
  const pct = Math.round((value || 0) * 100);
  const barColor =
    tone === "alert" ? "bg-alert" : tone === "amber" ? "bg-amber" : "bg-teal";

  return (
    <div className="flex items-center gap-3">
      <div className="h-1.5 w-28 overflow-hidden rounded-full bg-panel-raised">
        <div
          className={`h-full rounded-full ${barColor}`}
          style={{ width: `${pct}%` }}
        />
      </div>
      <span className="font-data text-sm text-ink">{formatPercent(value)}</span>
    </div>
  );
}
