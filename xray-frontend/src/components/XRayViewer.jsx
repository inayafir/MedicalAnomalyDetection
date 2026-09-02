import { useState } from "react";
import { colorForClass } from "../lib/format";

export default function XRayViewer({ imageSrc, heatmapSrc, bboxes = [] }) {
  const [mode, setMode] = useState("boxes"); // raw | boxes | heat
  const [naturalSize, setNaturalSize] = useState(null);

  const modes = [
    { key: "raw", label: "Original" },
    { key: "boxes", label: `Detections (${bboxes.length})`, disabled: bboxes.length === 0 },
    { key: "heat", label: "Grad-CAM", disabled: !heatmapSrc },
  ];

  return (
    <div className="overflow-hidden rounded-lg border border-line bg-panel">
      <div className="flex items-center justify-between border-b border-line px-4 py-2.5">
        <div className="flex gap-1">
          {modes.map((m) => (
            <button
              key={m.key}
              disabled={m.disabled}
              onClick={() => setMode(m.key)}
              className={`rounded px-2.5 py-1 font-data text-xs transition-colors ${
                mode === m.key
                  ? "bg-teal-dim text-teal"
                  : m.disabled
                  ? "cursor-not-allowed text-ink-faint"
                  : "text-ink-dim hover:text-ink"
              }`}
            >
              {m.label}
            </button>
          ))}
        </div>
      </div>

      {/* backlight viewer */}
      <div
        className="relative flex items-center justify-center px-6 py-10"
        style={{
          background:
            "radial-gradient(ellipse at center, rgba(207,231,238,0.06) 0%, rgba(207,231,238,0.0) 60%)",
        }}
      >
        <div className="relative inline-block max-w-full border border-line-soft shadow-[0_0_60px_-10px_rgba(207,231,238,0.08)]">
          <img
            src={mode === "heat" && heatmapSrc ? heatmapSrc : imageSrc}
            alt="Chest X-ray"
            className="block max-h-[560px] w-auto max-w-full select-none"
            onLoad={(e) =>
              setNaturalSize({
                w: e.currentTarget.naturalWidth,
                h: e.currentTarget.naturalHeight,
              })
            }
          />
          {mode === "boxes" &&
            naturalSize &&
            bboxes.map((box, i) => {
              const cls = box.class ?? box.class_ ?? "Finding";
              const color = colorForClass(cls);
              const left = (box.x1 / naturalSize.w) * 100;
              const top = (box.y1 / naturalSize.h) * 100;
              const width = ((box.x2 - box.x1) / naturalSize.w) * 100;
              const height = ((box.y2 - box.y1) / naturalSize.h) * 100;
              return (
                <div
                  key={i}
                  className="absolute border-2"
                  style={{
                    left: `${left}%`,
                    top: `${top}%`,
                    width: `${width}%`,
                    height: `${height}%`,
                    borderColor: color,
                  }}
                >
                  <span
                    className="absolute -top-5 left-0 whitespace-nowrap px-1 font-data text-[10px] text-void"
                    style={{ background: color }}
                  >
                    {cls} {Math.round((box.confidence || 0) * 100)}%
                  </span>
                </div>
              );
            })}
        </div>
      </div>
    </div>
  );
}
