import { colorForClass } from "../lib/format";

export default function DetectionTable({ bboxes = [] }) {
  if (bboxes.length === 0) {
    return (
      <p className="text-sm text-ink-dim">
        No localized findings were returned for this image.
      </p>
    );
  }

  return (
    <div className="overflow-hidden rounded-lg border border-line">
      <table className="w-full text-left text-sm">
        <thead>
          <tr className="border-b border-line text-xs text-ink-faint">
            <th className="px-3 py-2 font-normal">Finding</th>
            <th className="px-3 py-2 font-normal">Confidence</th>
            <th className="px-3 py-2 font-normal font-data">X1</th>
            <th className="px-3 py-2 font-normal font-data">Y1</th>
            <th className="px-3 py-2 font-normal font-data">X2</th>
            <th className="px-3 py-2 font-normal font-data">Y2</th>
          </tr>
        </thead>
        <tbody>
          {bboxes.map((box, i) => {
            const cls = box.class ?? box.class_ ?? "Finding";
            return (
              <tr
                key={i}
                className={i > 0 ? "border-t border-line-soft" : ""}
              >
                <td className="px-3 py-2">
                  <span className="flex items-center gap-2">
                    <span
                      className="h-2 w-2 rounded-full"
                      style={{ background: colorForClass(cls) }}
                    />
                    {cls}
                  </span>
                </td>
                <td className="px-3 py-2 font-data">
                  {Math.round((box.confidence || 0) * 100)}%
                </td>
                <td className="px-3 py-2 font-data text-ink-dim">{box.x1}</td>
                <td className="px-3 py-2 font-data text-ink-dim">{box.y1}</td>
                <td className="px-3 py-2 font-data text-ink-dim">{box.x2}</td>
                <td className="px-3 py-2 font-data text-ink-dim">{box.y2}</td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}
