import { useEffect, useState } from "react";
import { getHealth } from "../services/api";

export default function HealthBadge() {
  const [state, setState] = useState({ status: "checking" });

  useEffect(() => {
    let cancelled = false;
    async function check() {
      try {
        const h = await getHealth();
        if (!cancelled) setState({ status: "ok", ...h });
      } catch {
        if (!cancelled) setState({ status: "down" });
      }
    }
    check();
    const id = setInterval(check, 20000);
    return () => {
      cancelled = true;
      clearInterval(id);
    };
  }, []);

  const dotColor =
    state.status === "ok" && state.model_loaded
      ? "bg-teal"
      : state.status === "ok"
      ? "bg-amber"
      : "bg-alert";

  const label =
    state.status === "checking"
      ? "Checking backend"
      : state.status === "down"
      ? "Backend unreachable"
      : state.model_loaded
      ? "Models loaded"
      : "Models not loaded";

  return (
    <div className="flex items-center gap-2 rounded-full border border-line bg-panel px-3 py-1.5 text-xs text-ink-dim">
      <span className={`h-1.5 w-1.5 rounded-full ${dotColor}`} />
      <span className="font-data">{label}</span>
    </div>
  );
}
