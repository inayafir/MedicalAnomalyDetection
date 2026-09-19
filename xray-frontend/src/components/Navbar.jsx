import { NavLink } from "react-router-dom";
import HealthBadge from "./HealthBadge";

const links = [
  { to: "/", label: "Dashboard", end: true },
  { to: "/upload", label: "Upload" },
  { to: "/studies", label: "Studies" },
];

export default function Navbar() {
  return (
    <header className="sticky top-0 z-20 border-b border-line bg-void/95 backdrop-blur">
      <div className="mx-auto flex max-w-6xl items-center gap-8 px-6 py-4">
        <NavLink to="/" className="flex items-center gap-2.5">
          <svg width="22" height="22" viewBox="0 0 22 22" fill="none">
            <circle cx="11" cy="11" r="9" stroke="#3fb6ad" strokeWidth="1.4" />
            <circle cx="11" cy="11" r="3" fill="#3fb6ad" />
          </svg>
          <span className="text-[17px] font-medium tracking-tight text-ink">
            Lumen
          </span>
        </NavLink>

        <nav className="flex items-center gap-6 text-sm">
          {links.map((l) => (
            <NavLink
              key={l.to}
              to={l.to}
              end={l.end}
              className={({ isActive }) =>
                `border-b-2 py-1 transition-colors ${
                  isActive
                    ? "border-teal text-ink"
                    : "border-transparent text-ink-dim hover:text-ink"
                }`
              }
            >
              {l.label}
            </NavLink>
          ))}
        </nav>

        <div className="ml-auto">
          <HealthBadge />
        </div>
      </div>
    </header>
  );
}
