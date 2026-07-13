import type { RiskLevel } from "@/lib/types";

const STYLES: Record<RiskLevel, string> = {
  High: "text-[--risk-high] bg-[--risk-high-bg] ring-1 ring-inset ring-[--risk-high]/30",
  Medium: "text-[--risk-medium] bg-[--risk-medium-bg] ring-1 ring-inset ring-[--risk-medium]/30",
  Low: "text-[--risk-low] bg-[--risk-low-bg] ring-1 ring-inset ring-[--risk-low]/30",
};

export function RiskBadge({ level }: { level: RiskLevel }) {
  return (
    <span
      className={`inline-flex items-center gap-1.5 rounded-full px-2.5 py-1 text-xs font-medium ${STYLES[level]}`}
    >
      <span className="h-1.5 w-1.5 rounded-full bg-current" />
      {level} risk
    </span>
  );
}
