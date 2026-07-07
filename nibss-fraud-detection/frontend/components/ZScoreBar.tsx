import { featureLabel } from "@/lib/types";
import type { CandidateFeature } from "@/lib/discovery";

export function ZScoreBar({
  candidates,
  highlighted,
}: {
  candidates: CandidateFeature[];
  highlighted: string[];
}) {
  const maxAbsZ = Math.max(...candidates.map((c) => Math.abs(c.anomalyZ)), 0.01);

  return (
    <div className="flex flex-col gap-3">
      {candidates.map((c) => {
        const widthPct = (Math.abs(c.anomalyZ) / maxAbsZ) * 100;
        const isPositive = c.anomalyZ >= 0;
        const isHighlighted = highlighted.includes(c.feature);
        return (
          <div key={c.feature} className="flex flex-col gap-1.5">
            <div className="flex items-center justify-between text-sm">
              <span
                className={
                  isHighlighted ? "font-medium text-gradient" : "text-[--foreground]"
                }
              >
                {featureLabel(c.feature)}
                {isHighlighted && (
                  <span className="ml-2 rounded-full bg-white/10 px-2 py-0.5 text-[10px] uppercase tracking-wide text-[--muted]">
                    LLM pick
                  </span>
                )}
              </span>
              <span className="font-mono text-xs text-[--muted]">
                z={isPositive ? "+" : ""}
                {c.anomalyZ.toFixed(2)}
              </span>
            </div>
            <div className="h-1.5 w-full overflow-hidden rounded-full bg-white/5">
              <div
                className={`h-full rounded-full ${
                  isHighlighted
                    ? "bg-gradient-to-r from-[--accent-violet] to-[--accent-blue]"
                    : "bg-white/20"
                }`}
                style={{ width: `${Math.max(widthPct, 2)}%` }}
              />
            </div>
          </div>
        );
      })}
      <p className="mt-1 text-xs text-[--muted-2]">
        |z| = how many standard deviations the missed-fraud average sits from
        the legitimate-transaction baseline, per feature.
      </p>
    </div>
  );
}
