import type { DiscoveryData } from "@/lib/discovery";

export function AucComparison({ impact }: { impact: DiscoveryData["impact"] }) {
  const maxAuc = 1;
  const baselinePct = (impact.baselineAuc / maxAuc) * 100;
  const augmentedPct = (impact.augmentedAuc / maxAuc) * 100;

  return (
    <div className="flex flex-col gap-6">
      <div className="flex flex-col gap-4">
        <AucRow label="Baseline (17 features)" value={impact.baselineAuc} pct={baselinePct} />
        <AucRow
          label={`+ ${impact.candidateFeatures.length} candidate feature(s)`}
          value={impact.augmentedAuc}
          pct={augmentedPct}
          gradient
        />
      </div>

      <div className="grid grid-cols-3 gap-3">
        <MiniStat label="ΔAUC" value={`${impact.aucDelta >= 0 ? "+" : ""}${impact.aucDelta.toFixed(4)}`} />
        <MiniStat
          label="Newly caught"
          value={`+${impact.nNewlyCaught}`}
          tone="positive"
        />
        <MiniStat
          label="Newly missed"
          value={`-${impact.nNewlyMissed}`}
          tone={impact.nNewlyMissed > 0 ? "negative" : undefined}
        />
      </div>
    </div>
  );
}

function AucRow({
  label,
  value,
  pct,
  gradient,
}: {
  label: string;
  value: number;
  pct: number;
  gradient?: boolean;
}) {
  return (
    <div className="flex flex-col gap-1.5">
      <div className="flex items-center justify-between text-sm">
        <span className="text-[--foreground]">{label}</span>
        <span className="font-mono text-[--foreground]">{value.toFixed(4)}</span>
      </div>
      <div className="h-2.5 w-full overflow-hidden rounded-full bg-white/5">
        <div
          className={`h-full rounded-full ${
            gradient
              ? "bg-gradient-to-r from-[--accent-violet] to-[--accent-blue]"
              : "bg-white/25"
          }`}
          style={{ width: `${pct}%` }}
        />
      </div>
    </div>
  );
}

function MiniStat({
  label,
  value,
  tone,
}: {
  label: string;
  value: string;
  tone?: "positive" | "negative";
}) {
  const color =
    tone === "positive"
      ? "text-[--risk-low]"
      : tone === "negative"
        ? "text-[--risk-high]"
        : "text-[--foreground]";
  return (
    <div className="glass rounded-xl px-3 py-2.5">
      <div className="text-[10px] uppercase tracking-wide text-[--muted-2]">{label}</div>
      <div className={`font-mono text-lg ${color}`}>{value}</div>
    </div>
  );
}
