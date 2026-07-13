import { featureLabel, type ShapFeature } from "@/lib/types";

export function ShapBar({ features }: { features: ShapFeature[] }) {
  const maxAbs = Math.max(...features.map((f) => Math.abs(f.value)), 0.001);

  return (
    <div className="flex flex-col gap-3">
      {features.map((f) => {
        const widthPct = (Math.abs(f.value) / maxAbs) * 100;
        const isPositive = f.value >= 0;
        return (
          <div key={f.feature} className="flex flex-col gap-1.5">
            <div className="flex items-center justify-between text-sm">
              <span className="text-[--foreground]">{featureLabel(f.feature)}</span>
              <span
                className={`font-mono text-xs ${
                  isPositive ? "text-[--risk-high]" : "text-[--risk-low]"
                }`}
              >
                {isPositive ? "+" : ""}
                {f.value.toFixed(3)}
              </span>
            </div>
            <div className="h-1.5 w-full overflow-hidden rounded-full bg-white/5">
              <div
                className={`h-full rounded-full ${
                  isPositive
                    ? "bg-gradient-to-r from-[--risk-high]/60 to-[--risk-high]"
                    : "bg-gradient-to-r from-[--risk-low] to-[--risk-low]/60"
                }`}
                style={{ width: `${widthPct}%` }}
              />
            </div>
          </div>
        );
      })}
      <p className="mt-1 text-xs text-[--muted-2]">
        Positive values pushed the prediction toward fraud; negative values reduced risk.
      </p>
    </div>
  );
}
