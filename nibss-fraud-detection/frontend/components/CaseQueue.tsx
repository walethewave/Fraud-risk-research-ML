"use client";

import Link from "next/link";
import { useMemo, useState } from "react";
import { RiskBadge } from "./RiskBadge";
import { CaseTypeBadge } from "./CaseTypeBadge";
import { formatNaira, riskLevel, type FraudCase } from "@/lib/types";

type Filter = "all" | "true_fraud" | "false_positive";

export function CaseQueue({ cases }: { cases: FraudCase[] }) {
  const [filter, setFilter] = useState<Filter>("all");

  const filtered = useMemo(() => {
    if (filter === "all") return cases;
    return cases.filter((c) => c.caseType === filter);
  }, [cases, filter]);

  const counts = useMemo(
    () => ({
      all: cases.length,
      true_fraud: cases.filter((c) => c.caseType === "true_fraud").length,
      false_positive: cases.filter((c) => c.caseType === "false_positive").length,
    }),
    [cases]
  );

  return (
    <div className="flex flex-col gap-4">
      <div className="flex gap-2">
        {(
          [
            ["all", "All cases"],
            ["true_fraud", "Actual fraud"],
            ["false_positive", "False positives"],
          ] as [Filter, string][]
        ).map(([key, label]) => (
          <button
            key={key}
            onClick={() => setFilter(key)}
            className={`rounded-full px-3.5 py-1.5 text-sm transition-colors ${
              filter === key
                ? "glass-raised text-[--foreground]"
                : "text-[--muted] hover:text-[--foreground]"
            }`}
          >
            {label}
            <span className="ml-1.5 text-[--muted-2]">{counts[key]}</span>
          </button>
        ))}
      </div>

      <div className="glass overflow-hidden rounded-2xl">
        <table className="w-full text-sm">
          <thead>
            <tr className="border-b border-[--border-subtle] text-left text-xs uppercase tracking-wide text-[--muted-2]">
              <th className="px-5 py-3 font-medium">Transaction</th>
              <th className="px-5 py-3 font-medium">Channel / Bank</th>
              <th className="px-5 py-3 font-medium">Confidence</th>
              <th className="px-5 py-3 font-medium">Risk</th>
              <th className="px-5 py-3 font-medium">Outcome</th>
            </tr>
          </thead>
          <tbody>
            {filtered.map((c) => (
              <tr key={c.id} className="group border-b border-[--border-subtle] last:border-0">
                <td className="px-5 py-4">
                  <Link href={`/case/${c.id}`} className="block">
                    <div className="font-mono text-[--foreground] group-hover:text-gradient">
                      {formatNaira(c.amount)}
                    </div>
                    <div className="mt-0.5 text-xs text-[--muted-2]">
                      {c.location} &middot; {c.ageGroup}
                    </div>
                  </Link>
                </td>
                <td className="px-5 py-4">
                  <div className="text-[--foreground]">{c.channel}</div>
                  <div className="mt-0.5 text-xs text-[--muted-2]">{c.bank}</div>
                </td>
                <td className="px-5 py-4 font-mono text-[--foreground]">
                  {(c.fraudProbability * 100).toFixed(0)}%
                </td>
                <td className="px-5 py-4">
                  <RiskBadge level={riskLevel(c.fraudProbability)} />
                </td>
                <td className="px-5 py-4">
                  <CaseTypeBadge caseType={c.caseType} />
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
