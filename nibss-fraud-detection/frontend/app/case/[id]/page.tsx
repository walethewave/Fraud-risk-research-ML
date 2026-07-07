import Link from "next/link";
import { notFound } from "next/navigation";
import { RiskBadge } from "@/components/RiskBadge";
import { CaseTypeBadge } from "@/components/CaseTypeBadge";
import { ShapBar } from "@/components/ShapBar";
import { ChatPanel } from "@/components/ChatPanel";
import { getCaseById } from "@/lib/cases";
import { formatHour, formatNaira, riskLevel } from "@/lib/types";

export default async function CaseDetailPage({
  params,
}: {
  params: Promise<{ id: string }>;
}) {
  const { id } = await params;
  const fraudCase = getCaseById(Number(id));
  if (!fraudCase) notFound();

  return (
    <main className="mx-auto flex w-full max-w-5xl flex-1 flex-col gap-6 px-6 py-12 sm:px-10">
      <Link href="/" className="flex items-center gap-1.5 text-sm text-[--muted] hover:text-[--foreground]">
        <span aria-hidden>&larr;</span> Back to queue
      </Link>

      <div className="flex flex-wrap items-center justify-between gap-4">
        <div>
          <h1 className="font-mono text-3xl text-[--foreground]">{formatNaira(fraudCase.amount)}</h1>
          <p className="mt-1 text-sm text-[--muted]">
            {fraudCase.channel} &middot; {fraudCase.bank} &middot; {formatHour(fraudCase.hour)}
          </p>
        </div>
        <div className="flex gap-2">
          <RiskBadge level={riskLevel(fraudCase.fraudProbability)} />
          <CaseTypeBadge caseType={fraudCase.caseType} />
        </div>
      </div>

      <div className="grid grid-cols-1 gap-6 lg:grid-cols-5">
        <div className="flex flex-col gap-6 lg:col-span-2">
          <section className="glass rounded-2xl p-5">
            <h2 className="mb-4 text-sm font-medium text-[--muted]">Transaction details</h2>
            <dl className="grid grid-cols-2 gap-y-3 text-sm">
              <DetailRow label="Amount" value={formatNaira(fraudCase.amount)} />
              <DetailRow label="Channel" value={fraudCase.channel} />
              <DetailRow label="Hour" value={formatHour(fraudCase.hour)} />
              <DetailRow label="Bank" value={fraudCase.bank} />
              <DetailRow label="Location" value={fraudCase.location} />
              <DetailRow label="Age group" value={fraudCase.ageGroup} />
            </dl>
          </section>

          <section className="glass rounded-2xl p-5">
            <h2 className="mb-4 text-sm font-medium text-[--muted]">Model output</h2>
            <dl className="grid grid-cols-2 gap-y-3 text-sm">
              <DetailRow
                label="Prediction"
                value={fraudCase.predictedIsFraud ? "FRAUD" : "NOT FRAUD"}
              />
              <DetailRow
                label="Confidence"
                value={`${(fraudCase.fraudProbability * 100).toFixed(0)}%`}
              />
              <DetailRow
                label="Actual outcome"
                value={fraudCase.actualIsFraud ? "Confirmed fraud" : "Legitimate"}
              />
              <DetailRow label="LLM used" value={fraudCase.modelUsed} />
            </dl>
          </section>

          <section className="glass rounded-2xl p-5">
            <h2 className="mb-4 text-sm font-medium text-[--muted]">
              Top SHAP contributions
            </h2>
            <ShapBar features={fraudCase.topShapFeatures} />
          </section>
        </div>

        <div className="flex flex-col gap-6 lg:col-span-3">
          <section className="glass-raised rounded-2xl p-6">
            <h2 className="mb-3 flex items-center gap-2 text-sm font-medium text-[--muted]">
              <span className="h-1.5 w-1.5 rounded-full bg-[--accent-violet]" />
              Analyst explanation
            </h2>
            <p className="whitespace-pre-line text-[15px] leading-relaxed text-[--foreground]">
              {fraudCase.explanation}
            </p>
          </section>

          <ChatPanel fraudCase={fraudCase} />
        </div>
      </div>
    </main>
  );
}

function DetailRow({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex flex-col gap-0.5">
      <dt className="text-xs text-[--muted-2]">{label}</dt>
      <dd className="text-[--foreground]">{value}</dd>
    </div>
  );
}
