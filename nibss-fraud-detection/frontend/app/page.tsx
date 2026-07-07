import Link from "next/link";
import { CaseQueue } from "@/components/CaseQueue";
import { getAllCases } from "@/lib/cases";

export default function Home() {
  const cases = getAllCases();
  const avgFaithfulProxy = cases.length
    ? cases.reduce((sum, c) => sum + c.fraudProbability, 0) / cases.length
    : 0;

  return (
    <main className="mx-auto flex w-full max-w-6xl flex-1 flex-col gap-8 px-6 py-12 sm:px-10">
      <header className="flex flex-col gap-3">
        <div className="flex items-center justify-between gap-2">
          <div className="flex items-center gap-2 text-xs font-medium uppercase tracking-widest text-[--muted-2]">
            <span className="h-1.5 w-1.5 rounded-full bg-[--accent-violet]" />
            NIBSS Fraud Review
          </div>
          <Link
            href="/discovery"
            className="glass rounded-full px-3.5 py-1.5 text-xs font-medium text-[--foreground] hover:text-gradient"
          >
            Feature Discovery &rarr;
          </Link>
        </div>
        <h1 className="text-3xl font-semibold tracking-tight sm:text-4xl">
          Flagged transaction <span className="text-gradient">queue</span>
        </h1>
        <p className="max-w-2xl text-[--muted]">
          Random Forest predictions paired with SHAP-grounded, LLM-narrated
          explanations. Click a case to see the model&apos;s reasoning and ask
          follow-up questions.
        </p>
      </header>

      <div className="grid grid-cols-2 gap-4 sm:grid-cols-4">
        <StatTile label="Cases reviewed" value={String(cases.length)} />
        <StatTile
          label="Actual fraud"
          value={String(cases.filter((c) => c.caseType === "true_fraud").length)}
        />
        <StatTile
          label="False positives"
          value={String(cases.filter((c) => c.caseType === "false_positive").length)}
        />
        <StatTile
          label="Avg. confidence"
          value={`${(avgFaithfulProxy * 100).toFixed(0)}%`}
        />
      </div>

      <CaseQueue cases={cases} />
    </main>
  );
}

function StatTile({ label, value }: { label: string; value: string }) {
  return (
    <div className="glass flex flex-col gap-1 rounded-2xl px-5 py-4">
      <span className="text-xs uppercase tracking-wide text-[--muted-2]">{label}</span>
      <span className="font-mono text-2xl text-[--foreground]">{value}</span>
    </div>
  );
}
