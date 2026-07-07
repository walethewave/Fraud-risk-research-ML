import Link from "next/link";
import { ZScoreBar } from "@/components/ZScoreBar";
import { AucComparison } from "@/components/AucComparison";
import { getDiscoveryData } from "@/lib/discovery";
import { featureLabel } from "@/lib/types";

const VERDICT_STYLE: Record<string, string> = {
  IMPROVED: "text-[--risk-low] bg-[--risk-low-bg] ring-[--risk-low]/30",
  "NO MEANINGFUL CHANGE": "text-[--risk-medium] bg-[--risk-medium-bg] ring-[--risk-medium]/30",
  WORSE: "text-[--risk-high] bg-[--risk-high-bg] ring-[--risk-high]/30",
};

export default function DiscoveryPage() {
  const data = getDiscoveryData();
  const { falseNegativeAnalysis, llmProposal, impact } = data;

  return (
    <main className="mx-auto flex w-full max-w-5xl flex-1 flex-col gap-8 px-6 py-12 sm:px-10">
      <div className="flex flex-col gap-3">
        <Link href="/" className="flex items-center gap-1.5 text-sm text-[--muted] hover:text-[--foreground]">
          <span aria-hidden>&larr;</span> Back to queue
        </Link>
        <div className="flex items-center gap-2 text-xs font-medium uppercase tracking-widest text-[--muted-2]">
          <span className="h-1.5 w-1.5 rounded-full bg-[--accent-pink]" />
          LLM-Assisted Feature Discovery
        </div>
        <h1 className="text-3xl font-semibold tracking-tight sm:text-4xl">
          Closing the loop: <span className="text-gradient">missed fraud → new features</span>
        </h1>
        <p className="max-w-2xl text-[--muted]">
          Instead of stopping at explanation, this pipeline analyzes the{" "}
          {falseNegativeAnalysis.nFalseNegatives} fraud cases the model missed, asks an
          LLM to propose candidate features grounded in that analysis, and empirically
          tests whether adding them actually moves AUC.
        </p>
      </div>

      {/* Step 1 */}
      <section className="glass rounded-2xl p-6">
        <StepHeader n={1} title="Where does the model miss?" />
        <div className="mb-5 grid grid-cols-3 gap-3 text-sm">
          <Stat label="False negatives" value={falseNegativeAnalysis.nFalseNegatives} />
          <Stat label="True positives" value={falseNegativeAnalysis.nTruePositives} />
          <Stat label="True negatives" value={falseNegativeAnalysis.nTrueNegatives} />
        </div>
        <p className="mb-4 text-sm text-[--muted]">
          13 behavioral/velocity columns exist in the raw dataset but were dropped
          before modeling. Do any show missed-fraud transactions behaving differently
          from normal legitimate ones?
        </p>
        <ZScoreBar
          candidates={falseNegativeAnalysis.candidates}
          highlighted={llmProposal.recommendedFeatures}
        />
      </section>

      {/* Step 2 */}
      <section className="glass-raised rounded-2xl p-6">
        <StepHeader n={2} title="LLM proposes candidates" />
        <p className="mb-3 text-xs text-[--muted-2]">
          Single call to {llmProposal.modelUsed}, grounded strictly in the statistics
          above — no invented domain claims.
        </p>
        <pre className="whitespace-pre-wrap rounded-xl bg-black/20 p-4 text-sm leading-relaxed text-[--foreground]">
          {llmProposal.rawResponse}
        </pre>
      </section>

      {/* Step 3 */}
      <section className="glass rounded-2xl p-6">
        <StepHeader n={3} title="Does it actually move AUC?" />
        <p className="mb-4 text-sm text-[--muted]">
          Retrained the identical Random Forest, same split, with vs. without{" "}
          {impact.candidateFeatures.map((f, i) => (
            <span key={f}>
              {i > 0 && ", "}
              <span className="text-[--foreground]">{featureLabel(f)}</span>
            </span>
          ))}
          .
        </p>
        <AucComparison impact={impact} />
        <div
          className={`mt-5 inline-flex items-center gap-2 rounded-full px-3.5 py-1.5 text-sm font-medium ring-1 ring-inset ${VERDICT_STYLE[impact.verdict]}`}
        >
          <span className="h-1.5 w-1.5 rounded-full bg-current" />
          Verdict: {impact.verdict}
        </div>
      </section>

      <section className="glass rounded-2xl p-6 text-sm text-[--muted]">
        <p>
          The gain here is real but not from the features alone (each is near-random in
          isolation, AUC ≈ 0.53) — it comes from an interaction with the existing 17
          features. The baseline model has no customer-history baseline at all; the
          winning feature answers &quot;is this unusual for this specific customer?&quot;,
          a question the original 17 features literally could not ask. See{" "}
          <code className="rounded bg-white/10 px-1.5 py-0.5">FINDINGS.md</code> in the
          repo for the full writeup, including why this was checked for leakage and
          ruled out.
        </p>
      </section>
    </main>
  );
}

function StepHeader({ n, title }: { n: number; title: string }) {
  return (
    <div className="mb-4 flex items-center gap-3">
      <span className="flex h-7 w-7 items-center justify-center rounded-full bg-white/[0.06] font-mono text-xs text-[--muted]">
        {n}
      </span>
      <h2 className="text-base font-medium text-[--foreground]">{title}</h2>
    </div>
  );
}

function Stat({ label, value }: { label: string; value: number }) {
  return (
    <div className="rounded-xl bg-white/[0.03] px-3.5 py-3">
      <div className="text-[10px] uppercase tracking-wide text-[--muted-2]">{label}</div>
      <div className="font-mono text-xl text-[--foreground]">{value}</div>
    </div>
  );
}
