export function CaseTypeBadge({ caseType }: { caseType: "true_fraud" | "false_positive" }) {
  if (caseType === "true_fraud") {
    return (
      <span className="inline-flex items-center gap-1.5 rounded-full bg-white/5 px-2.5 py-1 text-xs font-medium text-[--muted] ring-1 ring-inset ring-white/10">
        Actual fraud
      </span>
    );
  }
  return (
    <span className="inline-flex items-center gap-1.5 rounded-full bg-white/5 px-2.5 py-1 text-xs font-medium text-[--muted] ring-1 ring-inset ring-white/10">
      False positive
    </span>
  );
}
