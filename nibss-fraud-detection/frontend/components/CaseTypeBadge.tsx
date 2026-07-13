export function CaseTypeBadge({
  caseType,
  predictedIsFraud,
}: {
  caseType: "true_fraud" | "false_positive";
  predictedIsFraud: boolean;
}) {
  // caseType is ground truth (was this transaction actually fraud).
  // predictedIsFraud is what the model said. The two can disagree —
  // that disagreement is the interesting part, so label it explicitly
  // instead of just restating ground truth.
  if (caseType === "true_fraud") {
    if (predictedIsFraud) {
      return <Badge tone="positive" label="Caught" sublabel="model flagged it, and it was fraud" />;
    }
    return <Badge tone="negative" label="Missed" sublabel="model said not fraud, but it was" />;
  }
  // caseType === "false_positive": ground truth is legitimate, model flagged it anyway.
  return <Badge tone="negative" label="False alarm" sublabel="model flagged it, but it was legitimate" />;
}

function Badge({
  tone,
  label,
  sublabel,
}: {
  tone: "positive" | "negative";
  label: string;
  sublabel: string;
}) {
  const toneClass =
    tone === "positive"
      ? "text-[--risk-low] bg-[--risk-low-bg] ring-[--risk-low]/25"
      : "text-[--risk-medium] bg-[--risk-medium-bg] ring-[--risk-medium]/25";
  return (
    <span
      title={sublabel}
      className={`inline-flex items-center gap-1.5 rounded-full px-2.5 py-1 text-xs font-medium ring-1 ring-inset ${toneClass}`}
    >
      {label}
    </span>
  );
}
