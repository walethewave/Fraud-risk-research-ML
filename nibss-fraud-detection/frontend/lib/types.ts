export type ShapFeature = {
  feature: string;
  value: number;
};

export type FraudCase = {
  id: number;
  caseType: "true_fraud" | "false_positive";
  actualIsFraud: boolean;
  predictedIsFraud: boolean;
  fraudProbability: number;
  amount: number;
  channel: string;
  hour: number;
  bank: string;
  location: string;
  ageGroup: string;
  topShapFeatures: ShapFeature[];
  explanation: string;
  modelUsed: string;
};

export type RiskLevel = "High" | "Medium" | "Low";

export function riskLevel(probability: number): RiskLevel {
  if (probability >= 0.7) return "High";
  if (probability >= 0.4) return "Medium";
  return "Low";
}

const FEATURE_LABELS: Record<string, string> = {
  amount: "Transaction amount",
  hour: "Hour of day",
  day_of_week: "Day of week",
  month: "Month",
  merchant_risk_score: "Merchant risk score",
  composite_risk: "Composite risk score",
  age_numeric: "Customer age",
};

export function featureLabel(feature: string): string {
  if (FEATURE_LABELS[feature]) return FEATURE_LABELS[feature];
  if (feature.startsWith("bank_")) return `Bank: ${feature.replace("bank_", "")}`;
  return feature.replace(/_/g, " ");
}

export function formatNaira(amount: number): string {
  return `₦${amount.toLocaleString("en-NG", { maximumFractionDigits: 0 })}`;
}

export function formatHour(hour: number): string {
  const period = hour < 12 ? "AM" : "PM";
  let displayHour = hour % 12;
  if (displayHour === 0) displayHour = 12;
  return `${displayHour} ${period}`;
}
