"""
Week 3 — Prompt builder.

Turns a transaction row + model prediction + top-3 SHAP contributions into
the frozen prompt template (design.md). The template was hand-tested against
Gemini Flash on real true-positive and false-positive cases before being
frozen here (see explainability/manual_prompt_test.md).
"""
from dataclasses import dataclass
from typing import List, Tuple

FEATURE_LABELS = {
    "amount": "transaction amount",
    "hour": "hour of day",
    "day_of_week": "day of week",
    "month": "month",
    "merchant_risk_score": "merchant risk score",
    "composite_risk": "composite risk score",
    "age_numeric": "customer age",
}


def _feature_label(feature_name: str) -> str:
    if feature_name in FEATURE_LABELS:
        return FEATURE_LABELS[feature_name]
    if feature_name.startswith("bank_"):
        return f"bank ({feature_name.removeprefix('bank_')})"
    return feature_name.replace("_", " ")


@dataclass
class Transaction:
    amount: float
    channel: str
    hour: int
    bank: str
    location: str
    age_group: str


@dataclass
class Prediction:
    is_fraud: bool
    fraud_probability: float


SYSTEM_PREAMBLE = "You are a fraud analyst assistant for a Nigerian bank."

PROMPT_TEMPLATE = """{preamble}
A machine learning model has flagged the following transaction as potentially {flag_word}.

TRANSACTION DETAILS:
- Amount: {naira_amount}
- Channel: {channel}
- Hour: {hour_str}
- Bank: {bank}
- Location: {location}
- Customer age group: {age_group}

MODEL OUTPUT:
- Fraud prediction: {prediction_label}
- Confidence: {confidence:.0f}%

KEY RISK FACTORS (from model analysis, ranked by contribution magnitude):
{risk_factors}

A positive contribution means the factor pushed the prediction toward fraud;
a negative contribution means it pushed toward legitimate (i.e. it reduced
risk, it did not add to it).

Write a brief explanation for the fraud analyst that:
1. States the primary reason this was flagged
2. Lists 2-3 supporting risk factors, using ONLY the factors listed above — do not invent or mention any other transaction detail as a risk factor
3. States the risk level (High/Medium/Low)
4. Recommends a specific action (block/step-up auth/monitor)

Keep it under 100 words. Be direct and specific."""


def format_naira(amount: float) -> str:
    return f"₦{amount:,.0f}"


def format_hour(hour: int) -> str:
    period = "AM" if hour < 12 else "PM"
    display_hour = hour % 12
    display_hour = 12 if display_hour == 0 else display_hour
    return f"{display_hour} {period}"


def build_prompt(
    transaction: Transaction,
    prediction: Prediction,
    top_shap_features: List[Tuple[str, float]],
) -> str:
    """top_shap_features: list of (feature_name, shap_value), highest |value| first."""
    risk_lines = []
    for feature_name, shap_value in top_shap_features:
        label = _feature_label(feature_name)
        sign = "+" if shap_value >= 0 else ""
        risk_lines.append(
            f"- {label.capitalize()} (contribution: {sign}{shap_value:.3f})"
        )

    return PROMPT_TEMPLATE.format(
        preamble=SYSTEM_PREAMBLE,
        flag_word="fraudulent" if prediction.is_fraud else "unusual",
        naira_amount=format_naira(transaction.amount),
        channel=transaction.channel,
        hour_str=format_hour(transaction.hour),
        bank=transaction.bank,
        location=transaction.location,
        age_group=transaction.age_group,
        prediction_label="FRAUD" if prediction.is_fraud else "NOT FRAUD",
        confidence=prediction.fraud_probability * 100,
        risk_factors="\n".join(risk_lines),
    )
