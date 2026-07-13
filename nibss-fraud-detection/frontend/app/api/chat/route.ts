import { GoogleGenAI } from "@google/genai";
import { getCaseById } from "@/lib/cases";
import { featureLabel, formatHour, formatNaira } from "@/lib/types";

export const runtime = "nodejs";

const MODEL_FALLBACK_CHAIN = [
  "gemini-2.5-flash",
  "gemini-2.0-flash",
  "gemini-2.5-flash-lite",
];

type ChatMessage = { role: "user" | "assistant"; text: string };

function buildSystemContext(fraudCase: ReturnType<typeof getCaseById>) {
  if (!fraudCase) return "";
  const factors = fraudCase.topShapFeatures
    .map((f) => `- ${featureLabel(f.feature)}: ${f.value >= 0 ? "+" : ""}${f.value.toFixed(3)}`)
    .join("\n");

  return `You are a fraud analyst assistant for a Nigerian bank, answering a follow-up
question about a transaction that was already reviewed. Stay grounded in the
data below — do not invent transaction details or risk factors that are not
listed. If asked something the data can't answer, say so plainly.

TRANSACTION:
- Amount: ${formatNaira(fraudCase.amount)}
- Channel: ${fraudCase.channel}
- Hour: ${formatHour(fraudCase.hour)}
- Bank: ${fraudCase.bank}
- Location: ${fraudCase.location}
- Customer age group: ${fraudCase.ageGroup}

MODEL OUTPUT:
- Prediction: ${fraudCase.predictedIsFraud ? "FRAUD" : "NOT FRAUD"}
- Confidence: ${(fraudCase.fraudProbability * 100).toFixed(0)}%
- Actual outcome: ${fraudCase.actualIsFraud ? "confirmed fraud" : "legitimate"}

TOP SHAP RISK FACTORS:
${factors}

ORIGINAL EXPLANATION GIVEN TO THE ANALYST:
${fraudCase.explanation}

Answer the analyst's follow-up question in under 80 words.`;
}

export async function POST(request: Request) {
  const apiKey = process.env.GEMINI_API_KEY;
  if (!apiKey) {
    return Response.json(
      { error: "GEMINI_API_KEY is not configured on the server." },
      { status: 500 }
    );
  }

  const { caseId, question, history } = (await request.json()) as {
    caseId: number;
    question: string;
    history: ChatMessage[];
  };

  const fraudCase = getCaseById(caseId);
  if (!fraudCase) {
    return Response.json({ error: "Case not found." }, { status: 404 });
  }
  if (!question || !question.trim()) {
    return Response.json({ error: "Question is required." }, { status: 400 });
  }

  const client = new GoogleGenAI({ apiKey });

  const conversation = [
    buildSystemContext(fraudCase),
    ...(history ?? []).map((m) => `${m.role === "user" ? "Analyst" : "Assistant"}: ${m.text}`),
    `Analyst: ${question}`,
  ].join("\n\n");

  let lastError: unknown = null;
  for (const model of MODEL_FALLBACK_CHAIN) {
    // One retry per model for transient errors (e.g. "high demand" 503s)
    // before moving on to the next model in the chain.
    for (let attempt = 0; attempt < 2; attempt++) {
      try {
        const response = await client.models.generateContent({
          model,
          contents: conversation,
        });
        return Response.json({ answer: response.text?.trim(), modelUsed: model });
      } catch (err) {
        lastError = err;
        const message = err instanceof Error ? err.message : String(err);
        const isQuotaOrMissing =
          message.includes("RESOURCE_EXHAUSTED") || message.includes("NOT_FOUND");
        const isTransient = message.includes("UNAVAILABLE") || message.includes("503");

        if (isQuotaOrMissing) break; // no point retrying this model, skip to next
        if (isTransient && attempt === 0) {
          await new Promise((r) => setTimeout(r, 1500));
          continue; // retry same model once
        }
        break; // move to next model
      }
    }
  }

  const message = lastError instanceof Error ? lastError.message : "LLM request failed.";
  return Response.json(
    { error: `All models unavailable (likely daily free-tier quota exhausted): ${message}` },
    { status: 503 }
  );
}
