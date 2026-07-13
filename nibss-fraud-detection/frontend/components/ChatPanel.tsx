"use client";

import { useState } from "react";
import type { FraudCase } from "@/lib/types";

type Message = { role: "user" | "assistant"; text: string };

export function ChatPanel({ fraudCase }: { fraudCase: FraudCase }) {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function sendMessage() {
    const question = input.trim();
    if (!question || loading) return;

    setInput("");
    setError(null);
    const nextMessages: Message[] = [...messages, { role: "user", text: question }];
    setMessages(nextMessages);
    setLoading(true);

    try {
      const res = await fetch("/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ caseId: fraudCase.id, question, history: messages }),
      });
      if (!res.ok) {
        const body = await res.json().catch(() => ({}));
        throw new Error(body.error || `Request failed (${res.status})`);
      }
      const data = await res.json();
      setMessages([...nextMessages, { role: "assistant", text: data.answer }]);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Something went wrong");
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="glass flex flex-col gap-4 rounded-2xl p-5">
      <div className="flex items-center gap-2">
        <span className="h-1.5 w-1.5 rounded-full bg-[--accent-blue]" />
        <h3 className="text-sm font-medium text-[--foreground]">Ask a follow-up</h3>
      </div>

      {messages.length > 0 && (
        <div className="flex flex-col gap-3 max-h-72 overflow-y-auto pr-1">
          {messages.map((m, i) => (
            <div
              key={i}
              className={`rounded-xl px-3.5 py-2.5 text-sm ${
                m.role === "user"
                  ? "self-end bg-white/[0.06] text-[--foreground]"
                  : "self-start glass-raised text-[--foreground]"
              }`}
            >
              {m.text}
            </div>
          ))}
        </div>
      )}

      {error && <p className="text-xs text-[--risk-high]">{error}</p>}

      <form
        onSubmit={(e) => {
          e.preventDefault();
          sendMessage();
        }}
        className="flex items-center gap-2"
      >
        <input
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder="e.g. Why wasn't the customer's account age considered?"
          className="flex-1 rounded-xl border border-[--border-subtle] bg-white/[0.03] px-3.5 py-2.5 text-sm text-[--foreground] placeholder:text-[--muted-2] outline-none focus:border-[--accent-violet]/50"
        />
        <button
          type="submit"
          disabled={loading || !input.trim()}
          className="rounded-xl bg-gradient-to-r from-[--accent-violet] to-[--accent-blue] px-4 py-2.5 text-sm font-medium text-black disabled:opacity-40"
        >
          {loading ? "..." : "Ask"}
        </button>
      </form>
    </div>
  );
}
