import { FormEvent, useRef, useState } from "react";
import { generateAb, sendAbVote, streamGenerate } from "./api";
import { ABChoice, PendingAb } from "./components/ABChoice";
import { ChatMessage, ChatMessageData } from "./components/ChatMessage";
import { getAbEnabled, getAbEveryN } from "./config";
import { renderMarkdown, stripSources } from "./markdown";

const GEN_PARAMS = {
  max_new_tokens: 512,
  temperature: 0.7,
  top_p: 0.9,
  top_k_sources: 3,
};

export default function App() {
  const [messages, setMessages] = useState<ChatMessageData[]>([]);
  const [pendingAb, setPendingAb] = useState<PendingAb | null>(null);
  const [input, setInput] = useState("");
  const [isBusy, setIsBusy] = useState(false);
  const [streamingText, setStreamingText] = useState<string | null>(null);
  const [errorText, setErrorText] = useState<string | null>(null);
  const scrollRef = useRef<HTMLDivElement>(null);

  function scrollToBottom() {
    requestAnimationFrame(() => {
      scrollRef.current?.scrollIntoView({ behavior: "smooth" });
    });
  }

  async function handleSubmit(e: FormEvent) {
    e.preventDefault();
    const text = input.trim();
    if (!text || isBusy) return;

    const historyForModel = messages.map((m) => ({
      role: m.role,
      content: m.role === "assistant" ? stripSources(m.content) : m.content,
    }));

    const nextMessages = [...messages, { role: "user" as const, content: text }];
    setMessages(nextMessages);
    setInput("");
    setErrorText(null);
    setIsBusy(true);
    scrollToBottom();

    const payload = {
      message: text,
      history: historyForModel,
      ...GEN_PARAMS,
    };

    const userTurnCount = nextMessages.filter((m) => m.role === "user").length;
    const abEveryN = getAbEveryN();
    const useAb = getAbEnabled() && abEveryN > 0 && userTurnCount % abEveryN === 0;

    try {
      if (useAb) {
        const resp = await generateAb(payload);
        setPendingAb({
          interactionId: resp.interaction_id,
          answerA: resp.answer_a,
          answerB: resp.answer_b,
        });
      } else {
        let finalText = "";
        let interactionId: string | null = null;
        setStreamingText("");

        for await (const event of streamGenerate(payload)) {
          if (event.type === "meta") {
            interactionId = event.interactionId;
            continue;
          }
          finalText += event.token;
          setStreamingText(finalText);
          scrollToBottom();
        }

        setMessages((prev) => [...prev, { role: "assistant", content: finalText, interactionId }]);
        setStreamingText(null);
      }
    } catch (err) {
      setErrorText(err instanceof Error ? err.message : String(err));
      setStreamingText(null);
    } finally {
      setIsBusy(false);
      scrollToBottom();
    }
  }

  async function handleAbChoice(choice: "A" | "B") {
    if (!pendingAb) return;
    const chosenText = choice === "A" ? pendingAb.answerA : pendingAb.answerB;
    const interactionId = pendingAb.interactionId;

    try {
      await sendAbVote(interactionId, choice);
    } catch (err) {
      setErrorText(err instanceof Error ? `Vote logging failed: ${err.message}` : "Vote logging failed");
    }

    setMessages((prev) => [...prev, { role: "assistant", content: chosenText, interactionId }]);
    setPendingAb(null);
    scrollToBottom();
  }

  return (
    <div className="app">
      <header className="app-header">
        <h1>TogetherCare AI</h1>
      </header>

      <main className="chat-log">
        {messages.map((m, i) => (
          <ChatMessage key={i} message={m} />
        ))}

        {streamingText !== null && (
          <div className="chat-message assistant">
            <div className="avatar" aria-hidden="true">
              A
            </div>
            <div className="bubble">
              <div className="content" dangerouslySetInnerHTML={renderMarkdown(streamingText || "…")} />
            </div>
          </div>
        )}

        {pendingAb && <ABChoice ab={pendingAb} onChoose={handleAbChoice} />}

        {errorText && <div className="chat-error">⚠️ {errorText}</div>}

        <div ref={scrollRef} />
      </main>

      <form className="chat-input" onSubmit={handleSubmit}>
        <input
          type="text"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder="Type your message..."
          disabled={isBusy}
        />
        <button type="submit" disabled={isBusy || !input.trim()}>
          Send
        </button>
      </form>
    </div>
  );
}
