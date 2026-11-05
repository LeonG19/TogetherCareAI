import { FormEvent, KeyboardEvent, useEffect, useRef, useState } from "react";
import { generateAb, ResponseMode, sendAbVote, streamGenerate } from "./api";
import { AnswerContent } from "./components/AnswerContent";
import { ABChoice, PendingAb } from "./components/ABChoice";
import { ChatMessage, ChatMessageData } from "./components/ChatMessage";
import { getAbEnabled, getAbEveryN } from "./config";
import { stripSources } from "./markdown";

const GEN_PARAMS = {
  max_new_tokens: 512,
  temperature: 0.7,
  top_p: 0.9,
  top_k_sources: 3,
};

// Response length options for the toggle. The backend decides the token
// budget for each mode.
const RESPONSE_MODES: { value: ResponseMode; label: string }[] = [
  { value: "short", label: "Quick" },
  { value: "detailed", label: "Detailed" },
];

// Example questions shown on the landing screen. Edit freely.
const EXAMPLE_QUESTIONS = [
  "What is ADRD?",
  "How can I help my loved one with memory loss?",
  "When should I call a doctor versus going to the emergency room?",
  "How can I manage caregiver stress and prevent burnout?",
  "How can I create a safe home environment to prevent falls?",
];

export default function App() {
  const [messages, setMessages] = useState<ChatMessageData[]>([]);
  const [pendingAb, setPendingAb] = useState<PendingAb | null>(null);
  const [input, setInput] = useState("");
  const [mode, setMode] = useState<ResponseMode>("short");
  const [isBusy, setIsBusy] = useState(false);
  const [streamingText, setStreamingText] = useState<string | null>(null);
  const [errorText, setErrorText] = useState<string | null>(null);
  const scrollRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);

  useEffect(() => {
    inputRef.current?.focus();
  }, []);

  function scrollToBottom() {
    requestAnimationFrame(() => {
      scrollRef.current?.scrollIntoView({ behavior: "smooth" });
    });
  }

  function handleSubmit(e: FormEvent) {
    e.preventDefault();
    void sendMessage(input);
  }

  // Enter sends, Shift+Enter inserts a newline. Ignore Enter while an IME is
  // composing so it can confirm a candidate instead of sending.
  function handleKeyDown(e: KeyboardEvent<HTMLTextAreaElement>) {
    if (e.key === "Enter" && !e.shiftKey && !e.nativeEvent.isComposing) {
      e.preventDefault();
      void sendMessage(input);
    }
  }

  async function sendMessage(raw: string) {
    const text = raw.trim();
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
      mode,
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

  const showLanding = messages.length === 0 && !isBusy && !pendingAb;

  return (
    <div className="app">
      <header className="app-header">
        <h1>TogetherCare AI</h1>
        <fieldset className="mode-toggle">
          <legend>Response length</legend>
          {RESPONSE_MODES.map((m) => (
            <label key={m.value} className={m.value === mode ? "mode-option selected" : "mode-option"}>
              <input
                type="radio"
                name="response-mode"
                value={m.value}
                checked={m.value === mode}
                onChange={() => setMode(m.value)}
              />
              {m.label}
            </label>
          ))}
        </fieldset>
      </header>

      <main className="chat-log">
        {showLanding && (
          <section className="landing" aria-labelledby="landing-hint">
            <p id="landing-hint" className="landing-hint">
              Not sure what to ask? Try one of these:
            </p>
            <ul className="chips">
              {EXAMPLE_QUESTIONS.map((q) => (
                <li key={q}>
                  <button type="button" className="chip" onClick={() => void sendMessage(q)} disabled={isBusy}>
                    {q}
                  </button>
                </li>
              ))}
            </ul>
          </section>
        )}

        {messages.map((m, i) => (
          <ChatMessage key={i} message={m} />
        ))}

        {streamingText !== null && (
          <div className="chat-message assistant">
            <div className="avatar" aria-hidden="true">
              A
            </div>
            <div className="bubble">
              <AnswerContent text={streamingText} />
            </div>
          </div>
        )}

        {pendingAb && <ABChoice ab={pendingAb} onChoose={handleAbChoice} />}

        {errorText && <div className="chat-error">⚠️ {errorText}</div>}

        <div ref={scrollRef} />
      </main>

      <form className="chat-input" onSubmit={handleSubmit}>
        <textarea
          ref={inputRef}
          rows={2}
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={handleKeyDown}
          aria-label="Ask a question"
          placeholder="Ask a question about caring for someone with Alzheimer's or dementia"
        />
        <button type="submit" className="send-button" disabled={isBusy || !input.trim()}>
          Send
          <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
            <line x1="22" y1="2" x2="11" y2="13" />
            <polygon points="22 2 15 22 11 13 2 9 22 2" />
          </svg>
        </button>
      </form>
    </div>
  );
}
