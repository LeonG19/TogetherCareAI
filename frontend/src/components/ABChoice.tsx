import { AnswerContent } from "./AnswerContent";

export interface PendingAb {
  interactionId: string;
  answerA: string;
  answerB: string;
}

export function ABChoice({
  ab,
  onChoose,
}: {
  ab: PendingAb;
  onChoose: (choice: "A" | "B") => void;
}) {
  return (
    <div className="chat-message assistant">
      <div className="avatar" aria-hidden="true">
        A
      </div>
      <div className="bubble ab-bubble">
        <p className="ab-label">Answer A (concise &amp; direct):</p>
        <AnswerContent text={ab.answerA} />
        <hr />
        <p className="ab-label">Answer B (warm &amp; empathic):</p>
        <AnswerContent text={ab.answerB} />
        <div className="ab-actions">
          <button type="button" onClick={() => onChoose("A")}>
            Prefer A
          </button>
          <button type="button" onClick={() => onChoose("B")}>
            Prefer B
          </button>
        </div>
      </div>
    </div>
  );
}
