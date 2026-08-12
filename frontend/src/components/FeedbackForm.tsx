import { useState } from "react";
import { sendComment } from "../api";

export function FeedbackForm({ interactionId }: { interactionId: string }) {
  const [open, setOpen] = useState(false);
  const [stars, setStars] = useState(5);
  const [comment, setComment] = useState("");
  const [status, setStatus] = useState<"idle" | "sending" | "sent" | "error">("idle");

  async function submit(e: React.FormEvent) {
    e.preventDefault();
    setStatus("sending");
    try {
      await sendComment(interactionId, stars, comment);
      setStatus("sent");
    } catch {
      setStatus("error");
    }
  }

  return (
    <div className="feedback">
      <button type="button" className="feedback-toggle" onClick={() => setOpen((v) => !v)}>
        {open ? "Hide rating" : "Rate this response (optional)"}
      </button>
      {open && (
        <form className="feedback-form" onSubmit={submit}>
          <label>
            Rating
            <div className="stars">
              {[1, 2, 3, 4, 5].map((n) => (
                <button
                  key={n}
                  type="button"
                  className={n <= stars ? "star star-filled" : "star"}
                  onClick={() => setStars(n)}
                  aria-label={`${n} star`}
                >
                  ★
                </button>
              ))}
            </div>
          </label>
          <label>
            Comment (optional)
            <textarea value={comment} onChange={(e) => setComment(e.target.value)} rows={3} />
          </label>
          <button type="submit" disabled={status === "sending"}>
            Submit feedback
          </button>
          {status === "sent" && <span className="feedback-status ok">Thanks! Feedback saved.</span>}
          {status === "error" && <span className="feedback-status err">Feedback failed. Try again.</span>}
        </form>
      )}
    </div>
  );
}
