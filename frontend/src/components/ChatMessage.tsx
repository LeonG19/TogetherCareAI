import { renderMarkdown } from "../markdown";
import { FeedbackForm } from "./FeedbackForm";

export interface ChatMessageData {
  role: "user" | "assistant";
  content: string;
  interactionId?: string | null;
}

export function ChatMessage({ message }: { message: ChatMessageData }) {
  return (
    <div className={`chat-message ${message.role}`}>
      <div className="avatar" aria-hidden="true">
        {message.role === "user" ? "U" : "A"}
      </div>
      <div className="bubble">
        <div className="content" dangerouslySetInnerHTML={renderMarkdown(message.content)} />
        {message.role === "assistant" && message.interactionId && (
          <FeedbackForm interactionId={message.interactionId} />
        )}
      </div>
    </div>
  );
}
