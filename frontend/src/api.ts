import { getApiKey, getBackendUrl } from "./config";
import { getSessionId } from "./session";

export interface ChatTurn {
  role: "user" | "assistant";
  content: string;
}

export interface GenerateRequest {
  message: string;
  history: ChatTurn[];
  max_new_tokens: number;
  temperature: number;
  top_p: number;
  top_k_sources: number;
}

export interface GenerateAbResponse {
  interaction_id: string;
  answer_a: string;
  answer_b: string;
}

type StreamEvent = { type: "token"; token: string } | { type: "meta"; interactionId: string };

function requestHeaders(): HeadersInit {
  const headers: Record<string, string> = {
    "Content-Type": "application/json",
    sesh_id: getSessionId(),
  };
  const apiKey = getApiKey();
  if (apiKey) headers["x-api-key"] = apiKey;
  return headers;
}

export async function backendHealth(): Promise<unknown> {
  const res = await fetch(`${getBackendUrl()}/health`);
  if (!res.ok) throw new Error(`Health check failed: ${res.status}`);
  return res.json();
}

/**
 * Streams tokens from /generate_stream. Mirrors the original Streamlit app's
 * SSE parsing: lines prefixed "data: " carry JSON ({"token": "..."} or a
 * {"event": "meta", "interaction_id": "..."} marker), terminated by "data: [DONE]".
 */
export async function* streamGenerate(payload: GenerateRequest): AsyncGenerator<StreamEvent> {
  const res = await fetch(`${getBackendUrl()}/generate_stream`, {
    method: "POST",
    headers: requestHeaders(),
    body: JSON.stringify(payload),
  });

  if (!res.ok || !res.body) {
    throw new Error(`generate_stream failed: ${res.status}`);
  }

  const reader = res.body.getReader();
  const decoder = new TextDecoder("utf-8");
  let buffer = "";

  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });

    let lineEnd: number;
    while ((lineEnd = buffer.indexOf("\n")) >= 0) {
      const rawLine = buffer.slice(0, lineEnd).trim();
      buffer = buffer.slice(lineEnd + 1);
      if (!rawLine.startsWith("data:")) continue;

      const data = rawLine.slice("data:".length).trim();
      if (data === "[DONE]") return;

      try {
        const obj = JSON.parse(data);
        if (obj.event === "meta" && obj.interaction_id) {
          yield { type: "meta", interactionId: obj.interaction_id };
          continue;
        }
        const token = obj.token ?? obj.text ?? "";
        if (token) yield { type: "token", token };
      } catch {
        yield { type: "token", token: data };
      }
    }
  }
}

export async function generateAb(payload: GenerateRequest): Promise<GenerateAbResponse> {
  const res = await fetch(`${getBackendUrl()}/generate_ab`, {
    method: "POST",
    headers: requestHeaders(),
    body: JSON.stringify(payload),
  });
  if (!res.ok) throw new Error(`generate_ab failed: ${res.status}`);
  return res.json();
}

export async function sendAbVote(interactionId: string, choice: "A" | "B"): Promise<void> {
  const res = await fetch(`${getBackendUrl()}/ab_vote`, {
    method: "POST",
    headers: requestHeaders(),
    body: JSON.stringify({ interaction_id: interactionId, choice }),
  });
  if (!res.ok) throw new Error(`ab_vote failed: ${res.status}`);
}

export async function sendComment(interactionId: string, stars: number, comment: string): Promise<void> {
  const res = await fetch(`${getBackendUrl()}/comment`, {
    method: "POST",
    headers: requestHeaders(),
    body: JSON.stringify({ interaction_id: interactionId, stars, comment }),
  });
  if (!res.ok) throw new Error(`comment failed: ${res.status}`);
}
