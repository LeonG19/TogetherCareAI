const SESSION_KEY = "togethercare_session_id";

export function getSessionId(): string {
  let id = localStorage.getItem(SESSION_KEY);
  if (!id) {
    id = crypto.randomUUID().replace(/-/g, "");
    localStorage.setItem(SESSION_KEY, id);
  }
  return id;
}
