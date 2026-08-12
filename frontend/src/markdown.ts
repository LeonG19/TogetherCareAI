import DOMPurify from "dompurify";
import { marked } from "marked";

const SOURCES_MARKER = "\n\n**Sources:**\n";

export function stripSources(text: string): string {
  const idx = text.indexOf(SOURCES_MARKER);
  return idx >= 0 ? text.slice(0, idx).trim() : text;
}

export function renderMarkdown(text: string): { __html: string } {
  const html = marked.parse(text, { async: false, breaks: true }) as string;
  return { __html: DOMPurify.sanitize(html) };
}
