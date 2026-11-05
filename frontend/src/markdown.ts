import DOMPurify from "dompurify";
import { marked } from "marked";

const SOURCES_MARKER = "\n\n**Sources:**\n";

export function stripSources(text: string): string {
  const idx = text.indexOf(SOURCES_MARKER);
  return idx >= 0 ? text.slice(0, idx).trim() : text;
}

export interface Source {
  title: string;
  url: string;
}

/**
 * Splits an answer into its text and the retrieved sources the backend appends
 * ("- title: url" lines under a "**Sources:**" heading). If the block can't be
 * parsed, the text is returned untouched so nothing is ever hidden.
 */
export function splitSources(text: string): { body: string; sources: Source[] } {
  const idx = text.indexOf(SOURCES_MARKER);
  if (idx < 0) return { body: text, sources: [] };

  const sources: Source[] = [];
  for (const line of text.slice(idx + SOURCES_MARKER.length).split("\n")) {
    const match = line.trim().match(/^-\s+(.*?):\s+(https?:\/\/\S+)$/);
    if (match) sources.push({ title: match[1] || match[2], url: match[2] });
  }
  return sources.length ? { body: text.slice(0, idx).trim(), sources } : { body: text, sources: [] };
}

export function renderMarkdown(text: string): { __html: string } {
  const html = marked.parse(text, { async: false, breaks: true }) as string;
  return { __html: DOMPurify.sanitize(html) };
}
