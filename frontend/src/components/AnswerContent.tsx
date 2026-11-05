import { renderMarkdown, splitSources } from "../markdown";

/** An answer's text, with any retrieved sources tucked into a dropdown. */
export function AnswerContent({ text }: { text: string }) {
  const { body, sources } = splitSources(text);
  return (
    <>
      <div className="content" dangerouslySetInnerHTML={renderMarkdown(body || "…")} />
      {sources.length > 0 && (
        <details className="sources">
          <summary>
            Sources ({sources.length})
          </summary>
          <ul>
            {sources.map((s) => (
              <li key={s.url}>
                <a href={s.url} target="_blank" rel="noopener noreferrer">
                  {s.title}
                </a>
              </li>
            ))}
          </ul>
        </details>
      )}
    </>
  );
}
