import { useState } from "react";
import { Check, Copy, Info, RefreshCw } from "lucide-react";

import Markdown from "./Markdown";
import SourcePanel from "./SourcePanel";
import ActivityTrace from "./ActivityTrace";
import { AssistantGlyph } from "./Brand";

export function UserMessage({ message }) {
  return (
    <article className="turn turn--user">
      <div className="ask">{message.content}</div>
    </article>
  );
}

function Actions({ content, onRegenerate }) {
  const [copied, setCopied] = useState(false);

  const copy = async () => {
    try {
      await navigator.clipboard.writeText(content);
      setCopied(true);
      setTimeout(() => setCopied(false), 1600);
    } catch {
      /* clipboard unavailable */
    }
  };

  return (
    <div className="turn__tools">
      <button
        type="button"
        className="btn-text btn-text--row"
        onClick={copy}
        aria-label="Copy answer"
      >
        {copied ? <Check size={12} /> : <Copy size={12} />}
        <span>{copied ? "Copied" : "Copy"}</span>
      </button>

      {onRegenerate && (
        <button
          type="button"
          className="btn-text btn-text--row"
          onClick={onRegenerate}
          aria-label="Regenerate answer"
        >
          <RefreshCw size={12} />
          <span>Regenerate</span>
        </button>
      )}
    </div>
  );
}

export function AssistantMessage({ message, onRegenerate, busy }) {
  const [focusId, setFocusId] = useState(null);
  const sources = message.citations ?? [];
  const unsupported = message.grounded === false;

  return (
    <article className="turn turn--reply">
      <div className="turn__rail">
        <AssistantGlyph />
      </div>

      <div className="turn__body">
        {(message.interpreted_as || sources.length > 0) && (
          <ActivityTrace
            stages={[{ stage: "searching" }]}
            interpretation={message.interpreted_as}
            sourceCount={sources.length}
          />
        )}

        {unsupported && (
          <div className="note note--caution">
            <Info size={13} aria-hidden="true" />
            <span>
              Low confidence: the retrieved passages only weakly match this
              question. Verify the answer against your material before relying
              on it.
            </span>
          </div>
        )}

        <div className="reply">
          <Markdown onCitationClick={setFocusId}>{message.content}</Markdown>
        </div>

        {sources.length > 0 && (
          <SourcePanel sources={sources} focusId={focusId} />
        )}

        {!busy && <Actions content={message.content} onRegenerate={onRegenerate} />}
      </div>
    </article>
  );
}

/** The in-flight turn: live stage line, then progressively rendered markdown. */
export function StreamingMessage({ stream }) {
  const [focusId, setFocusId] = useState(null);
  const hasText = stream.text.length > 0;

  return (
    <article className="turn turn--reply">
      <div className="turn__rail">
        <AssistantGlyph />
      </div>

      <div className="turn__body">
        <ActivityTrace
          stages={stream.stages}
          interpretation={stream.interpretation}
          sourceCount={stream.sources.length}
          live
        />

        {hasText && (
          <div className="reply">
            <Markdown onCitationClick={setFocusId}>{stream.text}</Markdown>
            <span className="cursor" aria-hidden="true" />
          </div>
        )}

        {hasText && stream.sources.length > 0 && (
          <SourcePanel sources={stream.sources} focusId={focusId} />
        )}
      </div>
    </article>
  );
}
