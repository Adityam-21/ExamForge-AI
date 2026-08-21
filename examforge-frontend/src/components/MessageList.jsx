import { useCallback, useLayoutEffect, useRef, useState } from "react";
import { ArrowDown, RotateCcw, TriangleAlert } from "lucide-react";

import { AssistantMessage, StreamingMessage, UserMessage } from "./Message";

const NEAR_BOTTOM_PX = 140;

/**
 * The single scrollable region of the application.
 *
 * The shell locks to the viewport, so this element - and only this element -
 * scrolls the conversation. A long answer can never push the sidebar, header or
 * composer out of reach.
 */
export default function MessageList({
  messages,
  stream,
  streamError,
  loading,
  onRegenerate,
  onRetry,
  header,
}) {
  const scrollRef = useRef(null);
  const [pinned, setPinned] = useState(true);

  const isNearBottom = useCallback(() => {
    const el = scrollRef.current;
    if (!el) return true;
    return el.scrollHeight - el.scrollTop - el.clientHeight < NEAR_BOTTOM_PX;
  }, []);

  const handleScroll = useCallback(() => {
    setPinned(isNearBottom());
  }, [isNearBottom]);

  // Follow new content only while already at the bottom, so reading back
  // through a long answer is never yanked away.
  useLayoutEffect(() => {
    if (!pinned) return;
    const el = scrollRef.current;
    if (el) el.scrollTop = el.scrollHeight;
  }, [messages, stream?.text, stream?.stages.length, pinned]);

  const jump = () => {
    const el = scrollRef.current;
    if (el) el.scrollTo({ top: el.scrollHeight, behavior: "smooth" });
    setPinned(true);
  };

  const lastAssistantId = [...messages]
    .reverse()
    .find((m) => m.role === "assistant")?.id;

  return (
    <div className="thread-wrap">
      <div className="thread" ref={scrollRef} onScroll={handleScroll}>
        <div className="thread__col">
          {header}

          {loading && (
            <div className="thread__loading" aria-live="polite">
              <span className="bar" />
              <span className="bar" />
              <span className="bar" />
            </div>
          )}

          {messages.map((message) =>
            message.role === "user" ? (
              <UserMessage key={message.id} message={message} />
            ) : (
              <AssistantMessage
                key={message.id}
                message={message}
                busy={Boolean(stream)}
                onRegenerate={
                  message.id === lastAssistantId && !stream
                    ? () => onRegenerate(message.id)
                    : undefined
                }
              />
            )
          )}

          {stream && <StreamingMessage stream={stream} />}

          {streamError && (
            <div className="note note--error" role="alert">
              <TriangleAlert size={14} aria-hidden="true" />
              <div className="note__body">
                <strong>That answer didn&apos;t complete</strong>
                <p>{streamError}</p>
              </div>
              <button type="button" className="btn btn--quiet" onClick={onRetry}>
                <RotateCcw size={12} aria-hidden="true" />
                <span>Retry</span>
              </button>
            </div>
          )}

          <div className="thread__tail" />
        </div>
      </div>

      {!pinned && (
        <button
          type="button"
          className="jump"
          onClick={jump}
          aria-label="Jump to latest message"
          title="Latest"
        >
          <ArrowDown size={14} />
        </button>
      )}
    </div>
  );
}
