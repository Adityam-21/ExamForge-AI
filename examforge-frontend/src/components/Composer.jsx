import { useEffect, useRef, useState } from "react";
import { ArrowUp, Paperclip, Square } from "lucide-react";

const MAX_HEIGHT = 200;

/**
 * Deliberately understated: a single hairline border on a surface barely
 * separated from the background. Focus is signalled by a border shift and a
 * faint ring, not glow or a bright fill - the conversation stays dominant.
 */
export default function Composer({
  onSend,
  onAttach,
  onStop,
  busy,
  disabled,
  placeholder,
}) {
  const [value, setValue] = useState("");
  const textareaRef = useRef(null);
  const fileRef = useRef(null);

  useEffect(() => {
    const el = textareaRef.current;
    if (!el) return;
    el.style.height = "auto";
    const next = Math.min(el.scrollHeight, MAX_HEIGHT);
    el.style.height = `${next}px`;
    el.style.overflowY = el.scrollHeight > MAX_HEIGHT ? "auto" : "hidden";
  }, [value]);

  useEffect(() => {
    if (!busy && !disabled) textareaRef.current?.focus();
  }, [busy, disabled]);

  const submit = () => {
    const text = value.trim();
    if (!text || busy || disabled) return;
    onSend(text);
    setValue("");
  };

  const handleKeyDown = (event) => {
    if (
      event.key === "Enter" &&
      !event.shiftKey &&
      !event.nativeEvent.isComposing
    ) {
      event.preventDefault();
      submit();
    }
  };

  const handleFiles = (event) => {
    const files = Array.from(event.target.files || []);
    if (files.length) onAttach(files);
    event.target.value = "";
  };

  const canSend = Boolean(value.trim()) && !disabled;

  return (
    <div className="dock">
      <div className={`composer ${disabled ? "is-disabled" : ""}`}>
        <input
          ref={fileRef}
          type="file"
          accept="application/pdf,.pdf"
          multiple
          onChange={handleFiles}
          hidden
          tabIndex={-1}
        />

        <button
          type="button"
          className="btn-icon composer__attach"
          onClick={() => fileRef.current?.click()}
          aria-label="Attach a PDF"
          title="Attach a PDF"
        >
          <Paperclip size={16} />
        </button>

        <textarea
          ref={textareaRef}
          className="composer__input"
          rows={1}
          value={value}
          placeholder={placeholder}
          onChange={(event) => setValue(event.target.value)}
          onKeyDown={handleKeyDown}
          disabled={disabled}
          aria-label="Ask a question about your material"
        />

        {busy ? (
          <button
            type="button"
            className="composer__send composer__send--stop"
            onClick={onStop}
            aria-label="Stop generating"
            title="Stop generating"
          >
            <Square size={11} fill="currentColor" />
          </button>
        ) : (
          <button
            type="button"
            className="composer__send"
            onClick={submit}
            disabled={!canSend}
            aria-label="Send question"
            title="Send"
          >
            <ArrowUp size={16} />
          </button>
        )}
      </div>

      <p className="dock__hint">
        {disabled ? (
          "Add study material to start asking questions."
        ) : (
          <>
            <kbd>Enter</kbd> to send · <kbd>Shift</kbd>+<kbd>Enter</kbd> for a new
            line · answers cite your material
          </>
        )}
      </p>
    </div>
  );
}
