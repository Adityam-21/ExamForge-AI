import { useRef, useState } from "react";
import {
  Check,
  FileText,
  Loader2,
  Moon,
  Plus,
  Sun,
  Trash2,
  Upload,
  X,
} from "lucide-react";

import { BrandMark, Wordmark } from "./Brand";
import { useActions, useAppState } from "../state/AppContext";

function formatBytes(bytes) {
  if (!bytes) return "";
  const mb = bytes / (1024 * 1024);
  return mb >= 1 ? `${mb.toFixed(1)} MB` : `${Math.round(bytes / 1024)} KB`;
}

function DocumentRow({ document, onRemove }) {
  const [confirming, setConfirming] = useState(false);

  return (
    <li className="doc">
      <FileText size={13} className="doc__icon" aria-hidden="true" />

      <span className="doc__body">
        <span className="doc__name" title={document.filename}>
          {document.filename}
        </span>
        <span className="doc__meta">
          {document.page_count}p · {document.chunk_count} passages
          {document.size_bytes ? ` · ${formatBytes(document.size_bytes)}` : ""}
        </span>
      </span>

      {confirming ? (
        <span className="doc__confirm">
          <button
            type="button"
            className="btn-text btn-text--danger"
            onClick={() => onRemove(document.id)}
          >
            Remove
          </button>
          <button
            type="button"
            className="btn-text"
            onClick={() => setConfirming(false)}
          >
            Keep
          </button>
        </span>
      ) : (
        <button
          type="button"
          className="btn-icon btn-icon--xs doc__action"
          onClick={() => setConfirming(true)}
          aria-label={`Remove ${document.filename}`}
          title="Remove"
        >
          <Trash2 size={12} />
        </button>
      )}
    </li>
  );
}

function UploadRow({ upload, onDismiss }) {
  const failed = upload.status === "error";

  return (
    <li className={`doc ${failed ? "doc--error" : "doc--busy"}`}>
      {failed ? (
        <X size={13} className="doc__icon" aria-hidden="true" />
      ) : (
        <Loader2 size={13} className="doc__icon spin" aria-hidden="true" />
      )}

      <span className="doc__body">
        <span className="doc__name" title={upload.filename}>
          {upload.filename}
        </span>
        <span className="doc__meta">
          {failed ? upload.error : "Extracting and embedding…"}
        </span>
      </span>

      {failed && (
        <button
          type="button"
          className="btn-icon btn-icon--xs doc__action"
          onClick={() => onDismiss(upload.id)}
          aria-label="Dismiss"
        >
          <X size={12} />
        </button>
      )}
    </li>
  );
}

export default function Sidebar({ open, onClose, theme, onToggleTheme }) {
  const state = useAppState();
  const actions = useActions();
  const fileRef = useRef(null);

  const handleFiles = (event) => {
    const files = Array.from(event.target.files || []);
    if (files.length) actions.uploadDocuments(files);
    event.target.value = "";
  };

  const select = (id) => {
    actions.selectConversation(id);
    onClose?.();
  };

  const start = async () => {
    await actions.newConversation();
    onClose?.();
  };

  return (
    <>
      <div
        className={`scrim ${open ? "is-visible" : ""}`}
        onClick={onClose}
        aria-hidden="true"
      />

      <aside
        className={`rail ${open ? "is-open" : ""}`}
        aria-label="ExamForge navigation"
      >
        <div className="rail__head">
          <span className="rail__brand">
            <BrandMark size={19} className="rail__mark" />
            <Wordmark />
          </span>

          <button
            type="button"
            className="btn-icon rail__close"
            onClick={onClose}
            aria-label="Close navigation"
          >
            <X size={17} />
          </button>
        </div>

        <div className="rail__actions">
          <button type="button" className="btn btn--ghost-block" onClick={start}>
            <Plus size={14} aria-hidden="true" />
            <span>New conversation</span>
          </button>
        </div>

        {/* Independently scrollable so a long history never affects the shell. */}
        <div className="rail__scroll">
          <div className="rail__group">
            <p className="rail__label">History</p>

            {state.conversations.length === 0 ? (
              <p className="rail__empty">
                Conversations you start will be listed here.
              </p>
            ) : (
              <ul className="convs">
                {state.conversations.map((conversation) => {
                  const isActive = conversation.id === state.activeId;
                  return (
                    <li key={conversation.id}>
                      <div className={`conv ${isActive ? "is-active" : ""}`}>
                        <button
                          type="button"
                          className="conv__open"
                          onClick={() => select(conversation.id)}
                          aria-current={isActive ? "true" : undefined}
                        >
                          <span className="conv__title">
                            {conversation.title || "Untitled"}
                          </span>
                        </button>
                        <button
                          type="button"
                          className="btn-icon btn-icon--xs conv__action"
                          onClick={() =>
                            actions.removeConversation(conversation.id)
                          }
                          aria-label={`Delete ${conversation.title || "conversation"}`}
                          title="Delete"
                        >
                          <Trash2 size={12} />
                        </button>
                      </div>
                    </li>
                  );
                })}
              </ul>
            )}
          </div>
        </div>

        <div className="rail__docs">
          <div className="rail__label-row">
            <p className="rail__label">Grounded in</p>
            <button
              type="button"
              className="btn-icon btn-icon--xs"
              onClick={() => fileRef.current?.click()}
              aria-label="Add a PDF"
              title="Add a PDF"
            >
              <Upload size={13} />
            </button>
          </div>

          <input
            ref={fileRef}
            type="file"
            accept="application/pdf,.pdf"
            multiple
            onChange={handleFiles}
            hidden
            tabIndex={-1}
          />

          {state.documents.length === 0 && state.uploads.length === 0 ? (
            <p className="rail__empty">
              No material yet. Add a PDF to give ExamForge something to search.
            </p>
          ) : (
            <ul className="docs">
              {state.documents.map((document) => (
                <DocumentRow
                  key={document.id}
                  document={document}
                  onRemove={actions.removeDocument}
                />
              ))}
              {state.uploads.map((upload) => (
                <UploadRow
                  key={upload.id}
                  upload={upload}
                  onDismiss={actions.dismissUpload}
                />
              ))}
            </ul>
          )}

          {state.documents.length > 0 && (
            <p className="rail__note">
              <Check size={11} aria-hidden="true" />
              Answers are limited to this material
            </p>
          )}
        </div>

        <div className="rail__foot">
          <button type="button" className="btn-text btn-text--row" onClick={onToggleTheme}>
            {theme === "dark" ? <Sun size={13} /> : <Moon size={13} />}
            <span>{theme === "dark" ? "Light" : "Dark"} appearance</span>
          </button>
        </div>
      </aside>
    </>
  );
}
