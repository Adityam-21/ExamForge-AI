import { useCallback, useEffect, useState } from "react";
import { AlertCircle, PanelLeft, PanelLeftClose, Plus } from "lucide-react";

import Composer from "./components/Composer";
import MessageList from "./components/MessageList";
import Sidebar from "./components/Sidebar";
import { BrandMark, Wordmark } from "./components/Brand";
import { NoDocuments, NoMessages } from "./components/EmptyState";
import { useActions, useAppState } from "./state/AppContext";

const THEME_KEY = "examforge.theme";

function useTheme() {
  const [theme, setTheme] = useState(() => {
    const stored = localStorage.getItem(THEME_KEY);
    if (stored === "light" || stored === "dark") return stored;
    return window.matchMedia?.("(prefers-color-scheme: light)").matches
      ? "light"
      : "dark";
  });

  useEffect(() => {
    document.documentElement.dataset.theme = theme;
    localStorage.setItem(THEME_KEY, theme);
  }, [theme]);

  const toggle = useCallback(
    () => setTheme((value) => (value === "dark" ? "light" : "dark")),
    []
  );

  return [theme, toggle];
}

function Booting() {
  return (
    <div className="screen">
      <BrandMark size={26} className="screen__mark" />
      <p className="screen__text">Connecting to ExamForge</p>
      <div className="screen__bar" aria-hidden="true">
        <span />
      </div>
    </div>
  );
}

function BootFailed({ error, onRetry }) {
  return (
    <div className="screen">
      <AlertCircle size={22} className="screen__icon-error" aria-hidden="true" />
      <h2 className="screen__title">Can&apos;t reach the server</h2>
      <p className="screen__text">{error}</p>
      <button type="button" className="btn btn--primary" onClick={onRetry}>
        Try again
      </button>
    </div>
  );
}

export default function App() {
  const state = useAppState();
  const actions = useActions();
  const [theme, toggleTheme] = useTheme();
  const [drawerOpen, setDrawerOpen] = useState(false);
  const [railHidden, setRailHidden] = useState(false);

  useEffect(() => {
    const onKey = (event) => {
      if (event.key === "Escape") setDrawerOpen(false);
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);

  if (state.boot === "loading") return <Booting />;
  if (state.boot === "error") {
    return <BootFailed error={state.bootError} onRetry={actions.retryBoot} />;
  }

  const hasDocs = state.documents.length > 0;
  const busy = Boolean(state.stream);
  const hasMessages = state.messages.length > 0;
  const active = state.conversations.find((c) => c.id === state.activeId);

  const totalPages = state.documents.reduce(
    (sum, doc) => sum + (doc.page_count || 0),
    0
  );

  return (
    <div className={`shell ${railHidden ? "shell--rail-hidden" : ""}`}>
      <Sidebar
        open={drawerOpen}
        onClose={() => setDrawerOpen(false)}
        theme={theme}
        onToggleTheme={toggleTheme}
      />

      <main className="main">
        <header className="topbar">
          <button
            type="button"
            className="btn-icon topbar__drawer"
            onClick={() => setDrawerOpen(true)}
            aria-label="Open navigation"
          >
            <PanelLeft size={17} />
          </button>

          <button
            type="button"
            className="btn-icon topbar__rail"
            onClick={() => setRailHidden((value) => !value)}
            aria-label={railHidden ? "Show sidebar" : "Hide sidebar"}
            title={railHidden ? "Show sidebar" : "Hide sidebar"}
          >
            {railHidden ? <PanelLeft size={17} /> : <PanelLeftClose size={17} />}
          </button>

          <div className="topbar__brand">
            <BrandMark size={17} className="topbar__mark" />
            <Wordmark />
          </div>

          <div className="topbar__center">
            {hasMessages && active?.title && (
              <span className="topbar__title" title={active.title}>
                {active.title}
              </span>
            )}
          </div>

          <div className="topbar__meta">
            {hasDocs && (
              <span
                className="chip chip--quiet"
                title="Material indexed for retrieval"
              >
                <span className="chip__dot" aria-hidden="true" />
                {state.documents.length} doc
                {state.documents.length === 1 ? "" : "s"}
                {totalPages ? ` · ${totalPages}p` : ""}
              </span>
            )}
            <button
              type="button"
              className="btn-icon"
              onClick={actions.newConversation}
              aria-label="New conversation"
              title="New conversation"
            >
              <Plus size={17} />
            </button>
          </div>
        </header>

        {!hasDocs ? (
          <NoDocuments onUpload={actions.uploadDocuments} />
        ) : (
          <MessageList
            key={state.activeId ?? "empty"}
            messages={state.messages}
            stream={state.stream}
            streamError={state.streamError}
            loading={state.messagesLoading}
            onRegenerate={actions.regenerate}
            onRetry={actions.retryLast}
            header={
              !hasMessages && !busy && !state.messagesLoading ? (
                <NoMessages
                  documents={state.documents}
                  onPick={actions.sendMessage}
                />
              ) : null
            }
          />
        )}

        <Composer
          onSend={actions.sendMessage}
          onAttach={actions.uploadDocuments}
          onStop={actions.stopGenerating}
          busy={busy}
          disabled={!hasDocs}
          placeholder={hasDocs ? "Ask about your material…" : "Add a PDF to begin"}
        />
      </main>
    </div>
  );
}
