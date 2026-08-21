import { useEffect, useMemo, useRef, useState } from "react";
import { ChevronRight, FileText, Library } from "lucide-react";

/**
 * Retrieved evidence for one answer, grouped by document.
 *
 * Every field originates from document metadata written at ingest time and from
 * the reranker's score - never from the model's prose. Passages are grouped
 * under their source file so a single PDF appears once, with its cited pages
 * summarised; expanding a passage still reveals its exact page and excerpt.
 * Cited passages lead; the rest stay one click away.
 */

function matchLabel(score) {
  if (score >= 0.6) return { text: "Strong", level: "high" };
  if (score >= 0.25) return { text: "Good", level: "mid" };
  if (score >= 0.05) return { text: "Partial", level: "low" };
  return { text: "Weak", level: "min" };
}

function Passage({ source, expanded, onToggle, highlighted }) {
  const ref = useRef(null);
  const match = matchLabel(source.relevance_score ?? 0);

  useEffect(() => {
    if (highlighted && ref.current) {
      ref.current.scrollIntoView({ behavior: "smooth", block: "center" });
    }
  }, [highlighted]);

  return (
    <li
      ref={ref}
      className={`src ${expanded ? "is-open" : ""} ${highlighted ? "is-focus" : ""}`}
    >
      <button
        type="button"
        className="src__bar"
        onClick={onToggle}
        aria-expanded={expanded}
      >
        <span className="src__num">{source.id}</span>

        <span className="src__ref">
          {source.page != null && (
            <span className="src__page">Page {source.page}</span>
          )}
        </span>

        <span className={`match match--${match.level}`} title="Reranker score">
          <span className="match__bars" aria-hidden="true">
            <i />
            <i />
            <i />
          </span>
          {match.text}
        </span>

        <ChevronRight
          size={13}
          className={`src__caret ${expanded ? "is-open" : ""}`}
          aria-hidden="true"
        />
      </button>

      {expanded && <p className="src__excerpt">{source.text}</p>}
    </li>
  );
}

function DocumentGroup({ group, expandedIds, onToggle, focusId }) {
  const pages = group.passages
    .map((p) => p.page)
    .filter((p) => p != null)
    .sort((a, b) => a - b);
  const uniquePages = [...new Set(pages)];
  const pageSummary =
    uniquePages.length > 0
      ? `Page${uniquePages.length === 1 ? "" : "s"} ${uniquePages.join(", ")}`
      : null;

  return (
    <li className="docgroup">
      <div className="docgroup__head">
        <FileText size={12} aria-hidden="true" />
        <span className="docgroup__name" title={group.source}>
          {group.source}
        </span>
        <span className="docgroup__meta">
          {group.passages.length} passage{group.passages.length === 1 ? "" : "s"}
          {pageSummary ? ` · ${pageSummary}` : ""}
        </span>
      </div>

      <ul className="docgroup__list">
        {group.passages.map((source) => (
          <Passage
            key={source.id}
            source={source}
            expanded={expandedIds.has(source.id)}
            highlighted={focusId === source.id}
            onToggle={() => onToggle(source.id)}
          />
        ))}
      </ul>
    </li>
  );
}

function groupByDocument(items) {
  const map = new Map();
  for (const item of items) {
    const key = item.source || "Document";
    if (!map.has(key)) map.set(key, { source: key, passages: [] });
    map.get(key).passages.push(item);
  }
  return [...map.values()];
}

export default function SourcePanel({ sources = [], focusId = null }) {
  const [expanded, setExpanded] = useState(() => new Set());
  const [showRest, setShowRest] = useState(false);

  const cited = useMemo(() => sources.filter((s) => s.cited), [sources]);
  const primary = cited.length ? cited : sources;
  const rest = cited.length ? sources.filter((s) => !s.cited) : [];

  const primaryGroups = useMemo(() => groupByDocument(primary), [primary]);
  const restGroups = useMemo(() => groupByDocument(rest), [rest]);

  useEffect(() => {
    if (focusId == null) return;
    setExpanded((prev) => new Set(prev).add(focusId));
    if (rest.some((s) => s.id === focusId)) setShowRest(true);
  }, [focusId]); // eslint-disable-line react-hooks/exhaustive-deps

  if (!sources.length) return null;

  const toggle = (id) =>
    setExpanded((prev) => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });

  const documentCount = new Set(sources.map((s) => s.source)).size;

  return (
    <section className="evidence" aria-label="Sources for this answer">
      <header className="evidence__head">
        <Library size={12} aria-hidden="true" />
        <h4>Sources</h4>
        <span className="evidence__count">
          {cited.length > 0
            ? `${cited.length} cited`
            : `${sources.length} retrieved`}
          {" · "}
          {documentCount} document{documentCount === 1 ? "" : "s"}
        </span>
      </header>

      <ul className="evidence__docs">
        {primaryGroups.map((group) => (
          <DocumentGroup
            key={group.source}
            group={group}
            expandedIds={expanded}
            onToggle={toggle}
            focusId={focusId}
          />
        ))}
      </ul>

      {rest.length > 0 && (
        <>
          <button
            type="button"
            className="evidence__more"
            onClick={() => setShowRest((value) => !value)}
            aria-expanded={showRest}
          >
            <ChevronRight
              size={12}
              className={showRest ? "is-open" : ""}
              aria-hidden="true"
            />
            {showRest ? "Hide" : "Show"} {rest.length} other passage
            {rest.length === 1 ? "" : "s"} considered
          </button>

          {showRest && (
            <ul className="evidence__docs evidence__docs--rest">
              {restGroups.map((group) => (
                <DocumentGroup
                  key={group.source}
                  group={group}
                  expandedIds={expanded}
                  onToggle={toggle}
                  focusId={focusId}
                />
              ))}
            </ul>
          )}
        </>
      )}
    </section>
  );
}
