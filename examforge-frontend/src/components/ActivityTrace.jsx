import { useEffect, useState } from "react";
import { ChevronRight } from "lucide-react";

/**
 * Reports genuine backend pipeline stages.
 *
 * Every entry corresponds to an event the retrieval pipeline emits as it runs
 * (understanding -> searching -> expanding -> reranking -> generating). No
 * private model reasoning is exposed, requested, or invented, and there are no
 * timers faking progress: if the backend does not emit a stage, it is not shown.
 *
 * The one exception is the brief window between submitting a question and the
 * first backend event arriving. To avoid a frozen-looking gap, a single honest
 * "Thinking" line is shown - clearly an assistant-activity indicator, not a
 * claim about a specific backend operation. It is replaced the instant a real
 * stage event lands.
 *
 * While streaming it reads as a single live line. Once the answer lands it
 * settles into one compact, collapsed summary row.
 */

const LABELS = {
  understanding: "Analysing your question",
  searching: "Reviewing your material",
  expanding: "Expanding the query",
  reranking: "Selecting relevant passages",
  generating: "Generating answer",
};

const ORDER = ["understanding", "searching", "expanding", "reranking", "generating"];

export default function ActivityTrace({
  stages = [],
  interpretation,
  sourceCount = 0,
  live = false,
}) {
  const [open, setOpen] = useState(false);

  // Rotate a gentle placeholder only until the first real stage arrives.
  const [waitIndex, setWaitIndex] = useState(0);
  const noRealStagesYet = live && stages.length === 0;

  useEffect(() => {
    if (!noRealStagesYet) return;
    if (window.matchMedia?.("(prefers-reduced-motion: reduce)").matches) return;
    const id = setInterval(
      () => setWaitIndex((i) => (i + 1) % WAITING.length),
      1800
    );
    return () => clearInterval(id);
  }, [noRealStagesYet]);

  // Non-live with nothing to show: render nothing.
  if (!live && !stages.length && !interpretation) return null;

  const ordered = [...stages].sort(
    (a, b) => ORDER.indexOf(a.stage) - ORDER.indexOf(b.stage)
  );
  const current = ordered[ordered.length - 1];

  let headline;
  if (live) {
    headline = noRealStagesYet
      ? WAITING[waitIndex]
      : LABELS[current?.stage] || current?.label || "Working";
  } else {
    headline =
      sourceCount > 0
        ? `Reviewed ${sourceCount} passage${sourceCount === 1 ? "" : "s"}`
        : "Retrieval complete";
  }

  const expandable = ordered.length > 0 || Boolean(interpretation);

  return (
    <div className="trace">
      <button
        type="button"
        className="trace__bar"
        onClick={() => expandable && setOpen((value) => !value)}
        aria-expanded={expandable ? open : undefined}
        aria-live={live ? "polite" : undefined}
        disabled={!expandable}
      >
        {live ? (
          <span className="trace__spinner" aria-hidden="true" />
        ) : (
          <span className="trace__tick" aria-hidden="true" />
        )}

        <span className={`trace__text ${live ? "trace__text--live" : ""}`}>
          {headline}
        </span>

        {expandable && (
          <ChevronRight
            size={13}
            className={`trace__caret ${open ? "is-open" : ""}`}
            aria-hidden="true"
          />
        )}
      </button>

      {open && (
        <ol className="trace__steps">
          {ordered.map((stage, index) => {
            const isCurrent = live && index === ordered.length - 1;
            return (
              <li
                key={stage.stage}
                className={`trace__step ${isCurrent ? "is-current" : "is-done"}`}
              >
                <span className="trace__node" aria-hidden="true" />
                <span>{LABELS[stage.stage] || stage.label || stage.stage}</span>
              </li>
            );
          })}

          {interpretation && (
            <li className="trace__step trace__step--note">
              <span className="trace__node trace__node--hollow" aria-hidden="true" />
              <span>
                Resolved as <em>&ldquo;{interpretation}&rdquo;</em>
              </span>
            </li>
          )}
        </ol>
      )}
    </div>
  );
}

// Shown only before the first real backend stage event. Presented as assistant
// activity, never as a claim about a specific operation.
const WAITING = ["Thinking…", "Getting started…", "Reviewing your material…"];
