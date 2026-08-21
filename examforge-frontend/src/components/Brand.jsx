/**
 * ExamForge identity marks.
 *
 * Inline SVG rather than an icon dependency: the mark is brand, not iconography.
 * The geometry is a hexagonal die with an ascending internal stroke - a crucible
 * refining raw material into something sharper. It reads as technical and
 * academic rather than playful.
 */

export function BrandMark({ size = 20, className = "" }) {
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 24 24"
      fill="none"
      className={className}
      aria-hidden="true"
      focusable="false"
    >
      <path
        d="M12 2.4 20.6 7.2v9.6L12 21.6 3.4 16.8V7.2L12 2.4Z"
        stroke="currentColor"
        strokeWidth="1.5"
        strokeLinejoin="round"
      />
      <path
        d="M8.2 15.1 12 9.4l3.8 5.7"
        stroke="currentColor"
        strokeWidth="1.5"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
      <path
        d="M9.9 12.6h4.2"
        stroke="currentColor"
        strokeWidth="1.5"
        strokeLinecap="round"
        opacity="0.55"
      />
    </svg>
  );
}

/** Compact glyph beside each assistant turn. Identity without a card. */
export function AssistantGlyph() {
  return (
    <span className="glyph" aria-hidden="true">
      <svg width="12" height="12" viewBox="0 0 24 24" fill="none">
        <path
          d="M12 2.4 20.6 7.2v9.6L12 21.6 3.4 16.8V7.2L12 2.4Z"
          stroke="currentColor"
          strokeWidth="2"
          strokeLinejoin="round"
        />
      </svg>
    </span>
  );
}

export function Wordmark() {
  return (
    <span className="wordmark">
      Exam<span className="wordmark__accent">Forge</span>
    </span>
  );
}
