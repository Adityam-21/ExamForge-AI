import { useRef, useState } from "react";
import { FileUp, Layers, ScanSearch, Quote } from "lucide-react";

import { BrandMark } from "./Brand";

const PILLARS = [
  {
    icon: Layers,
    title: "Hybrid retrieval",
    body: "Keyword and semantic search run together, then query rewriting widens the net.",
  },
  {
    icon: ScanSearch,
    title: "Reranked evidence",
    body: "A cross-encoder reorders every candidate passage before anything is written.",
  },
  {
    icon: Quote,
    title: "Traceable answers",
    body: "Each claim links to the document and page it came from.",
  },
];

const STARTERS = [
  "What are the key concepts here?",
  "Turn this into a revision sheet",
  "What's most likely to be examined?",
  "Explain the hardest topic simply",
];

/** No documents yet: upload is the only sensible next step. */
export function NoDocuments({ onUpload }) {
  const fileRef = useRef(null);
  const [dragging, setDragging] = useState(false);

  const accept = (list) =>
    Array.from(list || []).filter(
      (file) =>
        file.type === "application/pdf" ||
        file.name.toLowerCase().endsWith(".pdf")
    );

  return (
    <div className="stage">
      <div className="stage__col">
        <div className="hero">
          <BrandMark size={30} className="hero__mark" />
          <h1 className="hero__title">Your material, cross-examined</h1>
          <p className="hero__lede">
            Add lecture notes, a textbook chapter or past papers. ExamForge
            retrieves from your own documents and cites every answer back to the
            page.
          </p>
        </div>

        <div
          className={`drop ${dragging ? "is-dragging" : ""}`}
          onDragOver={(event) => {
            event.preventDefault();
            setDragging(true);
          }}
          onDragLeave={() => setDragging(false)}
          onDrop={(event) => {
            event.preventDefault();
            setDragging(false);
            const files = accept(event.dataTransfer.files);
            if (files.length) onUpload(files);
          }}
          onClick={() => fileRef.current?.click()}
          role="button"
          tabIndex={0}
          onKeyDown={(event) => {
            if (event.key === "Enter" || event.key === " ") {
              event.preventDefault();
              fileRef.current?.click();
            }
          }}
        >
          <input
            ref={fileRef}
            type="file"
            accept="application/pdf,.pdf"
            multiple
            hidden
            tabIndex={-1}
            onChange={(event) => {
              const files = accept(event.target.files);
              if (files.length) onUpload(files);
              event.target.value = "";
            }}
          />

          <FileUp size={19} className="drop__icon" aria-hidden="true" />
          <p className="drop__lead">Drop a PDF, or click to browse</p>
          <p className="drop__sub">Up to 10 MB per file · text-based PDFs</p>
        </div>

        <ul className="pillars">
          {PILLARS.map(({ icon: Icon, title, body }) => (
            <li key={title}>
              <Icon size={14} aria-hidden="true" />
              <div>
                <strong>{title}</strong>
                <span>{body}</span>
              </div>
            </li>
          ))}
        </ul>
      </div>
    </div>
  );
}

/** Documents exist, conversation is empty. */
export function NoMessages({ documents, onPick }) {
  const pages = documents.reduce((sum, d) => sum + (d.page_count || 0), 0);
  const label =
    documents.length === 1
      ? `${documents[0].filename} · ${pages} pages`
      : `${documents.length} documents · ${pages} pages`;

  return (
    <div className="opener">
      <p className="opener__eyebrow">Indexed and ready</p>
      <h2 className="opener__title">Ask anything about your material</h2>
      <p className="opener__meta">{label}</p>

      <ul className="starters">
        {STARTERS.map((starter) => (
          <li key={starter}>
            <button type="button" onClick={() => onPick(starter)}>
              {starter}
            </button>
          </li>
        ))}
      </ul>
    </div>
  );
}
