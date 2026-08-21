import { memo, useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import remarkMath from "remark-math";
import rehypeKatex from "rehype-katex";
import { Check, Copy } from "lucide-react";

/**
 * Turns bare citation markers into internal links so they can be rendered as
 * interactive pills without enabling raw HTML.
 *
 * `[1]` -> `[1](#efcite-1)`, skipping anything already a markdown link and
 * anything inside a fenced code block.
 */
function linkCitations(text) {
  if (!text) return "";

  return text
    .split(/(```[\s\S]*?```|`[^`\n]*`)/g)
    .map((segment, index) => {
      // Odd indices are the captured code spans/fences: leave them untouched.
      if (index % 2 === 1) return segment;
      return segment.replace(/\[(\d{1,2})\](?!\()/g, "[$1](#efcite-$1)");
    })
    .join("");
}

function CodeBlock({ children, className }) {
  const [copied, setCopied] = useState(false);
  const code = String(children ?? "").replace(/\n$/, "");
  const language = /language-(\w+)/.exec(className || "")?.[1];

  const copy = async () => {
    try {
      await navigator.clipboard.writeText(code);
      setCopied(true);
      setTimeout(() => setCopied(false), 1600);
    } catch {
      /* clipboard unavailable */
    }
  };

  return (
    <div className="code-block">
      <div className="code-block__bar">
        <span className="code-block__lang">{language || "code"}</span>
        <button
          type="button"
          className="btn-icon btn-icon--xs"
          onClick={copy}
          aria-label="Copy code"
        >
          {copied ? <Check size={13} /> : <Copy size={13} />}
        </button>
      </div>
      <pre>
        <code className={className}>{code}</code>
      </pre>
    </div>
  );
}

function Markdown({ children, onCitationClick }) {
  const components = {
    a({ href, children: label, ...props }) {
      const match = /^#efcite-(\d+)$/.exec(href || "");
      if (match) {
        const id = Number(match[1]);
        return (
          <button
            type="button"
            className="citation-pill"
            onClick={() => onCitationClick?.(id)}
            aria-label={`Show source ${id}`}
          >
            {id}
          </button>
        );
      }
      return (
        <a href={href} target="_blank" rel="noreferrer noopener" {...props}>
          {label}
        </a>
      );
    },

    code({ inline, className, children: code, ...props }) {
      if (inline) {
        return (
          <code className="inline-code" {...props}>
            {code}
          </code>
        );
      }
      return <CodeBlock className={className}>{code}</CodeBlock>;
    },

    pre({ children: inner }) {
      return <>{inner}</>;
    },

    table({ children: inner }) {
      return (
        <div className="table-scroll">
          <table>{inner}</table>
        </div>
      );
    },
  };

  return (
    <div className="prose">
      <ReactMarkdown
        remarkPlugins={[remarkGfm, remarkMath]}
        rehypePlugins={[[rehypeKatex, { throwOnError: false, strict: false }]]}
        components={components}
      >
        {linkCitations(children)}
      </ReactMarkdown>
    </div>
  );
}

export default memo(Markdown);
