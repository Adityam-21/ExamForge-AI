/**
 * ExamForge API client.
 *
 * The base URL comes from VITE_API_URL so the same build works against a local
 * backend and the deployed one. It is never hardcoded.
 */

const BASE_URL = (
  import.meta.env?.VITE_API_URL || "http://localhost:8000"
).replace(/\/$/, "");

export class ApiError extends Error {
  constructor(message, status) {
    super(message);
    this.name = "ApiError";
    this.status = status;
  }
}

async function request(path, options = {}) {
  let response;
  try {
    response = await fetch(`${BASE_URL}${path}`, options);
  } catch {
    throw new ApiError(
      "Can't reach the ExamForge server. Check your connection and try again.",
      0
    );
  }

  if (response.status === 204) return null;

  const isJson = (response.headers.get("content-type") || "").includes(
    "application/json"
  );
  const body = isJson ? await response.json().catch(() => null) : null;

  if (!response.ok) {
    throw new ApiError(
      detailToMessage(body?.detail) || `Request failed (${response.status})`,
      response.status
    );
  }

  return body;
}

/** FastAPI validation errors arrive as an array of objects, not a string. */
function detailToMessage(detail) {
  if (!detail) return null;
  if (typeof detail === "string") return detail;
  if (Array.isArray(detail)) {
    return detail.map((d) => d?.msg).filter(Boolean).join(". ") || null;
  }
  return null;
}

const json = (body) => ({
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify(body),
});

/* --- Sessions ------------------------------------------------------------- */

export const createSession = () => request("/api/sessions", { method: "POST" });

export const getSessionState = (sessionId) =>
  request(`/api/sessions/${sessionId}`);

/* --- Documents ------------------------------------------------------------ */

export const listDocuments = (sessionId) =>
  request(`/api/sessions/${sessionId}/documents`);

export function uploadDocument(sessionId, file, { signal } = {}) {
  const form = new FormData();
  form.append("file", file);
  return request(`/api/sessions/${sessionId}/documents`, {
    method: "POST",
    body: form,
    signal,
  });
}

export const deleteDocument = (sessionId, docId) =>
  request(`/api/sessions/${sessionId}/documents/${docId}`, {
    method: "DELETE",
  });

/* --- Conversations -------------------------------------------------------- */

export const listConversations = (sessionId) =>
  request(`/api/sessions/${sessionId}/conversations`);

export const createConversation = (sessionId) =>
  request(`/api/sessions/${sessionId}/conversations`, { method: "POST" });

export const listMessages = (conversationId) =>
  request(`/api/conversations/${conversationId}/messages`);

export const deleteConversation = (conversationId) =>
  request(`/api/conversations/${conversationId}`, { method: "DELETE" });

export const renameConversation = (conversationId, title) =>
  request(`/api/conversations/${conversationId}`, {
    ...json({ title }),
    method: "PATCH",
  });

/* --- Ask (non-streaming fallback) ----------------------------------------- */

export const ask = (conversationId, question, replaceMessageId = null) =>
  request(
    `/api/conversations/${conversationId}/ask`,
    json({ question, replace_message_id: replaceMessageId })
  );

/* --- Ask (streaming) ------------------------------------------------------ */

/**
 * Stream an answer over SSE.
 *
 * Events mirror real backend pipeline stages:
 *   start | stage | interpretation | sources | token | done | error
 *
 * @returns {Promise<void>} resolves when the stream closes.
 */
export async function askStream(
  conversationId,
  question,
  { replaceMessageId = null, signal, onEvent } = {}
) {
  let response;
  try {
    response = await fetch(
      `${BASE_URL}/api/conversations/${conversationId}/ask/stream`,
      {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Accept: "text/event-stream",
        },
        body: JSON.stringify({
          question,
          replace_message_id: replaceMessageId,
        }),
        signal,
      }
    );
  } catch (error) {
    if (error?.name === "AbortError") throw error;
    throw new ApiError(
      "Can't reach the ExamForge server. Check your connection and try again.",
      0
    );
  }

  if (!response.ok) {
    const body = await response.json().catch(() => null);
    throw new ApiError(
      detailToMessage(body?.detail) || `Request failed (${response.status})`,
      response.status
    );
  }

  if (!response.body) {
    throw new ApiError("Streaming is not supported by this browser.", 0);
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });

      // SSE frames are separated by a blank line.
      const frames = buffer.split("\n\n");
      buffer = frames.pop() ?? "";

      for (const frame of frames) {
        const parsed = parseFrame(frame);
        if (parsed) onEvent?.(parsed.event, parsed.data);
      }
    }

    const tail = parseFrame(buffer);
    if (tail) onEvent?.(tail.event, tail.data);
  } finally {
    reader.cancel().catch(() => {});
  }
}

function parseFrame(frame) {
  const trimmed = frame.trim();
  if (!trimmed) return null;

  let event = "message";
  const dataLines = [];

  for (const line of trimmed.split("\n")) {
    if (line.startsWith("event:")) event = line.slice(6).trim();
    else if (line.startsWith("data:")) dataLines.push(line.slice(5).trim());
  }

  if (!dataLines.length) return null;

  try {
    return { event, data: JSON.parse(dataLines.join("\n")) };
  } catch {
    return null;
  }
}

export const apiBaseUrl = BASE_URL;
