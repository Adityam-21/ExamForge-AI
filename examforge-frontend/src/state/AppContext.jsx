/**
 * Application state.
 *
 * One reducer owns session bootstrap, conversations, documents, the message
 * transcript and the live stream. Components stay presentational and read from
 * context, which keeps data flow traceable in one place.
 */

import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useReducer,
  useRef,
  useState,
} from "react";

import * as api from "../lib/api";

const SESSION_KEY = "examforge.session";
const CONVERSATION_KEY = "examforge.conversation";

const initialState = {
  boot: "loading", // loading | ready | error
  bootError: null,

  sessionId: null,
  documents: [],
  uploads: [], // in-flight uploads: { id, filename, size, status, error }

  conversations: [],
  activeId: null,

  messages: [],
  messagesLoading: false,

  // Live generation
  stream: null, // { stages, interpretation, sources, text, replacing }
  streamError: null,

  toast: null,
};

function reducer(state, action) {
  switch (action.type) {
    case "BOOT_OK":
      return {
        ...state,
        boot: "ready",
        bootError: null,
        sessionId: action.sessionId,
        documents: action.documents,
        conversations: action.conversations,
        activeId: action.activeId,
      };

    case "BOOT_FAIL":
      return { ...state, boot: "error", bootError: action.error };

    case "RETRY_BOOT":
      return { ...state, boot: "loading", bootError: null };

    /* Documents */
    case "UPLOAD_START":
      return { ...state, uploads: [...state.uploads, action.upload] };

    case "UPLOAD_DONE":
      return {
        ...state,
        uploads: state.uploads.filter((u) => u.id !== action.id),
        documents: [...state.documents, action.document],
      };

    case "UPLOAD_FAIL":
      return {
        ...state,
        uploads: state.uploads.map((u) =>
          u.id === action.id ? { ...u, status: "error", error: action.error } : u
        ),
      };

    case "UPLOAD_DISMISS":
      return { ...state, uploads: state.uploads.filter((u) => u.id !== action.id) };

    case "DOCUMENT_REMOVED":
      return {
        ...state,
        documents: state.documents.filter((d) => d.id !== action.id),
      };

    /* Conversations */
    case "CONVERSATION_ADDED":
      return {
        ...state,
        conversations: [action.conversation, ...state.conversations],
        activeId: action.conversation.id,
        messages: [],
        stream: null,
        streamError: null,
      };

    case "CONVERSATION_SELECTED":
      return {
        ...state,
        activeId: action.id,
        messages: [],
        messagesLoading: true,
        stream: null,
        streamError: null,
      };

    case "CONVERSATION_REMOVED": {
      const conversations = state.conversations.filter((c) => c.id !== action.id);
      const wasActive = state.activeId === action.id;
      return {
        ...state,
        conversations,
        activeId: wasActive ? conversations[0]?.id ?? null : state.activeId,
        messages: wasActive ? [] : state.messages,
        messagesLoading: wasActive && conversations.length > 0,
        stream: wasActive ? null : state.stream,
      };
    }

    case "CONVERSATION_PATCHED":
      return {
        ...state,
        conversations: state.conversations.map((c) =>
          c.id === action.id ? { ...c, ...action.patch } : c
        ),
      };

    /* Messages */
    case "MESSAGES_LOADED":
      if (action.conversationId !== state.activeId) return state;
      return { ...state, messages: action.messages, messagesLoading: false };

    case "MESSAGE_APPENDED":
      return { ...state, messages: [...state.messages, action.message] };

    case "MESSAGES_TRUNCATED_FROM": {
      const index = state.messages.findIndex((m) => m.id === action.id);
      if (index === -1) return state;
      return { ...state, messages: state.messages.slice(0, index) };
    }

    /* Streaming */
    case "STREAM_START":
      return {
        ...state,
        streamError: null,
        stream: {
          stages: [],
          interpretation: null,
          sources: [],
          text: "",
          replacing: action.replacing ?? null,
        },
      };

    case "STREAM_STAGE": {
      if (!state.stream) return state;
      const stages = state.stream.stages.some((s) => s.stage === action.stage.stage)
        ? state.stream.stages
        : [...state.stream.stages, action.stage];
      return { ...state, stream: { ...state.stream, stages } };
    }

    case "STREAM_INTERPRETATION":
      if (!state.stream) return state;
      return {
        ...state,
        stream: { ...state.stream, interpretation: action.question },
      };

    case "STREAM_SOURCES":
      if (!state.stream) return state;
      return { ...state, stream: { ...state.stream, sources: action.sources } };

    case "STREAM_TOKEN":
      if (!state.stream) return state;
      return {
        ...state,
        stream: { ...state.stream, text: state.stream.text + action.text },
      };

    case "STREAM_END":
      return { ...state, stream: null };

    /**
     * Turn the finished stream into a real transcript entry in one update.
     * Clearing the stream first and waiting for the server reload would blank
     * the answer for a frame.
     */
    case "STREAM_COMMIT": {
      if (!state.stream) return state;
      const message = {
        id: action.messageId,
        role: "assistant",
        content: state.stream.text,
        citations: action.citations ?? state.stream.sources,
        grounded: action.grounded,
        interpreted_as: state.stream.interpretation,
        created_at: action.createdAt ?? new Date().toISOString(),
      };
      return { ...state, messages: [...state.messages, message], stream: null };
    }

    case "STREAM_FAIL":
      return { ...state, stream: null, streamError: action.error };

    case "STREAM_CLEAR_ERROR":
      return { ...state, streamError: null };

    case "TOAST":
      return { ...state, toast: action.toast };

    default:
      return state;
  }
}

const StateContext = createContext(null);
const ActionsContext = createContext(null);

export function AppProvider({ children }) {
  const [state, dispatch] = useReducer(reducer, initialState);
  const [bootAttempt, setBootAttempt] = useState(0);
  const abortRef = useRef(null);

  /* --- Bootstrap ---------------------------------------------------------- */

  useEffect(() => {
    let cancelled = false;

    async function boot() {
      try {
        const stored = localStorage.getItem(SESSION_KEY);
        let sessionId = stored;
        let documents = [];
        let conversations = [];

        if (sessionId) {
          // The backend adopts unknown ids, so a stale id recovers rather than
          // dead-ending after a redeploy.
          const snapshot = await api.getSessionState(sessionId);
          documents = snapshot.documents;
          conversations = snapshot.conversations;
        } else {
          const created = await api.createSession();
          sessionId = created.session_id;
          localStorage.setItem(SESSION_KEY, sessionId);
        }

        if (cancelled) return;

        const remembered = localStorage.getItem(CONVERSATION_KEY);
        const activeId =
          conversations.find((c) => c.id === remembered)?.id ??
          conversations[0]?.id ??
          null;

        dispatch({
          type: "BOOT_OK",
          sessionId,
          documents,
          conversations,
          activeId,
        });
      } catch (error) {
        if (!cancelled) {
          dispatch({ type: "BOOT_FAIL", error: error.message });
        }
      }
    }

    boot();
    return () => {
      cancelled = true;
    };
  }, [bootAttempt]);

  /* --- Load messages when the active conversation changes ----------------- */

  useEffect(() => {
    if (!state.activeId) return;
    let cancelled = false;

    localStorage.setItem(CONVERSATION_KEY, state.activeId);

    api
      .listMessages(state.activeId)
      .then((messages) => {
        if (!cancelled) {
          dispatch({
            type: "MESSAGES_LOADED",
            conversationId: state.activeId,
            messages,
          });
        }
      })
      .catch(() => {
        if (!cancelled) {
          dispatch({
            type: "MESSAGES_LOADED",
            conversationId: state.activeId,
            messages: [],
          });
        }
      });

    return () => {
      cancelled = true;
    };
  }, [state.activeId]);

  /* --- Actions ------------------------------------------------------------ */

  const toast = useCallback((message, tone = "info") => {
    dispatch({ type: "TOAST", toast: { message, tone, at: Date.now() } });
  }, []);

  const retryBoot = useCallback(() => {
    dispatch({ type: "RETRY_BOOT" });
    setBootAttempt((n) => n + 1);
  }, []);

  const uploadDocuments = useCallback(
    async (files) => {
      const sessionId = state.sessionId;
      if (!sessionId) return;

      for (const file of files) {
        const id = `${file.name}-${Date.now()}-${Math.random()}`;
        dispatch({
          type: "UPLOAD_START",
          upload: {
            id,
            filename: file.name,
            size: file.size,
            status: "processing",
          },
        });

        try {
          const document = await api.uploadDocument(sessionId, file);
          dispatch({ type: "UPLOAD_DONE", id, document });
        } catch (error) {
          dispatch({ type: "UPLOAD_FAIL", id, error: error.message });
        }
      }
    },
    [state.sessionId]
  );

  const dismissUpload = useCallback((id) => {
    dispatch({ type: "UPLOAD_DISMISS", id });
  }, []);

  const removeDocument = useCallback(
    async (docId) => {
      try {
        await api.deleteDocument(state.sessionId, docId);
        dispatch({ type: "DOCUMENT_REMOVED", id: docId });
      } catch (error) {
        toast(error.message, "error");
      }
    },
    [state.sessionId, toast]
  );

  const newConversation = useCallback(async () => {
    try {
      const conversation = await api.createConversation(state.sessionId);
      dispatch({ type: "CONVERSATION_ADDED", conversation });
      return conversation;
    } catch (error) {
      toast(error.message, "error");
      return null;
    }
  }, [state.sessionId, toast]);

  const selectConversation = useCallback(
    (id) => {
      if (id === state.activeId) return;
      abortRef.current?.abort();
      dispatch({ type: "CONVERSATION_SELECTED", id });
    },
    [state.activeId]
  );

  const removeConversation = useCallback(
    async (id) => {
      try {
        await api.deleteConversation(id);
        if (id === state.activeId) abortRef.current?.abort();
        dispatch({ type: "CONVERSATION_REMOVED", id });
      } catch (error) {
        toast(error.message, "error");
      }
    },
    [state.activeId, toast]
  );

  /* --- Ask --------------------------------------------------------------- */

  const runStream = useCallback(
    async (conversationId, question, replaceMessageId) => {
      const controller = new AbortController();
      abortRef.current = controller;

      dispatch({ type: "STREAM_START", replacing: replaceMessageId });

      let finished = false;

      try {
        await api.askStream(conversationId, question, {
          replaceMessageId,
          signal: controller.signal,
          onEvent: (event, data) => {
            switch (event) {
              case "stage":
                dispatch({ type: "STREAM_STAGE", stage: data });
                break;
              case "interpretation":
                dispatch({
                  type: "STREAM_INTERPRETATION",
                  question: data.question,
                });
                break;
              case "sources":
                dispatch({ type: "STREAM_SOURCES", sources: data.sources });
                break;
              case "token":
                dispatch({ type: "STREAM_TOKEN", text: data.text });
                break;
              case "error":
                finished = true;
                dispatch({ type: "STREAM_FAIL", error: data.message });
                break;
              case "done":
                finished = true;
                dispatch({
                  type: "STREAM_COMMIT",
                  messageId: data.message_id,
                  citations: data.citations,
                  grounded: data.grounded,
                  createdAt: data.created_at,
                });
                break;
              default:
                break;
            }
          },
        });
      } catch (error) {
        if (error?.name === "AbortError") {
          // Stopped by the user. The backend persists the partial answer, so
          // reloading the transcript keeps the UI honest about what was saved.
          finished = true;
        } else {
          finished = true;
          dispatch({ type: "STREAM_FAIL", error: error.message });
        }
      } finally {
        abortRef.current = null;
        if (!finished) dispatch({ type: "STREAM_END" });
      }

      // Reconcile with the server so ids and stored content are authoritative.
      try {
        const messages = await api.listMessages(conversationId);
        dispatch({ type: "MESSAGES_LOADED", conversationId, messages });
      } catch {
        /* transient - the committed view remains */
      }

      try {
        const conversations = await api.listConversations(state.sessionId);
        const updated = conversations.find((c) => c.id === conversationId);
        if (updated) {
          dispatch({
            type: "CONVERSATION_PATCHED",
            id: conversationId,
            patch: updated,
          });
        }
      } catch {
        /* non-critical */
      }
    },
    [state.sessionId]
  );

  const sendMessage = useCallback(
    async (question) => {
      const text = question.trim();
      if (!text) return;

      let conversationId = state.activeId;
      if (!conversationId) {
        const created = await newConversation();
        if (!created) return;
        conversationId = created.id;
      }

      // Optimistic user turn so the message appears instantly.
      dispatch({
        type: "MESSAGE_APPENDED",
        message: {
          id: `pending-${Date.now()}`,
          role: "user",
          content: text,
          citations: [],
          grounded: true,
          created_at: new Date().toISOString(),
          pending: true,
        },
      });

      await runStream(conversationId, text, null);
    },
    [state.activeId, newConversation, runStream]
  );

  const regenerate = useCallback(
    async (assistantMessageId) => {
      const index = state.messages.findIndex((m) => m.id === assistantMessageId);
      if (index < 1) return;

      const priorUser = [...state.messages.slice(0, index)]
        .reverse()
        .find((m) => m.role === "user");
      if (!priorUser) return;

      dispatch({ type: "MESSAGES_TRUNCATED_FROM", id: assistantMessageId });
      await runStream(state.activeId, priorUser.content, assistantMessageId);
    },
    [state.messages, state.activeId, runStream]
  );

  const retryLast = useCallback(async () => {
    const lastUser = [...state.messages].reverse().find((m) => m.role === "user");
    if (!lastUser) return;
    dispatch({ type: "STREAM_CLEAR_ERROR" });
    await runStream(state.activeId, lastUser.content, null);
  }, [state.messages, state.activeId, runStream]);

  const stopGenerating = useCallback(() => {
    abortRef.current?.abort();
  }, []);

  const actions = useMemo(
    () => ({
      retryBoot,
      uploadDocuments,
      dismissUpload,
      removeDocument,
      newConversation,
      selectConversation,
      removeConversation,
      sendMessage,
      regenerate,
      retryLast,
      stopGenerating,
      toast,
    }),
    [
      retryBoot,
      uploadDocuments,
      dismissUpload,
      removeDocument,
      newConversation,
      selectConversation,
      removeConversation,
      sendMessage,
      regenerate,
      retryLast,
      stopGenerating,
      toast,
    ]
  );

  return (
    <StateContext.Provider value={state}>
      <ActionsContext.Provider value={actions}>{children}</ActionsContext.Provider>
    </StateContext.Provider>
  );
}

export function useAppState() {
  const context = useContext(StateContext);
  if (!context) throw new Error("useAppState must be used inside AppProvider");
  return context;
}

export function useActions() {
  const context = useContext(ActionsContext);
  if (!context) throw new Error("useActions must be used inside AppProvider");
  return context;
}
