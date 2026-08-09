import { useEffect, useState } from "react";
import "./App.css";

import Header from "./components/Header";
import UploadSection from "./components/UploadSection";
import QuestionSection from "./components/QuestionSection";
import Footer from "./components/Footer";

import {
  askQuestion,
  createSession,
  uploadFile,
} from "./services/api";

function App() {
  const [sessionId, setSessionId] = useState("");
  const [sessionError, setSessionError] = useState("");

  const [selectedFile, setSelectedFile] = useState(null);
  const [uploadStatus, setUploadStatus] = useState("idle");
  const [uploadMessage, setUploadMessage] = useState("");

  const [question, setQuestion] = useState("");
  const [answer, setAnswer] = useState("");
  const [askStatus, setAskStatus] = useState("idle");
  const [askError, setAskError] = useState("");

  useEffect(() => {
    async function initializeSession() {
      try {
        const data = await createSession();

        setSessionId(data.session_id);
        setSessionError("");
      } catch (error) {
        console.error(error);

        setSessionError(
          "Unable to connect to the backend. Please make sure the backend server is running."
        );
      }
    }

    initializeSession();
  }, []);

  function handleFileChange(event) {
    const file = event.target.files?.[0];

    if (!file) {
      return;
    }

    if (file.type !== "application/pdf") {
      setSelectedFile(null);
      setUploadStatus("error");
      setUploadMessage("Please select a valid PDF file.");
      return;
    }

    setSelectedFile(file);
    setUploadStatus("idle");
    setUploadMessage("");
    setAnswer("");
    setAskError("");
  }

  async function handleUpload() {
    if (!selectedFile || !sessionId) {
      return;
    }

    try {
      setUploadStatus("uploading");
      setUploadMessage("");

      const data = await uploadFile(sessionId, selectedFile);

      setUploadStatus("success");

      setUploadMessage(
        data.message ||
        "Your document has been processed successfully. You can now ask questions."
      );
    } catch (error) {
      console.error(error);

      setUploadStatus("error");
      setUploadMessage(
        error.message || "Something went wrong while uploading the PDF."
      );
    }
  }

  async function handleAsk() {
    if (!question.trim() || !sessionId) {
      return;
    }

    try {
      setAskStatus("loading");
      setAskError("");
      setAnswer("");

      const data = await askQuestion(sessionId, question.trim());

      setAnswer(data.answer || "No answer was returned.");
      setAskStatus("success");
    } catch (error) {
      console.error(error);

      setAskStatus("error");
      setAskError(
        error.message || "Unable to generate an answer. Please try again."
      );
    }
  }

  const isDocumentReady = uploadStatus === "success";

  return (
    <div className="app-shell">
      <Header />

      <main className="main-content">
        <section className="hero-section">
          <div className="hero-content">
            <span className="eyebrow">RAG-POWERED STUDY ASSISTANT</span>

            <h1>
              Turn your study material into
              <span> instant answers.</span>
            </h1>

            <p>
              Upload your exam material, ask questions in natural language,
              and get answers grounded in your document.
            </p>
          </div>

          <div className="hero-stats">
            <div className="stat-card">
              <strong>PDF</strong>
              <span>Upload notes</span>
            </div>

            <div className="stat-card">
              <strong>AI</strong>
              <span>Context-aware answers</span>
            </div>

            <div className="stat-card">
              <strong>RAG</strong>
              <span>Document-grounded</span>
            </div>
          </div>
        </section>

        {sessionError && (
          <div className="global-error">
            <strong>Connection issue:</strong> {sessionError}
          </div>
        )}

        <div className="workspace">
          <UploadSection
            selectedFile={selectedFile}
            uploadStatus={uploadStatus}
            uploadMessage={uploadMessage}
            onFileChange={handleFileChange}
            onUpload={handleUpload}
            disabled={!sessionId}
          />

          <QuestionSection
            question={question}
            answer={answer}
            askStatus={askStatus}
            askError={askError}
            isDocumentReady={isDocumentReady}
            onQuestionChange={setQuestion}
            onAsk={handleAsk}
          />
        </div>
      </main>

      <Footer />
    </div>
  );
}

export default App;