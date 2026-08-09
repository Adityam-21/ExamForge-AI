function QuestionSection({
    question,
    answer,
    askStatus,
    askError,
    isDocumentReady,
    onQuestionChange,
    onAsk,
}) {
    const isLoading = askStatus === "loading";

    function handleKeyDown(event) {
        if (
            event.key === "Enter" &&
            !event.shiftKey &&
            question.trim() &&
            isDocumentReady &&
            !isLoading
        ) {
            event.preventDefault();
            onAsk();
        }
    }

    return (
        <section className="workspace-card question-card">
            <div className="section-heading">
                <div>
                    <span className="step-label">STEP 02</span>
                    <h2>Ask your question</h2>
                </div>

                <span className="section-status">
                    {isDocumentReady ? "Ready to ask" : "Waiting for PDF"}
                </span>
            </div>

            <div className="question-input-wrapper">
                <textarea
                    value={question}
                    onChange={(event) => onQuestionChange(event.target.value)}
                    onKeyDown={handleKeyDown}
                    placeholder={
                        isDocumentReady
                            ? "Ask anything about your uploaded material..."
                            : "Upload and process a PDF first..."
                    }
                    disabled={!isDocumentReady || isLoading}
                    rows="5"
                />

                <div className="question-footer">
                    <span>
                        {isDocumentReady
                            ? "Press Enter to ask · Shift + Enter for a new line"
                            : "Your question box will unlock after the PDF is processed"}
                    </span>

                    <button
                        type="button"
                        className="ask-button"
                        onClick={onAsk}
                        disabled={
                            !isDocumentReady ||
                            !question.trim() ||
                            isLoading
                        }
                    >
                        {isLoading ? "Thinking..." : "Ask AI →"}
                    </button>
                </div>
            </div>

            {askError && (
                <div className="feedback-message error-message">
                    {askError}
                </div>
            )}

            <div className={`answer-panel ${answer ? "has-answer" : ""}`}>
                <div className="answer-panel-header">
                    <div>
                        <span className="answer-label">EXAMFORGE RESPONSE</span>
                        <h3>Your answer</h3>
                    </div>

                    {answer && <span className="answer-status">Grounded in PDF</span>}
                </div>

                <div className="answer-content">
                    {isLoading ? (
                        <div className="answer-placeholder">
                            <span className="loading-dot"></span>
                            ExamForge is searching your study material...
                        </div>
                    ) : answer ? (
                        <p>{answer}</p>
                    ) : (
                        <div className="answer-placeholder">
                            Your answer will appear here after you ask a question.
                        </div>
                    )}
                </div>
            </div>
        </section>
    );
}

export default QuestionSection;