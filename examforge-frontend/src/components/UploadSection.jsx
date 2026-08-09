function UploadSection({
    selectedFile,
    uploadStatus,
    uploadMessage,
    onFileChange,
    onUpload,
    disabled,
}) {
    const isUploading = uploadStatus === "uploading";

    return (
        <section className="workspace-card upload-card">
            <div className="section-heading">
                <div>
                    <span className="step-label">STEP 01</span>
                    <h2>Upload study material</h2>
                </div>

                <span className="section-status">
                    {uploadStatus === "success" ? "Ready" : "PDF only"}
                </span>
            </div>

            <label
                className={`upload-dropzone ${selectedFile ? "has-file" : ""
                    } ${isUploading ? "is-uploading" : ""}`}
            >
                <input
                    type="file"
                    accept=".pdf,application/pdf"
                    onChange={onFileChange}
                    disabled={isUploading || disabled}
                />

                <div className="upload-icon">↑</div>

                {!selectedFile ? (
                    <div className="upload-copy">
                        <strong>Choose a PDF or drop it here</strong>
                        <span>Your notes, syllabus, or study material</span>
                    </div>
                ) : (
                    <div className="file-content">
                        <strong>{selectedFile.name}</strong>
                        <span>
                            {(selectedFile.size / 1024 / 1024).toFixed(2)} MB · PDF selected
                        </span>
                    </div>
                )}
            </label>

            <div className="upload-action-row">
                <div className="upload-hint">
                    {uploadStatus === "success"
                        ? "Your document is ready for questions."
                        : "Upload one PDF to begin."}
                </div>

                <button
                    type="button"
                    className="primary-button"
                    onClick={onUpload}
                    disabled={
                        !selectedFile ||
                        disabled ||
                        isUploading ||
                        uploadStatus === "success"
                    }
                >
                    {isUploading
                        ? "Processing..."
                        : uploadStatus === "success"
                            ? "Document Ready"
                            : "Process PDF"}
                </button>
            </div>

            {uploadMessage && (
                <div
                    className={`feedback-message ${uploadStatus === "error" ? "error-message" : "success-message"
                        }`}
                >
                    {uploadMessage}
                </div>
            )}
        </section>
    );
}

export default UploadSection;