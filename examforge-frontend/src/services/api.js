const API_BASE_URL = "http://127.0.0.1:8000";

export async function createSession() {
    const response = await fetch(`${API_BASE_URL}/session`, {
        method: "POST",
    });

    if (!response.ok) {
        throw new Error("Failed to create session");
    }

    return response.json();
}

export async function uploadFile(sessionId, file) {
    const formData = new FormData();
    formData.append("file", file);

    const response = await fetch(
        `${API_BASE_URL}/upload?session_id=${encodeURIComponent(sessionId)}`,
        {
            method: "POST",
            body: formData,
        }
    );

    if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.detail || "Failed to upload file");
    }

    return response.json();
}

export async function askQuestion(sessionId, question) {
    const response = await fetch(`${API_BASE_URL}/ask`, {
        method: "POST",
        headers: {
            "Content-Type": "application/json",
        },
        body: JSON.stringify({
            session_id: sessionId,
            question: question,
        }),
    });

    if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.detail || "Failed to get an answer");
    }

    return response.json();
}