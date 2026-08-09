import uuid
import os
import tempfile
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from app.services.ingestion import ingest_pdf
from app.services.agent import examforge_graph

app = FastAPI()


@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "sessions": len(memory_store),
        "uploaded_sessions": len(uploaded_sessions),
    }


app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "https://exam-forge-ai.vercel.app"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

memory_store: dict[str, list] = {}
uploaded_sessions: set[str] = set()


class AskRequest(BaseModel):
    session_id: str
    question: str


@app.post("/session")
async def create_session():
    session_id = str(uuid.uuid4())
    memory_store[session_id] = []
    return {"session_id": session_id}


@app.post("/upload")
async def upload(session_id: str, file: UploadFile = File(...)):
    if session_id not in memory_store:
        raise HTTPException(status_code=404, detail="Session not found")

    # Validate file type
    if file.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    # Read uploaded file
    file_data = await file.read()

    # Reject empty files
    if not file_data:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")

    # 10 MB upload limit
    max_file_size = 10 * 1024 * 1024

    if len(file_data) > max_file_size:
        raise HTTPException(
            status_code=413,
            detail="PDF file is too large. Maximum allowed size is 10 MB",
        )

    # Create temporary PDF
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        temp_file.write(file_data)
        temp_path = temp_file.name

    try:
        result = ingest_pdf(temp_path, session_id, file.filename)

        uploaded_sessions.add(session_id)

    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

    return result


@app.post("/ask")
async def ask(request: AskRequest):
    session_id = request.session_id

    if session_id not in memory_store:
        raise HTTPException(
            status_code=404, detail="Session not found, upload a PDF first"
        )

    if session_id not in uploaded_sessions:
        raise HTTPException(status_code=400, detail="No PDF uploaded for this session")

    try:
        result = examforge_graph.invoke(
            {
                "session_id": session_id,
                "question": request.question,
                "memory": memory_store[session_id],
                "context": [],
                "answer": "",
                "citations": [],
            }
        )

        memory_store[session_id] = result.get("memory", [])

        return {
            "answer": result.get("answer", ""),
            "citations": result.get("citations", []),
        }

    except Exception as e:
        print(f"ASK ERROR: {type(e).__name__}: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Error processing question: {str(e)}"
        )
