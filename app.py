import asyncio
import threading
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from rag import ask_question, clear_memory, get_base_retriever, get_llm


BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
TEMPLATES_DIR = BASE_DIR / "templates"


app = FastAPI(title="JainGPT AI")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

if STATIC_DIR.exists():
    app.mount(
        "/static",
        StaticFiles(directory=str(STATIC_DIR)),
        name="static"
    )

templates = Jinja2Templates(
    directory=str(TEMPLATES_DIR)
)


@app.on_event("startup")
async def startup_event():
    def prewarm():
        try:
            print("Pre-warming JainGPT AI service...", flush=True)
            get_base_retriever()
            get_llm()
            print("JainGPT AI service ready!", flush=True)
        except Exception as e:
            print(f"Pre-warm notice: {e}", flush=True)
    threading.Thread(target=prewarm, daemon=True).start()


class ChatRequest(BaseModel):
    message: str


from datetime import datetime, timezone

@app.get("/health")
async def health():
    return {
        "success": True,
        "status": "ok",
        "timestamp": datetime.now(timezone.utc).isoformat()
    }


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="index.html"
    )


@app.post("/chat")
async def chat(request: ChatRequest):
    answer = await asyncio.to_thread(ask_question, request.message)
    return {
        "answer": answer
    }


@app.post("/api/chat")
async def api_chat(request: ChatRequest):
    """Clean API endpoint for website integration."""
    answer = await asyncio.to_thread(ask_question, request.message)
    return {
        "response": answer,
        "answer": answer
    }


@app.post("/clear-memory")
async def clear_chat_memory():
    clear_memory()
    return {
        "message": "Memory cleared"
    }
