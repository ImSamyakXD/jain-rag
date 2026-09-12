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


app = FastAPI(title="RealJainism AI")

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
            print("Starting background pre-warm of Vectorstore & LLM...", flush=True)
            get_base_retriever()
            get_llm()
            print("Pre-warm successfully completed!", flush=True)
        except Exception as e:
            print(f"Pre-warm notice: {e}", flush=True)
    threading.Thread(target=prewarm, daemon=True).start()


class ChatRequest(BaseModel):
    message: str


@app.get("/health")
async def health():
    return {"status": "ok", "service": "RealJainism AI RAG"}


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="index.html"
    )


@app.post("/chat")
async def chat(request: ChatRequest):
    # Run synchronous heavy ML/API logic in a threadpool to prevent blocking the event loop
    answer = await asyncio.to_thread(ask_question, request.message)

    return {
        "answer": answer
    }


@app.post("/clear-memory")
async def clear_chat_memory():
    clear_memory()

    return {
        "message": "Memory cleared"
    }
