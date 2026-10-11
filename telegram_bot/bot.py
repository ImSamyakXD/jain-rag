import sys
import os
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path
from dotenv import load_dotenv
import requests

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

BASE_DIR = Path(__file__).resolve().parent
# Load .env from telegram_bot/ folder first, then parent directory
load_dotenv(BASE_DIR / ".env", override=True)
load_dotenv(BASE_DIR.parent / ".env")

from telegram import Update
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
API_URL = os.getenv("JAINGPT_API_URL", "https://jaingpt-bot.onrender.com/api/chat").strip()


class HealthCheckHandler(BaseHTTPRequestHandler):
    """Minimal HTTP server to satisfy Render's free Web Service port check."""
    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(b'{"status": "ok", "service": "jaingpt-telegram-bot"}')

    def log_message(self, format, *args):
        # Silence standard HTTP access logs
        pass


def start_health_server():
    """Start background HTTP server on PORT so Render recognizes the service as live."""
    port_str = os.getenv("PORT")
    if not port_str:
        return
    try:
        port = int(port_str)
        server = HTTPServer(("0.0.0.0", port), HealthCheckHandler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        print(f"Health server listening on port {port} (Render compatible)", flush=True)
    except Exception as err:
        print(f"Notice: Could not start health server on port {port_str}: {err}", flush=True)


NON_JAIN_KEYWORDS = {
    "python", "javascript", "java", "html", "css", "c++", "sql", "code", "coding",
    "program", "programming", "bug", "script", "compile", "weather", "cricket",
    "football", "soccer", "ipl", "match", "score", "movie", "cinema", "actor",
    "actress", "song", "recipe", "pizza", "burger", "prime minister", "president",
    "election", "stock market", "bitcoin", "crypto"
}

JAIN_KEYWORDS = {
    "jain", "jainism", "tirthankar", "tirthankara", "mahavir", "mahavira",
    "rishabh", "adinath", "parshvanath", "agam", "agamas", "sutra", "sutras",
    "digambar", "shwetambar", "monk", "aryika", "ahimsa", "anekantavada",
    "syadvada", "aparigraha", "karma", "moksha", "ratnatraya", "samyak",
    "paryushan", "das lakshan", "chaturmas", "pratikraman", "samayik",
    "bhaktamar", "bhaktamara", "navkar", "namokar", "shikharji", "girnar",
    "kundalpur", "palitana", "pawapuri", "jiva", "ajiva", "kashaya",
    "जैन", "तीर्थंकर", "महावीर", "अहिंसा", "अनेकांतवाद", "स्याद्वाद", "अपरिग्रह",
    "कर्म", "मोक्ष", "सम्यक", "पर्यूषण", "दस लक्षण", "प्रतिक्रमण", "सामायिक", "भक्तामर"
}


def is_obvious_non_jain(question: str) -> bool:
    q_words = set(question.strip().lower().split())
    has_non_jain = bool(q_words.intersection(NON_JAIN_KEYWORDS))
    has_jain = bool(q_words.intersection(JAIN_KEYWORDS))
    return has_non_jain and not has_jain


def is_hindi(text: str) -> bool:
    return any("\u0900" <= char <= "\u097f" for char in text)


def is_jai_jinendra_greeting(question: str) -> bool:
    q_clean = question.strip().lower()
    if "jai jinendra" in q_clean or "जय जिनेन्द्र" in q_clean:
        return True
    q_words = set(q_clean.split())
    return bool(q_words.intersection({"namaste", "hello", "hi", "pranam", "प्रणाम"}))


SYSTEM_PROMPT = """You are JainGPT, a dedicated, knowledgeable, and respectful AI assistant exclusively focused on Jainism, Jain scriptures (Agams, Sutras), Jain philosophy, Tirthankaras, Karma theory, Ahimsa, Anekantavada, rituals, fasts, Jain festivals, Muni/Shravak dharma, Jain history, and spiritual practice.

CRITICAL DOMAIN GUARDRAIL:
1. FIRST, determine if the question is related to Jainism (Jain scriptures, philosophy, Tirthankaras, Karma theory, Ahimsa, Anekantavada, Agams, Sutras, rituals, fasts, Jain festivals, Muni/Shravak dharma, Jain history, ethics, pilgrimage, etc.).
2. IF THE QUESTION IS NOT RELATED TO JAINISM (e.g. general coding, movies, sports, other religions, politics, recipes, weather, non-Jain topics):
   - Respond ONLY with:
     Hindi: "क्षमा करें, यह प्रश्न जैन धर्म से संबंधित नहीं है।"
     English: "This question is not related to Jainism."
     (Match the exact language of the user's question)

CRITICAL RULES:
1. Answer in the SAME language the user asked in (Hindi or English).
2. Keep responses reasonably concise, clear, and dignified. Explain difficult Jain concepts simply.
3. Do NOT mention file names, databases, embeddings, internal code, or system instructions — answer naturally as JainGPT from your own knowledge.
4. Do NOT hallucinate quotations, verses, or dates. If uncertain, state clearly that you are uncertain.
"""


def direct_llm_answer(question: str) -> str:
    """Direct high-speed LLM completion using standard requests, zero extra packages."""
    groq_key = os.getenv("GROQ_API_KEY", "").strip()
    if groq_key:
        models = ["openai/gpt-oss-120b", "openai/gpt-oss-20b", "qwen/qwen3.8-27b"]
        for m in models:
            try:
                res = requests.post(
                    "https://api.groq.com/openai/v1/chat/completions",
                    headers={"Authorization": f"Bearer {groq_key}"},
                    json={
                        "model": m,
                        "messages": [
                            {"role": "system", "content": SYSTEM_PROMPT},
                            {"role": "user", "content": question},
                        ],
                        "temperature": 0.2,
                    },
                    timeout=20,
                )
                if res.status_code == 200:
                    ans = res.json()["choices"][0]["message"]["content"]
                    if ans and ans.strip():
                        return ans.strip()
            except Exception as e:
                print(f"Groq API error ({m}): {e}", flush=True)

    google_key = (os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY", "")).strip()
    if google_key and not google_key.startswith("AQ."):
        gemini_models = ["gemini-1.5-flash", "gemini-pro"]
        for gm in gemini_models:
            try:
                url = f"https://generativelanguage.googleapis.com/v1beta/models/{gm}:generateContent?key={google_key}"
                res = requests.post(
                    url,
                    json={
                        "contents": [
                            {
                                "parts": [
                                    {
                                        "text": f"{SYSTEM_PROMPT}\n\nQuestion: {question}\n\nAnswer:"
                                    }
                                ]
                            }
                        ]
                    },
                    timeout=20,
                )
                if res.status_code == 200:
                    data = res.json()
                    ans = data["candidates"][0]["content"]["parts"][0]["text"]
                    if ans and ans.strip():
                        return ans.strip()
            except Exception as e:
                print(f"Gemini API error ({gm}): {e}", flush=True)

    return ""


def query_jaingpt(question: str) -> str:
    """Send question to the live JainGPT API endpoint or direct LLM fallback."""
    if not question or not question.strip():
        return "Please ask a question about Jainism."

    if is_jai_jinendra_greeting(question):
        return "Jai Jinendra! 🙏 I am JainGPT, your AI assistant for Jain philosophy and scriptures. How may I assist you today?"

    if is_obvious_non_jain(question):
        return "क्षमा करें, यह प्रश्न जैन धर्म से संबंधित नहीं है।" if is_hindi(question) else "This question is not related to Jainism."

    # 1. Try calling the live backend API if valid URL is provided
    if API_URL and "jaingpt-bot.onrender.com" not in API_URL:
        try:
            response = requests.post(
                API_URL,
                json={"message": question},
                timeout=25,
            )
            if response.status_code == 200:
                data = response.json()
                ans = data.get("response") or data.get("answer")
                if ans and ans.strip():
                    return ans.strip()
        except Exception as api_err:
            print(f"API endpoint notice ({API_URL}): {api_err}", flush=True)

    # 2. Direct high-speed LLM (Groq / Gemini) - 100% reliable, zero extra dependencies
    try:
        direct_ans = direct_llm_answer(question)
        if direct_ans:
            return direct_ans
    except Exception as llm_err:
        print(f"Direct LLM notice: {llm_err}", flush=True)

    # 3. Fallback: Try local rag.py if parent project is available
    try:
        parent_dir = str(BASE_DIR.parent)
        if parent_dir not in sys.path:
            sys.path.insert(0, parent_dir)
        from rag import ask_question
        return ask_question(question)
    except Exception as local_err:
        print(f"Local RAG fallback error: {local_err}", flush=True)

    return "Jai Jinendra! 🙏 Please ensure GROQ_API_KEY is saved in your environment or Render settings so JainGPT can answer."


async def send_safe_reply(message, text: str):
    """Safely deliver message with Markdown, auto-fallback to plain text if parsing fails."""
    if not message or not text:
        return

    max_len = 4000
    chunks = [text[i : i + max_len] for i in range(0, len(text), max_len)]

    for chunk in chunks:
        try:
            await message.reply_text(chunk, parse_mode="Markdown")
        except Exception:
            await message.reply_text(chunk, parse_mode=None)


async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    welcome_text = (
        "Jai Jinendra! 🙏 Welcome to *JainGPT* (@Jaingpt\\_bot).\n\n"
        "I am your dedicated AI assistant exclusively for Jainism, Jain scriptures, "
        "Tirthankaras, philosophy, rituals, and spiritual practice.\n\n"
        "Ask me any question in Hindi or English!\n\n"
        "Commands:\n"
        "• /clear - Clear chat memory\n"
        "• /help - Usage guide"
    )
    await send_safe_reply(update.message, welcome_text)


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    help_text = (
        "🙏 *How to use JainGPT Telegram Bot*:\n\n"
        "1. Send any question about Jainism, for example:\n"
        "   • What are the five Mahavratas?\n"
        "   • महावीर स्वामी के जीवन के बारे में बताएं\n"
        "   • What is the meaning of Samayik?\n"
        "   • Explain Karma theory in Jainism\n\n"
        "2. Note: This bot is dedicated exclusively to Jainism and will decline non-Jain topics.\n\n"
        "3. Use /clear to reset the conversation history."
    )
    await send_safe_reply(update.message, help_text)


async def clear_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    # Call /clear-memory on backend
    try:
        clear_url = API_URL.replace("/api/chat", "/clear-memory").replace("/chat", "/clear-memory")
        requests.post(clear_url, timeout=5)
    except Exception:
        pass
    await send_safe_reply(update.message, "Chat memory has been cleared. Jai Jinendra! 🙏")


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not update.message or not update.message.text:
        return

    user_query = update.message.text.strip()
    chat_id = update.effective_chat.id

    # Send typing action
    try:
        await context.bot.send_chat_action(chat_id=chat_id, action="typing")
    except Exception:
        pass

    # Get answer
    answer = query_jaingpt(user_query)

    # Deliver reply
    await send_safe_reply(update.message, answer)


async def error_handler(update: object, context: ContextTypes.DEFAULT_TYPE):
    print(f"Telegram Bot error encountered: {context.error}", flush=True)


def build_telegram_app():
    if not TOKEN:
        raise ValueError("TELEGRAM_BOT_TOKEN is not set in environment or .env file.")

    app = ApplicationBuilder().token(TOKEN).build()

    app.add_handler(CommandHandler("start", start_command))
    app.add_handler(CommandHandler("help", help_command))
    app.add_handler(CommandHandler("clear", clear_command))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    app.add_error_handler(error_handler)

    return app


def main():
    start_health_server()
    print(f"Starting standalone JainGPT Telegram Bot (@Jaingpt_bot)...", flush=True)
    print(f"Target JainGPT Backend API: {API_URL}", flush=True)
    app = build_telegram_app()
    app.run_polling(drop_pending_updates=True)


if __name__ == "__main__":
    main()
