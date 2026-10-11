import sys
import os
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


def query_jaingpt(question: str) -> str:
    """Send question to the live JainGPT API endpoint, with fallback to local rag.py if offline."""
    # 1. Try calling the live Render / FastAPI backend
    try:
        response = requests.post(
            API_URL,
            json={"message": question},
            timeout=30,
        )
        if response.status_code == 200:
            data = response.json()
            return data.get("response") or data.get("answer") or "No response received."
    except Exception as api_err:
        print(f"API endpoint notice ({API_URL}): {api_err}", flush=True)

    # 2. Fallback: Try local rag.py if parent project is available
    try:
        parent_dir = str(BASE_DIR.parent)
        if parent_dir not in sys.path:
            sys.path.insert(0, parent_dir)
        from rag import ask_question
        return ask_question(question)
    except Exception as local_err:
        print(f"Local RAG fallback error: {local_err}", flush=True)
        return "⏳ JainGPT service is currently connecting. Please try again in a moment!"


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
    print(f"Starting standalone JainGPT Telegram Bot (@Jaingpt_bot)...", flush=True)
    print(f"Target JainGPT Backend API: {API_URL}", flush=True)
    app = build_telegram_app()
    app.run_polling(drop_pending_updates=True)


if __name__ == "__main__":
    main()
