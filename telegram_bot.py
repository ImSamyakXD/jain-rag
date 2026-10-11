import sys
import os
import asyncio
from pathlib import Path
from dotenv import load_dotenv

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env", override=True)

from telegram import Update
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

from rag import ask_question, clear_memory

TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "8902525488:AAGrU6eNvzJgPmDSyZ3h1EoMe7aQPhuVDVY").strip()


async def send_safe_reply(message, text: str):
    """Safely send replies with Markdown formatting and auto-fallback to plain text."""
    if not message or not text:
        return

    max_len = 4000
    chunks = [text[i : i + max_len] for i in range(0, len(text), max_len)]

    for chunk in chunks:
        try:
            await message.reply_text(chunk, parse_mode="Markdown")
        except Exception:
            # Fall back to plain text if Markdown parsing fails (e.g. unescaped entities)
            await message.reply_text(chunk, parse_mode=None)


async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    welcome_text = (
        "Jai Jinendra! 🙏 Welcome to *JainGPT* (@Jaingpt\\_bot).\n\n"
        "I am your dedicated AI assistant exclusively for Jainism, Jain scriptures, "
        "Tirthankaras, philosophy, rituals, and spiritual practice.\n\n"
        "Ask me any question in Hindi or English!\n\n"
        "Commands:\n"
        "• /clear - Reset chat memory\n"
        "• /help - Usage guide"
    )
    await send_safe_reply(update.message, welcome_text)


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    help_text = (
        "🙏 *How to use JainGPT Telegram Bot*:\n\n"
        "1. Simply send any question about Jainism, for example:\n"
        "   • What are the five Mahavratas?\n"
        "   • महावीर स्वामी के जीवन के बारे में बताएं\n"
        "   • What is the meaning of Samayik?\n"
        "   • Explain Karma theory in Jainism\n\n"
        "2. Note: This bot is dedicated exclusively to Jainism and will decline non-Jain topics.\n\n"
        "3. Use /clear to reset the conversation history."
    )
    await send_safe_reply(update.message, help_text)


async def clear_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    clear_memory()
    await send_safe_reply(update.message, "Chat memory has been cleared. Jai Jinendra! 🙏")


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not update.message or not update.message.text:
        return

    user_query = update.message.text.strip()
    chat_id = update.effective_chat.id

    # Send typing action indicator
    try:
        await context.bot.send_chat_action(chat_id=chat_id, action="typing")
    except Exception:
        pass

    # Generate answer using JainGPT logic from rag.py
    answer = ask_question(user_query)

    # Deliver reply safely
    await send_safe_reply(update.message, answer)


async def error_handler(update: object, context: ContextTypes.DEFAULT_TYPE):
    """Log errors caused by updates."""
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


def start_bot_thread():
    """Start Telegram bot in a daemon thread (e.g. inside FastAPI startup)."""
    import threading

    def _run():
        try:
            print("Starting JainGPT Telegram Bot (@Jaingpt_bot)...", flush=True)
            app = build_telegram_app()
            app.run_polling(drop_pending_updates=True)
        except Exception as e:
            print(f"Telegram Bot thread notice: {e}", flush=True)

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()
    return thread


def main():
    print(f"Starting JainGPT Telegram Bot (@Jaingpt_bot)...", flush=True)
    app = build_telegram_app()
    app.run_polling(drop_pending_updates=True)


if __name__ == "__main__":
    main()
