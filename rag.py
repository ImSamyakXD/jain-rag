import sys
import os
from pathlib import Path
from dotenv import load_dotenv

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

conversation_history = []
_llm = None

# Fast pre-filter lists for 0.00s non-Jain rejection
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
    return any('\u0900' <= char <= '\u097f' for char in text)


def get_llms():
    import os
    from dotenv import load_dotenv
    load_dotenv(override=True)

    candidates = []
    groq_key = os.getenv("GROQ_API_KEY")
    google_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")

    if groq_key and groq_key.strip() and not groq_key.strip().startswith("your_"):
        try:
            from langchain_groq import ChatGroq
            for gm in ["groq/compound-mini", "groq/compound", "openai/gpt-oss-20b"]:
                candidates.append(ChatGroq(
                    model_name=gm,
                    groq_api_key=groq_key.strip(),
                    temperature=0
                ))
        except Exception as e:
            print(f"Groq init notice: {e}")

    if google_key and google_key.strip() and not google_key.strip().startswith("your_"):
        gemini_models = ["gemini-flash-latest", "gemini-1.5-flash", "gemini-pro"]
        for m in gemini_models:
            candidates.append(ChatGoogleGenerativeAI(
                model=m,
                google_api_key=google_key.strip(),
                temperature=0
            ))

    return candidates


def get_llm():
    global _llm
    if _llm is None:
        llms = get_llms()
        if llms:
            _llm = llms[0]
    return _llm


def get_base_retriever():
    """Stub retriever for app.py startup pre-warm compatibility."""
    return None


def is_jai_jinendra_greeting(question: str) -> bool:
    q_clean = question.strip().lower()
    if "jai jinendra" in q_clean or "जय जिनेन्द्र" in q_clean:
        return True
    q_words = set(q_clean.split())
    return bool(q_words.intersection({"namaste", "hello", "hi", "pranam", "प्रणाम"}))


direct_jainism_prompt = PromptTemplate.from_template(
    """
You are JainGPT, a dedicated, knowledgeable, and respectful AI assistant exclusively focused on Jainism, Jain scriptures (Agams, Sutras), Jain philosophy, Tirthankaras, Karma theory, Ahimsa, Anekantavada, rituals, fasts, Jain festivals, Muni/Shravak dharma, Jain history, and spiritual practice.

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

Question:
{question}

Answer:
"""
)


def ask_question(question: str) -> str:
    if not question or not question.strip():
        return "Please ask a question about Jainism."

    if is_jai_jinendra_greeting(question):
        custom_reply = "Jai Jinendra! 🙏 I am JainGPT, your AI assistant for Jain philosophy and scriptures. How may I assist you today?"
        conversation_history.append({
            "question": question,
            "answer": custom_reply
        })
        return custom_reply

    # Fast pre-filter for obvious non-Jain queries (0.00s)
    if is_obvious_non_jain(question):
        refusal = "क्षमा करें, यह प्रश्न जैन धर्म से संबंधित नहीं है।" if is_hindi(question) else "This question is not related to Jainism."
        conversation_history.append({
            "question": question,
            "answer": refusal
        })
        return refusal

    answer = None
    last_err = None
    llm_candidates = get_llms()

    for candidate_llm in llm_candidates:
        try:
            chain = direct_jainism_prompt | candidate_llm | StrOutputParser()
            res = chain.invoke({"question": question})
            if res and res.strip():
                answer = res.strip()
                break
        except Exception as llm_err:
            try:
                print(f"LLM candidate execution notice: {llm_err}")
            except Exception:
                pass
            last_err = llm_err

    if not answer:
        if not llm_candidates:
            return "Jai Jinendra! 🙏 No active API key found. Please ensure GROQ_API_KEY (gsk_...) or GOOGLE_API_KEY (AIzaSy...) is saved in your .env file."
        return f"Jai Jinendra! 🙏 AI service notice: {last_err if last_err else 'Unable to generate response'}. Please ensure your API key in .env is active and saved."

    conversation_history.append({
        "question": question,
        "answer": answer
    })

    if len(conversation_history) > 10:
        conversation_history.pop(0)

    return answer


def clear_memory():
    conversation_history.clear()