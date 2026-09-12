from pathlib import Path
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser


BASE_DIR = Path(__file__).resolve().parent
CHROMA_DIR = BASE_DIR / "chroma_db_v2"

load_dotenv()

_embeddings = None
_llm = None
_vectorstore = None
_base_retriever = None

def get_llm():
    global _llm
    if _llm is None:
        _llm = ChatGoogleGenerativeAI(
            model="gemini-flash-latest",
            temperature=0
        )
    return _llm

def get_base_retriever():
    global _embeddings, _vectorstore, _base_retriever
    if _base_retriever is None:
        try:
            _embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
            )
            if CHROMA_DIR.exists():
                _vectorstore = Chroma(
                    collection_name="jain_knowledge_v1",
                    embedding_function=_embeddings,
                    persist_directory=str(CHROMA_DIR)
                )
                _base_retriever = _vectorstore.as_retriever(
                    search_kwargs={"k": 6}
                )
        except Exception as e:
            print(f"Vectorstore init notice: {e}")
            _base_retriever = None
    return _base_retriever

conversation_history = []

PRONOUNS = {"yeh", "woh", "iska", "uska", "isne", "usne", "it", "this", "that", "they", "he", "she", "his", "her", "their", "these", "those", "isey", "usey", "यह", "वह", "इसका", "उसका", "इसने", "उसने"}

def needs_rewrite(question: str) -> bool:
    words = set(question.lower().split())
    return bool(words.intersection(PRONOUNS))

rewrite_prompt = PromptTemplate.from_template(
    """
You are helping a Jain scripture assistant understand follow-up questions, which may be in Hindi, English, or a mix of both.

Conversation history:
{history}

Current question:
{question}

Rewrite the current question so it is completely understandable on its own, resolving pronouns (yeh, woh, iska, uska, it, this, that, etc.) using the conversation history.

Keep the rewritten question in the SAME language as the original question.
If the question is already complete, keep it essentially unchanged.

Return ONLY the rewritten question.
"""
)

direct_jainism_prompt = PromptTemplate.from_template(
    """
You are JainGPT, a dedicated and respectful assistant exclusively for Jainism, Jain scriptures (Agams, Sutras), Jain philosophy, Tirthankaras, rituals, history, and practice.

CRITICAL DOMAIN GUARDRAIL:
1. FIRST, check if the question is related to Jainism (Jain scriptures, philosophy, Tirthankaras, Karma theory, Ahimsa, Anekantavada, Agams, Sutras, rituals, fasts, Jain festivals, Muni/Shravak dharma, Jain history, etc.).
2. IF THE QUESTION IS NOT RELATED TO JAINISM (e.g. general coding, movies, sports, other religions, politics, recipes, weather, non-Jain topics):
   - Decline strictly with:
     Hindi: "क्षमा करें, यह प्रश्न जैन धर्म या जैन दर्शन से संबंधित नहीं है। मैं केवल जैन धर्म और दर्शन से जुड़े प्रश्नों के उत्तर देने के लिए समर्पित हूँ।"
     English: "I apologize, but this question is not related to Jainism or Jain philosophy. I am dedicated exclusively to answering questions related to Jainism."

CRITICAL LANGUAGE RULE:
Answer in the SAME language the user asked in (Hindi or English).

STRICT RULES:
1. Do NOT mention file names, file extensions (.pdf, .docx, .txt), or internal database names in your response.
2. Provide a clear, dignified, authentic, and structured explanation of the Jain principle or topic.
3. If context is provided below, synthesize and incorporate it cleanly.

Context:
{context}

Question:
{question}

Answer:
"""
)


def format_docs(docs):
    formatted_chunks = []
    for doc in docs:
        content = doc.page_content.strip()
        if content:
            category = doc.metadata.get('category', 'general').capitalize()
            formatted_chunks.append(f"Category: {category}\nContent: {content}")
    return "\n\n---\n\n".join(formatted_chunks)


def search_web_fallback(query: str) -> str:
    try:
        from duckduckgo_search import DDGS
        results = list(DDGS().text(f"Jainism {query}", max_results=4))
        if not results:
            return ""
        formatted = []
        for r in results:
            title = r.get("title", "")
            body = r.get("body", "")
            if body:
                formatted.append(f"Source Title: {title}\nSummary: {body}")
        return "\n\n".join(formatted)
    except Exception as e:
        print(f"Web search error: {e}")
        return ""


def rewrite_question(question, history):
    if not history or not needs_rewrite(question):
        return question

    history_text = "\n".join(
        f"User: {item['question']}\n"
        f"Assistant: {item['answer']}"
        for item in history[-5:]
    )

    rewritten = (
        rewrite_prompt
        | get_llm()
        | StrOutputParser()
    ).invoke({
        "history": history_text,
        "question": question
    })

    return rewritten.strip()


def is_jai_jinendra_greeting(q: str) -> bool:
    cleaned = q.strip().lower().replace("!", "").replace(".", "").replace(",", "").replace("?", "")
    greetings = {
        "jai jinendra", "jai jinendraa", "jai jinendram", "jai jinendr",
        "jai jinendarr", "जय जिनेन्द्र", "जय जिनेन्द्र!", "जय जिनेंद्र"
    }
    return cleaned in greetings or cleaned == "jai jinendra" or cleaned == "जय जिनेन्द्र"


FAST_TOPIC_ANSWERS = {
    "पंच महाव्रत": """**पंच महाव्रत (Five Great Vows of Jainism):**

1. **अहिंसा महाव्रत (Ahimsa)** — मन, वचन और काया से किसी भी जीव को लेशमात्र भी कष्ट न देना।
2. **सत्य महाव्रत (Satya)** — सर्वथा क्रोध, लोभ, भय, हास्य रहित होकर निर्दोष सत्य बोलना।
3. **अस्तेय महाव्रत (Asteya)** — बिना दी गई किसी भी वस्तु को ग्रहण न करना (चोरी का त्याग)।
4. **ब्रह्मचर्य महाव्रत (Brahmacharya)** — सर्व प्रकार के मैथुन और विषय-वासना का पूर्ण त्याग करना।
5. **अपरिग्रह महाव्रत (Aparigraha)** — धन, धान्य, वस्त्र, मकान आदि सर्व परिग्रह का त्याग करना।""",

    "अनेकांतवाद": """**अनेकांतवाद (Anekantavada — Theory of Non-Absolutism):**

अनेकांतवाद जैन दर्शन की वह अनुपम देन है जो सिखाती है कि परम सत्य के अनेक पहलू होते हैं।

- **मूल सिद्धांत**: किसी वस्तु या घटना को केवल एक दृष्टिकोण से देखकर अंतिम निर्णय नहीं लेना चाहिए।
- **स्याद्वाद (Syadvada)**: भाषिक अभिव्यक्ति में "स्यात्" (किसी अपेक्षा से) पद का प्रयोग करके सापेक्ष सत्य को प्रकट किया जाता है।
- **सहिष्णुता**: यह सिद्धांत वैचारिक अहिंसा और दूसरों के दृष्टिकोण के प्रति सम्मान और सहिष्णुता का मार्ग प्रशस्त करता है।""",

    "karma theory": """**Jain Karma Theory (कर्म सिद्धांत):**

In Jainism, Karma is not a divine reward/punishment system, but a subtle physical matter (**Karma Varganas**) that attaches to the Soul (**Jiva**) due to passions (**Kashayas** like anger, pride, deceit, greed).

1. **Main Types**:
   - **Ghatiya Karmas** (Harming soul's natural qualities: Knowledge, Perception, Bliss, Energy).
   - **Aghatiya Karmas** (Determining physical body, lifespan, status, feelings).
2. **Shedding Karmas (Nirjara)**: Through austerity (Tapa), meditation, self-control (Samyama), and devotion, the soul sheds all karmic particles to achieve **Moksha** (Liberation).""",

    "तीर्थंकर": """**चौबीस तीर्थंकर (24 Tirthankaras):**

जैन धर्म में धर्म-तीर्थ (संसार-समुद्र को पार कराने वाला धर्म का मार्ग) का प्रवर्त्तन करने वाले सर्वज्ञ, वीतराग पुरुष को **तीर्थंकर** कहते हैं।

- **प्रथम तीर्थंकर**: भगवान **ऋषभदेव (आदिनाथ)**
- **२३वें तीर्थंकर**: भगवान **पार्श्वनाथ स्वामी**
- **२४वें तीर्थंकर**: भगवान **महावीर स्वामी** (वर्तमान शासन नायक)

प्रत्येक तीर्थंकर के ५ कल्याणक (गर्भ, जन्म, दीक्षा, ज्ञान, मोक्ष) मनाए जाते हैं।""",

    "ahimsa": """**Ahimsa in Jain Philosophy (अहिंसा):**

**"Ahimsā Paramo Dharmaḥ"** (Non-violence is the supreme spiritual duty).

- **Definition**: Avoiding injury to any living being (**Jiva**) through **Mind (Manas), Speech (Vachana), or Body (Kaya)** — directly doing it, causing others to do it, or consenting to it (*Krita, Karita, Anumodita*).
- **Compassion for all beings**: Applies to humans, animals, plants, insects, and even micro-organisms (Ekendriya Jivas).
- It is the foundation of Jain ethics, diet, and liberation."""
}


def ask_question(question):
    if is_jai_jinendra_greeting(question):
        custom_reply = "Jai Jinendra! 🙏 I am JainGPT, your scriptural AI assistant. I can answer your questions on Jainism, Agams, philosophy, and daily practice!"
        conversation_history.append({
            "question": question,
            "answer": custom_reply
        })
        return custom_reply

    q_clean = question.strip().lower()
    for topic_key, fast_ans in FAST_TOPIC_ANSWERS.items():
        if topic_key in q_clean or q_clean in topic_key:
            conversation_history.append({
                "question": question,
                "answer": fast_ans
            })
            return fast_ans

    standalone_question = rewrite_question(
        question,
        conversation_history
    )

    # 1. Check local ChromaDB vector store if available
    context = ""
    retriever = get_base_retriever()
    if retriever:
        try:
            docs = retriever.invoke(standalone_question)
            context = format_docs(docs)
        except Exception as e:
            print(f"Retriever notice: {e}")

    # 2. If local docs context is sparse, perform instant web search fallback
    if not context.strip():
        print(f"Performing web search for Jainism query: '{standalone_question}'")
        context = search_web_fallback(standalone_question)

    if not context.strip():
        context = "No specific external documents retrieved."

    # 3. Generate answer using direct_jainism_prompt with domain guardrail
    answer = (
        direct_jainism_prompt
        | get_llm()
        | StrOutputParser()
    ).invoke({
        "context": context,
        "question": standalone_question
    }).strip()

    conversation_history.append({
        "question": question,
        "answer": answer
    })

    if len(conversation_history) > 10:
        conversation_history.pop(0)

    return answer


def clear_memory():
    conversation_history.clear()
