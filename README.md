# jain-rag (JainDarshan AI)

A bilingual (Hindi/English) RAG chatbot for Jain scriptures. Answers
in whichever language you ask in, grounded only in the documents you
provide — it won't invent verses or teachings.

Built for SCANNED PDFs of Agams/Sutras: pages with no selectable
text are automatically run through OCR (Hindi + English).

## 1. Install system dependencies (required for OCR)

This is the part that trips people up — `pip install` alone is NOT
enough, because OCR needs two external programs on your system.

**Windows:**
1. Install Tesseract OCR: https://github.com/UB-Mannheim/tesseract/wiki
   - During setup, tick **Hindi** under additional language data.
   - Default install path is usually
     `C:\Program Files\Tesseract-OCR\tesseract.exe`
2. Install Poppler for Windows:
   https://github.com/oschwartz10612/poppler-windows/releases
   - Download, extract it anywhere (e.g. `C:\poppler-24.x\`)
3. Open `ingest.py` and set the paths near the top if they're not on
   your PATH:
   ```python
   pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
   POPPLER_PATH = r"C:\poppler-24.x\Library\bin"
   ```

**macOS:**
```bash
brew install tesseract tesseract-lang poppler
```

**Ubuntu/Debian/WSL:**
```bash
sudo apt-get install tesseract-ocr tesseract-ocr-hin poppler-utils
```

Check it worked:
```bash
tesseract --list-langs
```
You should see `hin` in the list. If not, the Hindi language pack
didn't install correctly and OCR will silently fail on Hindi text.

## 2. Set up the Python environment

```bash
python3.11 -m venv venv

# Windows PowerShell
venv\Scripts\Activate.ps1
# macOS/Linux
source venv/bin/activate

pip install -r requirements.txt
```

## 3. Add your Gemini API key

Create a `.env` file in this folder:
```
GOOGLE_API_KEY=your_key_here
```
Get a free key at https://aistudio.google.com/apikey

## 4. Add your scripture PDFs

```
data/agams/          -> Acharanga, Sutrakritanga, etc.
data/sutras/          -> Tattvartha Sutra, etc.
data/commentaries/    -> bhashya/tika/scholarly commentary
data/general/         -> anything else (Tirthankar bios, etc.)
```

## 5. Build the knowledge base

```bash
python ingest.py
```

This will print, page by page, which pages needed OCR — expect this
to be slow for scanned PDFs (OCR takes a few seconds per page). A
100-page scanned Agam might take 10-20 minutes depending on your
machine. Grab chai.

## 6. Run the bot

```bash
uvicorn app:app --reload
```

Open http://127.0.0.1:8000 and ask in Hindi or English.

## Notes on accuracy

- OCR on scanned Devanagari text is genuinely harder than English —
  expect some recognition errors, especially with complex conjuncts
  (संयुक्ताक्षर) or worn/faded scans. The bot is instructed to flag
  uncertain-looking passages rather than confidently repeat OCR
  errors, but it's not foolproof.
- For anything you plan to publish or rely on precisely (exact verse
  wording, citations), treat the bot's output as a starting point
  and verify against the original scripture — this matters
  especially for religious/scriptural accuracy.
- If a question's answer isn't in your knowledge base, the bot says
  so rather than answering from general training knowledge — this is
  intentional, so it doesn't blend unverified outside claims with
  your actual source texts.
