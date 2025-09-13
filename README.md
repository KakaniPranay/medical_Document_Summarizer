# Comprehensive Medical Document Summarization System

## Setup (Linux / macOS / Windows WSL recommended)
1. Create virtualenv:
   python -m venv venv
   source venv/bin/activate   # or venv\Scripts\activate on Windows

2. Install requirements:
   pip install -r requirements.txt

3. Download NLTK punkt:
   python -c "import nltk; nltk.download('punkt')"

4. (Optional but recommended) Install spaCy small model:
   python -m spacy download en_core_web_sm

5. Run the app:
   python app.py

6. Open http://127.0.0.1:5000 in your browser.

### Notes
- For best abstractive summaries you can provide an OpenAI API key as environment variable `OPENAI_API_KEY`; the app will use it automatically.
- If no OpenAI key and transformers + torch are installed, the app will use a transformers summarization pipeline.
- If neither are available or memory is constrained, the app falls back to TextRank extractive summary only.
