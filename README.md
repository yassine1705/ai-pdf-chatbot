# AI PDF Chatbot

An AI-powered document assistant that allows users to ask questions about PDF files and receive answers based on their content.

## Features

- Upload and read PDF documents
- Extract PDF text using PyMuPDF
- OCR support for scanned PDFs using Tesseract
- Semantic search using Sentence Transformers
- Generate answers locally using Qwen 2.5 through Ollama
- Display relevant source passages and page numbers
- Maintain conversation history during a session

## Technologies

- Python
- Streamlit
- PyMuPDF
- Sentence Transformers
- Tesseract OCR
- Ollama
- Qwen 2.5
## Installation (Windows)

### 1. Install Python

Install Python 3.10 or newer from the [official Python website](https://www.python.org/downloads/).

During installation, enable **Add Python to PATH**.

### 2. Clone the repository

Open PowerShell and run:

```powershell
git clone https://github.com/yassine1705/ai-pdf-chatbot.git
cd ai-pdf-chatbot
```

### 3. Create a virtual environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, you can run the commands below without activating the environment:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run app.py
```

### 4. Install Python dependencies

```powershell
python -m pip install -r requirements.txt
```

### 5. Install Ollama and the Qwen model

Install [Ollama](https://ollama.com/download), then open PowerShell and run:

```powershell
ollama pull qwen2.5:3b
```

Make sure Ollama is running before launching the application.

### 6. Install Tesseract OCR (optional)

Install [Tesseract OCR for Windows](https://github.com/UB-Mannheim/tesseract/wiki).

For scanned PDFs, install the language data you need and ensure the Tesseract executable path matches the path configured in `app.py`.

### 7. Run the application

From the project folder, run:

```powershell
python -m streamlit run app.py
```

Streamlit will provide a local address, usually `http://localhost:8501`. Open it in your browser.


## How It Works

1. Upload a PDF.
2. Extract its text, using OCR when enabled.
3. Split the text into passages and create embeddings.
4. Retrieve passages relevant to the question.
5. Send the question and retrieved context to the local Qwen model.
6. Display the answer and relevant source passages.

## Notes

- Ollama must be installed and the Qwen model must be available locally.
- OCR requires Tesseract OCR and the appropriate language data.
- The application is designed to run locally.

## Future Improvements

- Improve answer accuracy and source verification
- Support additional document formats
- Improve performance and user interface

## Author

Personal AI and Retrieval-Augmented Generation (RAG) learning project.
