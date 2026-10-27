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

## Installation

1. Install Python, Ollama, and Tesseract OCR if you need scanned PDF support.
2. Install the Python dependencies:

   `pip install -r requirements.txt`

3. Download the model:

   `ollama pull qwen2.5:3b`

## Run the Application

`python -m streamlit run app.py`

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
