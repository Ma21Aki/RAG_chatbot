# utils/pdf_loader.py
import pdfplumber

def extract_text_from_pdfs(pdf_paths):
    all_texts = []
    for path in pdf_paths:
        with pdfplumber.open(path) as pdf:
            for page in pdf.pages:
                text = page.extract_text()
                if text:
                    all_texts.append(text)
    return "\n".join(all_texts)

