import pdfplumber

def extract_text_from_pdf(pdf_path):
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text += page.extract_text()
    return text

# Use relative path for GitHub
pdf_path = "My_Resume.pdf"
text = extract_text_from_pdf(pdf_path)
print(text)
