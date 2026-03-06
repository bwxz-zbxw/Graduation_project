import os
from pypdf import PdfReader

def read_pdf(file_path):
    print(f"--- Reading {os.path.basename(file_path)} ---")
    try:
        reader = PdfReader(file_path)
        full_text = ""
        for page in reader.pages:
            full_text += page.extract_text() + "\n"
        print(full_text)
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
    print("\n" + "="*50 + "\n")

file_path = r"c:\Users\ASUS\Desktop\Graduation_project\文献\西电云平台使用教程.pdf"
read_pdf(file_path)
