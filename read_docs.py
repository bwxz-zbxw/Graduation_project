import docx
import os

files = [
    r"c:\Users\ASUS\Desktop\Graduation_project\文献\毕设计划书-周博文.docx",
    r"c:\Users\ASUS\Desktop\Graduation_project\文献\开题报告.docx",
    r"c:\Users\ASUS\Desktop\Graduation_project\文献\毕设进度说明.docx"
]

def read_docx(file_path):
    print(f"--- Reading {os.path.basename(file_path)} ---")
    try:
        doc = docx.Document(file_path)
        full_text = []
        for para in doc.paragraphs:
            full_text.append(para.text)
        print("\n".join(full_text))
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
    print("\n" + "="*50 + "\n")

for f in files:
    read_docx(f)
