import os
import fitz  # PyMuPDF for PDF extraction
import pandas as pd
import torch
import faiss
import numpy as np
import time
from transformers import AutoTokenizer, AutoModelForCausalLM

# 🔹 1️⃣ Mount Google Drive to access your trained model
from google.colab import drive
drive.mount('/content/drive')

# 🔹 2️⃣ Define paths
base_dir = "/content/drive/MyDrive/"
pdf_folder = os.path.join(base_dir, "LamaLegalTest1/pdfs)  # Folder containing new PDFs
output_folder = os.path.join(base_dir, "LamaLegalTest1/outputs/")  # Folder to save Excel files
processed_folder = os.path.join(base_dir, "LamaLegalTest1/processed_pdfs/")  # Folder for processed PDFs

# Ensure output and processed folders exist
os.makedirs(output_folder, exist_ok=True)
os.makedirs(processed_folder, exist_ok=True)

# 🔹 3️⃣ Load your custom trained model from Google Drive
trained_model_path = os.path.join(base_dir, "TinyLamaLegalModel/")
tokenizer = AutoTokenizer.from_pretrained(trained_model_path)
model = AutoModelForCausalLM.from_pretrained(trained_model_path, device_map="auto")

# 🔹 4️⃣ Define the fixed questions
questions = [
    "What are the facts of the case?",
    "What was the verdict of the case?",
    "What legal precedents were mentioned?",
    "What are the key arguments in the case?"
]

# 🔹 5️⃣ Function to extract text from a PDF
def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = " ".join([page.get_text() for page in doc])
    return text

# 🔹 6️⃣ Function to generate embeddings from text
def generate_embeddings(texts):
    inputs = tokenizer(texts, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    embeddings = outputs.last_hidden_state.mean(dim=1).detach().cpu().numpy()
    return embeddings

# 🔹 7️⃣ Function to return the most relevant text chunk using FAISS
def retrieve_relevant_text(chunks, question):
    # Create FAISS index
    embeddings = generate_embeddings(chunks)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)

    # Encode the question and search
    question_embedding = generate_embeddings([question])
    distances, indices = index.search(question_embedding, 1)  # Top 1 result

    return chunks[indices[0][0]]

# 🔹 8️⃣ Function to process a single PDF and save answers
def process_pdf_and_generate_answers(pdf_path):
    text = extract_text_from_pdf(pdf_path)

    # Split text into smaller chunks (for FAISS)
    chunk_size = 1000  # Adjust as needed
    chunks = [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]

    # Generate answers for each question
    qa_pairs = []
    for question in questions:
        relevant_text = retrieve_relevant_text(chunks, question)
        qa_pairs.append((question, relevant_text))

    # Save results to Excel
    pdf_name = os.path.basename(pdf_path).replace(".pdf", "")
    output_file = os.path.join(output_folder, f"{pdf_name}_answers.xlsx")
    df = pd.DataFrame(qa_pairs, columns=["Question", "Answer"])
    df.to_excel(output_file, index=False)
    print(f"✅ Answers saved to {output_file}")

# 🔹 9️⃣ Function to process all PDFs and rename them
def process_all_pdfs():
    pdf_files = [f for f in os.listdir(pdf_folder) if f.endswith(".pdf")]

    for pdf_file in pdf_files:
        pdf_path = os.path.join(pdf_folder, pdf_file)
        print(f"📄 Processing: {pdf_file}")

        process_pdf_and_generate_answers(pdf_path)

        # Generate timestamp and rename the PDF
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        new_pdf_name = f"{os.path.splitext(pdf_file)[0]}_processed_{timestamp}.pdf"
        new_pdf_path = os.path.join(processed_folder, new_pdf_name)

        os.rename(pdf_path, new_pdf_path)
        print(f"✅ Moved and renamed to: {new_pdf_path}")

# 🔹 🔟 Run the program for all PDFs in the folder
process_all_pdfs()
