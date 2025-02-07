
import os
import fitz  # PyMuPDF for PDF extraction
import pandas as pd
import torch
import faiss
import numpy as np
import time
import gc  # Garbage Collection
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm  # Progress Bar
from google.colab import drive
import shutil

# 🔹 1️⃣ Free GPU Memory (If Available)
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
gc.collect()

# 🔹 2️⃣ Mount Google Drive
drive.flush_and_unmount()
shutil.rmtree('/content/drive', ignore_errors=True)
drive.mount('/content/drive', force_remount=True)

# 🔹 3️⃣ Define Paths
base_dir = "/content/drive/MyDrive/"
pdf_folder = os.path.join(base_dir, "LamaLegalTest1/pdfs")
output_folder = os.path.join(base_dir, "LamaLegalTest1/outputs/")
processed_folder = os.path.join(base_dir, "LamaLegalTest1/processed_pdfs/")

os.makedirs(output_folder, exist_ok=True)
os.makedirs(processed_folder, exist_ok=True)

print("✅ Google Drive mounted successfully!")
print(f"📂 PDF Folder: {pdf_folder}")
print(f"📂 Output Folder: {output_folder}")
print(f"📂 Processed Folder: {processed_folder}")

# 🔹 4️⃣ Set Device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"✅ Using device: {device}")

# 🔹 5️⃣ Load Model
trained_model_path = os.path.join(base_dir, "TinyLamaLegalModel/")
tokenizer = AutoTokenizer.from_pretrained(trained_model_path)
model = AutoModelForCausalLM.from_pretrained(trained_model_path).to(device)
model.gradient_checkpointing_enable()  # Reduce memory usage

print("✅ Model loaded successfully!")

# 🔹 6️⃣ Updated Questions for Legal Analysis
questions = [
    "Where was the case heard? (Court and Location)",
    "Who were the judges on the bench? (Names & Designations)",
    "When was the judgment or order pronounced? (Date of Pronouncement)",
    "What type of case is this? (Civil, Criminal, etc.)",
    "What is the case number or neutral citation?",
    "Are there any other citations with year/volume?",
    "Who is the petitioner in this case?",
    "Who is the respondent in this case?",
    "What was the final decision of the court? (Disposal Nature)",
    "What is the main issue or topic of this case in one sentence?",
    "Which laws or legal provisions are relevant to this case? (Acts, Rules, etc.)",
    "What specific sections, rules, circulars, or notifications are mentioned?",
    "Are there any other cases referred to in this document? (With citation numbers)",
    "Does this case refer to a previous court decision? (Lower court reference)",
    "Who represented the petitioner? (Lawyer’s name, if mentioned)",
    "Who represented the respondent? (Lawyer’s name, if mentioned)",
    "What are the facts and background of this case?",
    "What is the background of the parties and the transaction in question?",
    "What happened in the lower courts before this case reached here?",
    "What are the key legal questions or issues raised in this case?"
]

# 🔹 7️⃣ Extract Text from a PDF
def extract_text_from_pdf(pdf_path):
    try:
        doc = fitz.open(pdf_path)
        text = " ".join([page.get_text() for page in doc])
        if not text.strip():
            tqdm.write(f"⚠️ Warning: No text found in {os.path.basename(pdf_path)}")
        return text
    except Exception as e:
        tqdm.write(f"❌ Error reading {pdf_path}: {e}")
        return ""

# 🔹 8️⃣ Generate Embeddings (Optimized)
def generate_embeddings(texts):
    inputs = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=256
    ).to(device)

    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)

    hidden_states = outputs.hidden_states[-1]
    embeddings = hidden_states.mean(dim=1).detach().cpu().numpy()

    del inputs, outputs, hidden_states
    torch.cuda.empty_cache()
    gc.collect()

    return embeddings

# 🔹 9️⃣ Retrieve the Most Relevant Text Using FAISS
def retrieve_relevant_text(chunks, question):
    if not chunks:
        return "⚠️ No text extracted from the PDF."

    embeddings = generate_embeddings(chunks)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)

    question_embedding = generate_embeddings([question])
    distances, indices = index.search(question_embedding, 1)

    return chunks[indices[0][0]]

# 🔹 🔟 Process a Single PDF and Save Answers
def process_pdf_and_generate_answers(pdf_path):
    tqdm.write(f"📄 Processing: {os.path.basename(pdf_path)}")

    text = extract_text_from_pdf(pdf_path)
    if not text:
        tqdm.write(f"⚠️ Skipping {os.path.basename(pdf_path)} due to empty content.")
        return

    chunk_size = 1000
    chunks = [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]

    qa_pairs = []
    for question in tqdm(questions, desc="🔍 Answering questions", unit="question", leave=True):
        relevant_text = retrieve_relevant_text(chunks, question)
        qa_pairs.append((question, relevant_text))

        # ✅ Show real-time question-answer logging
        tqdm.write(f"✔️ {question} ➝ {relevant_text[:100]}...")  # Preview first 100 chars

    pdf_name = os.path.basename(pdf_path).replace(".pdf", "")
    output_file = os.path.join(output_folder, f"{pdf_name}_answers.xlsx")
    df = pd.DataFrame(qa_pairs, columns=["Question", "Answer"])
    df.to_excel(output_file, index=False)
    tqdm.write(f"✅ Answers saved to {output_file}")

    del text, chunks, qa_pairs
    torch.cuda.empty_cache()
    gc.collect()

# 🔹 1️⃣1️⃣ Process All PDFs and Rename Them
def process_all_pdfs():
    pdf_files = [f for f in os.listdir(pdf_folder) if f.endswith(".pdf")]

    if not pdf_files:
        print("✅ No new PDFs to process. All caught up!")
        return

    print(f"📂 Found {len(pdf_files)} PDFs. Starting processing...\n")

    for pdf_file in tqdm(pdf_files, desc="📄 Processing PDFs", unit="file", leave=True):
        pdf_path = os.path.join(pdf_folder, pdf_file)
        process_pdf_and_generate_answers(pdf_path)

        # Rename and move processed PDF
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        new_pdf_name = f"{os.path.splitext(pdf_file)[0]}_processed_{timestamp}.pdf"
        new_pdf_path = os.path.join(processed_folder, new_pdf_name)

        os.rename(pdf_path, new_pdf_path)
        tqdm.write(f"✅ Processed and moved to: {new_pdf_path}\n")

        torch.cuda.empty_cache()
        gc.collect()

# 🔹 1️⃣2️⃣ Run the Program
print("\n🚀 Starting batch processing of PDFs...\n")
process_all_pdfs()
print("\n🎉 All PDFs processed successfully!")
