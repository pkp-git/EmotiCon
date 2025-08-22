import pandas as pd
import PyPDF2
from sentence_transformers import SentenceTransformer
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Load text emotion model (same as audio.py)
TEXT_MODEL_NAME = "j-hartmann/emotion-english-distilroberta-base"
text_tokenizer = AutoTokenizer.from_pretrained(TEXT_MODEL_NAME)
text_model = AutoModelForSequenceClassification.from_pretrained(TEXT_MODEL_NAME)

def extract_pdf_chunks(pdf_path, chunk_size=300, overlap=50):
    """Extracts text from PDF and chunks it."""
    reader = PyPDF2.PdfReader(pdf_path)
    full_text = ""
    for page in reader.pages:
        full_text += page.extract_text() + " "
    words = full_text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i+chunk_size])
        if chunk:
            chunks.append(chunk)
    return chunks

def embed_chunks(chunks, model_name='all-MiniLM-L6-v2'):
    """Embeds text chunks using SentenceTransformer."""
    model = SentenceTransformer(model_name)
    embeddings = model.encode(chunks, convert_to_numpy=True)
    return embeddings

def get_text_emotion(text):
    """Get emotion from text using the loaded model."""
    inputs = text_tokenizer(text, return_tensors="pt", truncation=True)
    with torch.no_grad():
        logits = text_model(**inputs).logits
        probs = torch.softmax(logits, dim=-1).squeeze()
        label_id = torch.argmax(probs).item()
        label = text_model.config.id2label[label_id]
        confidence = probs[label_id].item()
    return label, confidence

def retrieve_context(query, chunk_texts, chunk_embeds, model_name='all-MiniLM-L6-v2', top_k=1):
    """RAG: Retrieve most relevant chunk for a query."""
    model = SentenceTransformer(model_name)
    query_embed = model.encode([query], convert_to_numpy=True)[0]
    sims = np.dot(chunk_embeds, query_embed) / (np.linalg.norm(chunk_embeds, axis=1) * np.linalg.norm(query_embed) + 1e-8)
    top_idx = np.argsort(sims)[-top_k:][::-1]
    return [chunk_texts[i] for i in top_idx]

def enrich_segments_with_context(pdf_path, csv_path, out_csv_path):
    # Chunk and embed PDF
    chunks = extract_pdf_chunks(pdf_path)
    chunk_embeds = embed_chunks(chunks)
    # Load segments
    df = pd.read_csv(csv_path)
    context_list = []
    context_emotion_list = []
    for transcript in df['Transcript']:
        context = retrieve_context(transcript, chunks, chunk_embeds)[0]
        emotion, conf = get_text_emotion(context)
        context_list.append(context)
        context_emotion_list.append(f"{emotion} ({conf:.2f})")
    df['Context'] = context_list
    df['Context Emotion'] = context_emotion_list
    df.to_csv(out_csv_path, index=False)
    print(f"Saved enriched CSV to {out_csv_path}")

# Example usage:
# enrich_segments_with_context("context/Final Context - 1.pdf", "segments.csv", "segments_with_context.csv")
# ...existing code...

if __name__ == "__main__":
    import sys
    if len(sys.argv) == 4:
        pdf_path = sys.argv[1]
        csv_path = sys.argv[2]
        out_csv_path = sys.argv[3]
        enrich_segments_with_context(pdf_path, csv_path, out_csv_path)
    else:
        print("Usage: python rag.py <pdf_path> <csv_path> <out_csv_path>")