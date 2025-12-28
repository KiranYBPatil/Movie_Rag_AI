from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import pipeline
import torch
import re

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # OK for demo
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Reduce torch memory usage
torch.set_num_threads(1)

# Load FLAN-T5 small (CPU only)
generator = pipeline(
    "text2text-generation",
    model="google/flan-t5-small",
    device=-1
)

# Load knowledge base
with open("knowledge.txt", "r", encoding="utf-8") as f:
    documents = [line.strip() for line in f if line.strip()]

class Question(BaseModel):
    question: str

def retrieve_context(question: str):
    question = question.lower()
    for doc in documents:
        if any(word in doc.lower() for word in re.findall(r"\w+", question)):
            return doc
    return documents[0]  # fallback

@app.get("/")
def home():
    return {"message": "RAG AI Assistant is running 🚀"}

@app.post("/ask")
def ask(q: Question):
    context = retrieve_context(q.question)

    prompt = f"""
Context:
{context}

Question:
{q.question}

Answer:
"""

    result = generator(
        prompt,
        max_length=100,
        do_sample=False
    )[0]["generated_text"]

    return {"answer": result}
