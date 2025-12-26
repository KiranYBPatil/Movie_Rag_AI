from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import pipeline
from sentence_transformers import SentenceTransformer, util
import torch

app = FastAPI()

# ✅ CORS FIX (THIS SOLVES YOUR ERROR)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # React dev server
    allow_credentials=True,
    allow_methods=["*"],  # Allows OPTIONS, POST, GET
    allow_headers=["*"],
)

# Load embedding model
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# Load FLAN-T5 small
generator = pipeline(
    "text2text-generation",
    model="google/flan-t5-small",
    device=-1
)

# Load knowledge
with open("knowledge.txt", "r", encoding="utf-8") as f:
    documents = f.readlines()

doc_embeddings = embedder.encode(documents, convert_to_tensor=True)

class Question(BaseModel):
    question: str

@app.get("/")
def home():
    return {"message": "RAG AI Assistant is running 🚀"}

@app.post("/ask")
def ask(question: Question):
    q_embedding = embedder.encode(question.question, convert_to_tensor=True)
    scores = util.cos_sim(q_embedding, doc_embeddings)[0]
    best_doc = documents[int(torch.argmax(scores))]

    prompt = f"""
Context:
{best_doc}

Question:
{question.question}

Answer:
"""

    result = generator(prompt, max_length=120)[0]["generated_text"]
    return {"answer": result}
