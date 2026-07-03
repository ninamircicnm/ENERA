from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from backend.rag_pipeline import get_answer_with_sources

app = FastAPI(
    title="RAG Asistent — Obnovljivi izvori energije",
    description="AI asistent za podršku poslovnom odlučivanju u domeni obnovljivih izvora energije",
    version="1.0.0"
)

# CORS — potrebno za Streamlit frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Modeli zahtjeva i odgovora
class UpitRequest(BaseModel):
    question: str

class IzvorResponse(BaseModel):
    naziv_dokumenta: str
    stranica: int | str
    kategorija: str

class OdgovorResponse(BaseModel):
    answer: str
    sources: list[IzvorResponse]

# Endpoint za zdravlje servera
@app.get("/health")
def health_check():
    return {"status": "ok"}

# Glavni endpoint
@app.post("/ask", response_model=OdgovorResponse)
def ask(request: UpitRequest):
    result = get_answer_with_sources(request.question)
    return result