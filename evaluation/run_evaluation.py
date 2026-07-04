"""
run_evaluation.py — Pokretanje RAGAS automatske evaluacije

Korištenje (iz korijena projekta, s aktivnim venv):
    py evaluation/run_evaluation.py

Preduvjeti:
    - pip install ragas datasets langchain-ollama
    - Ollama mora biti pokrenuta (gemma3:4b i nomic-embed-text)
    - ChromaDB mora biti popunjena (backend/indexer.py)

VAŽNO: "contexts" polje koje se šalje u RAGAS dolazi iz
rag_result["retrieved_chunks"] — stvarnog teksta odlomaka koje je
ChromaDB dohvatio za dano pitanje. NE koristi se reference_context
(ručno napisani zlatni standard) jer bi to umjetno naduvalo
Context Recall i učinilo metriku besmislenom.
"""

import json
import csv
from pathlib import Path
from datetime import datetime

# Dodaj korijen projekta u Python path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.rag_pipeline import get_answer_with_sources
from evaluation.eval_dataset import EVAL_DATASET, build_ragas_dataset

# RAGAS uvozi 
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    answer_relevancy,
    faithfulness,
    context_precision,
    context_recall,
)
from langchain_ollama import ChatOllama, OllamaEmbeddings

# Konfiguracija RAGAS-a s lokalnim modelima
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper

ragas_llm = LangchainLLMWrapper(
    ChatOllama(model="gemma3:4b", temperature=0.0)
)
ragas_embeddings = LangchainEmbeddingsWrapper(
    OllamaEmbeddings(model="nomic-embed-text")
)

METRICS = [
    answer_relevancy,
    faithfulness,
    context_precision,
    context_recall,
]
for metric in METRICS:
    metric.llm = ragas_llm
    if hasattr(metric, "embeddings"):
        metric.embeddings = ragas_embeddings


def prikupi_odgovore(eval_dataset: list) -> list:
    """Poziva RAG pipeline za svako pitanje i bilježi odgovor + stvarno dohvaćene chunkove."""
    print(f"\n{'='*60}")
    print(f"PRIKUPLJANJE ODGOVORA ({len(eval_dataset)} pitanja)")
    print(f"{'='*60}")

    rezultati = []
    for i, item in enumerate(eval_dataset, 1):
        print(f"\n[{i:02d}/{len(eval_dataset)}] {item['id']}: {item['question'][:70]}...")
        try:
            rag_result = get_answer_with_sources(item["question"])
            odgovor = rag_result.get("answer", "")
            sources = rag_result.get("sources", [])
            # Stvarni tekst dohvaćenih odlomaka iz ChromaDB — ovo je ono što
            # RAGAS treba u "contexts" polju (NE reference_context, koji je
            # naš ručno napisani zlatni standard za usporedbu, ne stvarni dohvat)
            retrieved_chunks = rag_result.get("retrieved_chunks", [])
        except Exception as e:
            print(f" Greška: {e}")
            odgovor = ""
            sources = []
            retrieved_chunks = []

        if not retrieved_chunks:
            # Bilježimo prazan dohvat kao stvarni neuspjeh, ne maskiramo ga
            # zlatnim kontekstom — prazan dohvat treba sniziti Context Recall,
            # ne biti skriven umjetno dobrim rezultatom.
            print("  UPOZORENJE: pipeline nije vratio nijedan chunk")
            retrieved_chunks = ["[Nema dohvaćenog konteksta]"]

        rezultati.append({
            **item,
            "generated_answer": odgovor,
            "retrieved_contexts": retrieved_chunks,
            "sources_meta": sources,  # metapodaci za citation_check.py
        })
        print(f" Odgovor ({len(odgovor)} znakova), {len(retrieved_chunks)} chunkova")

    return rezultati


def pripremi_ragas_dataset(rezultati: list) -> Dataset:
    """Pretvara rezultate u HuggingFace Dataset format koji RAGAS očekuje."""
    return Dataset.from_dict({
        "question":    [r["question"] for r in rezultati],
        "answer":      [r["generated_answer"] for r in rezultati],
        "contexts":    [r["retrieved_contexts"] for r in rezultati],
        "ground_truth":[r["ground_truth"] for r in rezultati],
    })


def spremi_rezultate(ragas_score, rezultati: list, output_dir: Path):
    """Sprema rezultate evaluacije u JSON i CSV format."""
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Agregatne metrike
    metrics_dict = {
        "timestamp": timestamp,
        "n_pitanja": len(rezultati),
        "answer_relevancy":  round(float(ragas_score["answer_relevancy"]), 4),
        "faithfulness":      round(float(ragas_score["faithfulness"]), 4),
        "context_precision": round(float(ragas_score["context_precision"]), 4),
        "context_recall":    round(float(ragas_score["context_recall"]), 4),
    }

    json_path = output_dir / f"ragas_metrics_{timestamp}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(metrics_dict, f, ensure_ascii=False, indent=2)
    print(f"\n Agregatne metrike: {json_path}")

    # Detalji po pitanju
    csv_path = output_dir / f"ragas_details_{timestamp}.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "id", "document", "category", "question",
            "ground_truth", "generated_answer", "sources_meta"
        ])
        writer.writeheader()
        for r in rezultati:
            writer.writerow({
                "id":               r["id"],
                "document":         r["document"],
                "category":         r["category"],
                "question":         r["question"],
                "ground_truth":     r["ground_truth"],
                "generated_answer": r["generated_answer"],
                "sources_meta":     r.get("sources_meta", []),
            })
    print(f"Detalji po pitanju: {csv_path}")

    return metrics_dict


def spremi_sirove_rezultate(rezultati: list, output_dir: Path) -> Path:
    """
    Sprema sirove rezultate (uključujući sources_meta i retrieved_contexts)
    u JSON — koristi se kao ulaz za citation_check.py, bez potrebe za
    ponovnim pozivanjem RAG pipeline-a.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = output_dir / f"raw_results_{timestamp}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(rezultati, f, ensure_ascii=False, indent=2)
    print(f"Sirovi rezultati (za citation_check.py): {path}")
    return path


def main():
    print("RAG ASISTENT — RAGAS EVALUACIJA")

    # 1. Prikupi odgovore iz pipeline-a
    rezultati = prikupi_odgovore(EVAL_DATASET)

    # 2. Pripremi dataset za RAGAS
    print("POKRETANJE RAGAS METRIKA")
    ragas_ds = pripremi_ragas_dataset(rezultati)

    # 3. Pokreni evaluaciju
    ragas_score = evaluate(ragas_ds, metrics=METRICS)

    # 4. Ispiši rezultate
    print("\n")
    print("REZULTATI EVALUACIJE")
    print("="*60)
    print(f"  Answer Relevancy:  {ragas_score['answer_relevancy']:.4f}")
    print(f"  Faithfulness:      {ragas_score['faithfulness']:.4f}")
    print(f"  Context Precision: {ragas_score['context_precision']:.4f}")
    print(f"  Context Recall:    {ragas_score['context_recall']:.4f}")

    # 5. Spremi rezultate
    output_dir = Path(__file__).parent / "results"
    spremi_rezultate(ragas_score, rezultati, output_dir)
    spremi_sirove_rezultate(rezultati, output_dir)


if __name__ == "__main__":
    main()
