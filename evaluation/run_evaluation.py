"""
run_evaluation.py — Pokretanje RAGAS automatske evaluacije

Korištenje (iz korijena projekta, s aktivnim venv (Python 3.10 radi stabilnosti)):
    py evaluation/run_evaluation.py

Preduvjeti:
    - Ollama mora biti pokrenuta (gemma3:4b i nomic-embed-text)
    - ChromaDB mora biti popunjena (indexer.py)
"""

import json
import csv
import math
import time
from pathlib import Path
from datetime import datetime

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.rag_pipeline import get_answer_with_sources
from evaluation.eval_dataset import EVAL_DATASET

from ragas.metrics import (
    answer_relevancy,
    faithfulness,
    context_precision,
    context_recall,
)
from ragas.dataset_schema import SingleTurnSample
from langchain_ollama import OllamaLLM, OllamaEmbeddings
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper

# Lokalni modeli za RAGAS evaluaciju
# zašto se koristi OllamaLLM, radi problema sa kompatibilnošću pa sam pokušala i sa OllamaLLM pa je i ostalo tako
ragas_llm = LangchainLLMWrapper(
    OllamaLLM(model="gemma3:4b", temperature=0.0)
)
ragas_embeddings = LangchainEmbeddingsWrapper(
    OllamaEmbeddings(model="nomic-embed-text")
)

METRICS = [answer_relevancy, faithfulness, context_precision, context_recall]
METRIC_NAMES = ["answer_relevancy", "faithfulness", "context_precision", "context_recall"]

for metric in METRICS:
    metric.llm = ragas_llm
    if hasattr(metric, "embeddings"):
        metric.embeddings = ragas_embeddings

# Konfiguracija otpornosti na padove Ollame
MAX_RETRIES = 3
RETRY_DELAY_SEC = 8
INTER_CALL_DELAY_SEC = 2

def pozovi_pipeline_s_retryem(question: str) -> dict:
    """
    Poziva RAG pipeline uz automatsko ponavljanje pri tranzijentnim greškama Ollama servera. 
    Nakon MAX_RETRIES pokušaja vraća prazan rezultat umjesto da prekine cijeli evaluacijski run.
    """
    zadnja_greska = None
    for pokusaj in range(1, MAX_RETRIES + 1):
        try:
            return get_answer_with_sources(question)
        except Exception as e:
            zadnja_greska = e
            print(f"  Pokušaj {pokusaj}/{MAX_RETRIES} neuspješan: {e}")
            if pokusaj < MAX_RETRIES:
                print(f"  Čekam {RETRY_DELAY_SEC}s (Ollama se oporavlja)...")
                time.sleep(RETRY_DELAY_SEC)

    print(f"  Svi pokušaji neuspješni. Zadnja greška: {zadnja_greska}")
    return {"answer": "", "sources": [], "retrieved_chunks": []}

#Poziva RAG pipeline za svako pitanje i bilježi odgovor + dohvaćene chunkove.
def prikupi_odgovore(eval_dataset: list) -> list:
    print(f"PRIKUPLJANJE ODGOVORA ({len(eval_dataset)} pitanja)")

    rezultati = []
    for i, item in enumerate(eval_dataset, 1):
        print(f"\n[{i:02d}/{len(eval_dataset)}] {item['id']}: {item['question'][:70]}...")

        rag_result = pozovi_pipeline_s_retryem(item["question"])
        odgovor = rag_result.get("answer", "")
        sources = rag_result.get("sources", [])
        retrieved_chunks = rag_result.get("retrieved_chunks", [])

        if not retrieved_chunks:
            print("  UPOZORENJE: pipeline nije vratio nijedan chunk")
            retrieved_chunks = ["[Nema dohvaćenog konteksta]"]

        rezultati.append({
            **item,
            "generated_answer": odgovor,
            "retrieved_contexts": retrieved_chunks,
            "sources_meta": sources,
        })
        print(f"  Odgovor ({len(odgovor)} znakova), {len(retrieved_chunks)} chunkova")
        time.sleep(INTER_CALL_DELAY_SEC)

    return rezultati

# Računa sve RAGAS metrike sinkrono, uzorak po uzorak, metrika po metriku.
def izracunaj_sve_metrike(rezultati: list) -> dict:
    print(f"\nRACUNANJE RAGAS METRIKA ({len(rezultati)} uzoraka x {len(METRICS)} metrike)")

    scores = {name: [] for name in METRIC_NAMES}

    for i, r in enumerate(rezultati, 1):
        uzorak = SingleTurnSample(
            user_input=r["question"],
            response=r["generated_answer"],
            retrieved_contexts=r["retrieved_contexts"],
            reference=r["ground_truth"],
        )

        print(f"  [{i:02d}/{len(rezultati)}] {r['id']}", end="", flush=True)

        for metric, name in zip(METRICS, METRIC_NAMES):
            vrijednost = metric.single_turn_score(uzorak)
            scores[name].append(vrijednost)
            status = f"{vrijednost:.3f}" if not math.isnan(vrijednost) else "NaN"
            print(f"  {name[:3]}={status}", end="", flush=True)

        print()

    return scores

# Računanje prosjeka po metrici
def agregiraj_score(scores: dict) -> dict:
    agregirano = {}
    for name, vrijednosti in scores.items():
        valjane = [v for v in vrijednosti if not math.isnan(v)]
        agregirano[name] = sum(valjane) / len(valjane) if valjane else float("nan")
    return agregirano

# Sprema agregatne metrike (JSON) i detalje po pitanju (CSV).
def spremi_rezultate(agregirano: dict, scores: dict, rezultati: list, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # JSON — agregatne metrike
    def fmt(v):
        return round(v, 4) if not math.isnan(v) else None

    metrics_dict = {
        "timestamp":         timestamp,
        "n_pitanja":         len(rezultati),
        "answer_relevancy":  fmt(agregirano["answer_relevancy"]),
        "faithfulness":      fmt(agregirano["faithfulness"]),
        "context_precision": fmt(agregirano["context_precision"]),
        "context_recall":    fmt(agregirano["context_recall"]),
    }
    json_path = output_dir / f"ragas_metrics_{timestamp}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(metrics_dict, f, ensure_ascii=False, indent=2)
    print(f"\n  Agregatne metrike: {json_path}")

    # CSV — detalji po pitanju s individualnim score-ovima
    csv_path = output_dir / f"ragas_details_{timestamp}.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "id", "document", "category", "question",
            "ground_truth", "generated_answer",
            "answer_relevancy", "faithfulness",
            "context_precision", "context_recall",
            "sources_meta",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for idx, r in enumerate(rezultati):
            def cell(name):
                v = scores[name][idx]
                return round(v, 4) if not math.isnan(v) else "nan"
            writer.writerow({
                "id":               r["id"],
                "document":         r["document"],
                "category":         r["category"],
                "question":         r["question"],
                "ground_truth":     r["ground_truth"],
                "generated_answer": r["generated_answer"],
                "answer_relevancy": cell("answer_relevancy"),
                "faithfulness":     cell("faithfulness"),
                "context_precision":cell("context_precision"),
                "context_recall":   cell("context_recall"),
                "sources_meta":     str(r.get("sources_meta", [])),
            })
    print(f"  Detalji po pitanju:  {csv_path}")

    return metrics_dict

# spremanje sirovih razultata radi citation_check.py da se ne mora ponovo pozivati
def spremi_sirove_rezultate(rezultati: list, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = output_dir / f"raw_results_{timestamp}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(rezultati, f, ensure_ascii=False, indent=2)
    print(f"  Sirovi rezultati:    {path}")
    return path


def provjeri_ollama_dostupnost():
    import urllib.request
    try:
        urllib.request.urlopen("http://127.0.0.1:11434", timeout=5)
        print("Ollama server: dostupan")
    except Exception as e:
        print(f"UPOZORENJE: Ollama server možda nije dostupan ({e}).")
        print("Provjeri da je Ollama pokrenuta prije nastavka.")


def main():
    print("RAG ASISTENT — RAGAS EVALUACIJA")
    provjeri_ollama_dostupnost()

    # 1. Prikupi odgovore iz RAG pipeline-a
    rezultati = prikupi_odgovore(EVAL_DATASET)

    # 2. Izracunaj RAGAS metrike sinkrono (zaobilazi executor)
    scores = izracunaj_sve_metrike(rezultati)

    # 3. Agregiraj i ispisi
    agregirano = agregiraj_score(scores)

    print("REZULTATI EVALUACIJE")

    nan_count = sum(1 for v in agregirano.values() if math.isnan(v))
    if nan_count == len(agregirano):
        print("UPOZORENJE: sve metrike su NaN — provjeri Ollamu i modele.\n")

    def ispisi(label, key):
        v = agregirano[key]
        print(f"  {label}  {v:.4f}" if not math.isnan(v) else f"  {label}  NaN")

    ispisi("Answer Relevancy: ", "answer_relevancy")
    ispisi("Faithfulness:     ", "faithfulness")
    ispisi("Context Precision:", "context_precision")
    ispisi("Context Recall:   ", "context_recall")

    # 4. Spremi rezultate
    print("\nSPREMANJE REZULTATA")
    output_dir = Path(__file__).parent / "results"
    spremi_rezultate(agregirano, scores, rezultati, output_dir)
    spremi_sirove_rezultate(rezultati, output_dir)


if __name__ == "__main__":
    main()