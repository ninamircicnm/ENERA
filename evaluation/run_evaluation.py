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

NAPOMENA O STABILNOSTI: pri uzastopnom pozivanju 20 pitanja zaredom
Ollama server lokalno povremeno prekida vezu (tranzijentni pad procesa
"model runner". Da bi cijeli evaluacijski run preživio takav
pojedinačni pad umjesto da se prekine na pola, prikupi_odgovore koristi
retry s pauzom (RETRY_DELAY_SEC) i kratku pauzu između SVIH poziva
(INTER_CALL_DELAY_SEC) kako bi se rasteretio lokalni model runner.
"""

import json
import csv
import time
from pathlib import Path
from datetime import datetime

# Dodaj korijen projekta u Python path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.rag_pipeline import get_answer_with_sources
from evaluation.eval_dataset import EVAL_DATASET

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
from ragas.run_config import RunConfig

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

RAGAS_RUN_CONFIG = RunConfig(max_workers=1, timeout=120) 

for metric in METRICS:
    metric.llm = ragas_llm
    if hasattr(metric, "embeddings"):
        metric.embeddings = ragas_embeddings

# Rezervni mehanizam: postavljamo run_config direktno na svaku metriku
# jer ragas==0.2.15 ne propagira uvijek run_config iz evaluate() poziva
# na sve metrike — direktno postavljanje osigurava max_workers=1 svugdje.
for metric in METRICS:
    if hasattr(metric, "run_config"):
        metric.run_config = RAGAS_RUN_CONFIG

# max_workers=1 i timeout=120 izbjegavaju paralelni asyncio koji na Python 3.14 (Windows)
# uzrokuje "RuntimeError: Timeout should be used inside a task" unutar
# ragas.executor / nest_asyncio mehanizma. Sporije, ali pouzdano.
# NAPOMENA: ragas==0.2.15 prima run_config u evaluate(), ali ga ne propagira
# konzistentno na sve metrike u svim verzijama — kao rezervni mehanizam
# postavljamo run_config direktno i na svaku metriku.


# Konfiguracija otpornosti na tranzijentne padove Ollame 
MAX_RETRIES = 3            # koliko se puta ponavlja isto pitanje nakon greške
RETRY_DELAY_SEC = 8        # pauza prije ponovnog pokušaja (daje Ollami vremena da se oporavi)
INTER_CALL_DELAY_SEC = 2   # pauza nakon SVAKOG poziva (uspješnog ili ne) — rasterećuje model runner


def pozovi_pipeline_s_retryem(question: str, max_retries: int = MAX_RETRIES) -> dict:
    """
    Poziva get_answer_with_sources uz ponavljanje pri tranzijentnim
    greškama (npr. Ollama server privremeno prekine vezu / padne proces
    modela). Nakon max_retries neuspjelih pokušaja, vraća prazan rezultat
    umjesto da sruši cijeli evaluacijski run.
    """
    zadnja_greska = None
    for pokusaj in range(1, max_retries + 1):
        try:
            return get_answer_with_sources(question)
        except Exception as e:
            zadnja_greska = e
            print(f"  Pokušaj {pokusaj}/{max_retries} neuspješan: {e}")
            if pokusaj < max_retries:
                print(f"  Čekam {RETRY_DELAY_SEC}s prije ponovnog pokušaja "
                      f"(Ollama se možda oporavlja)...")
                time.sleep(RETRY_DELAY_SEC)

    print(f"  Svi pokušaji neuspješni za ovo pitanje. Zadnja greška: {zadnja_greska}")
    return {"answer": "", "sources": [], "retrieved_chunks": []}


def prikupi_odgovore(eval_dataset: list) -> list:
    """Poziva RAG pipeline za svako pitanje i bilježi odgovor + stvarno dohvaćene chunkove."""
    print(f"PRIKUPLJANJE ODGOVORA ({len(eval_dataset)} pitanja)")

    rezultati = []
    for i, item in enumerate(eval_dataset, 1):
        print(f"\n[{i:02d}/{len(eval_dataset)}] {item['id']}: {item['question'][:70]}...")

        rag_result = pozovi_pipeline_s_retryem(item["question"])
        odgovor = rag_result.get("answer", "")
        sources = rag_result.get("sources", [])
        # Stvarni tekst dohvaćenih odlomaka iz ChromaDB — ovo je ono što
        # RAGAS treba u "contexts" polju (NE reference_context, koji je
        # naš ručno napisani zlatni standard za usporedbu, ne stvarni dohvat)
        retrieved_chunks = rag_result.get("retrieved_chunks", [])

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

        # Pauza nakon svakog poziva (uspješnog ili ne) da se Ollama
        # model runner stigne rasteretiti prije sljedećeg upita
        time.sleep(INTER_CALL_DELAY_SEC)

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

    # Agregatne metrike — sigurna_vrijednost() štiti od slučaja kad RAGAS
    # vrati listu NaN-ova umjesto skalara (kad asyncio jobovi padnu)
    metrics_dict = {
        "timestamp": timestamp,
        "n_pitanja": len(rezultati),
        "answer_relevancy":  round(sigurna_vrijednost(ragas_score, "answer_relevancy"), 4),
        "faithfulness":      round(sigurna_vrijednost(ragas_score, "faithfulness"), 4),
        "context_precision": round(sigurna_vrijednost(ragas_score, "context_precision"), 4),
        "context_recall":    round(sigurna_vrijednost(ragas_score, "context_recall"), 4),
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


def provjeri_ollama_dostupnost():
    """Brza provjera da je Ollama server dostupan prije pokretanja punog runa."""
    import urllib.request
    try:
        urllib.request.urlopen("http://127.0.0.1:11434", timeout=5)
        print("Ollama server: dostupan")
    except Exception as e:
        print(f"UPOZORENJE: Ollama server možda nije dostupan ({e}).")
        print("Provjeri da je Ollama pokrenuta prije nastavka.")

def sigurna_vrijednost(ragas_score, kljuc: str) -> float:
    """
    Ekstrahira skalarnu vrijednost iz RAGAS rezultata. Ako je vrijednost
    lista (slučaj kad su pojedinačni pozivi metrika propali pa je rezultat
    niz NaN-ova umjesto skalara), računa prosjek zanemarujući NaN.
    Vraća float('nan') ako nema valjanih vrijednosti.
    """
    import math
    vrijednost = ragas_score[kljuc]
    if isinstance(vrijednost, (list, tuple)):
        valjane = [v for v in vrijednost if v is not None and not (isinstance(v, float) and math.isnan(v))]
        return sum(valjane) / len(valjane) if valjane else float("nan")
    return float(vrijednost)

def main():
    print("RAG ASISTENT — RAGAS EVALUACIJA")
    provjeri_ollama_dostupnost()

    # 1. Prikupi odgovore iz pipeline-a
    rezultati = prikupi_odgovore(EVAL_DATASET)

    # 2. Pripremi dataset za RAGAS
    print("POKRETANJE RAGAS METRIKA")
    ragas_ds = pripremi_ragas_dataset(rezultati)

    # 3. Pokreni evaluaciju
    ragas_score = evaluate(ragas_ds, metrics=METRICS, run_config=RAGAS_RUN_CONFIG)

    # 4. Ispiši rezultate
    print("REZULTATI EVALUACIJE")
    print(f"  Answer Relevancy:  {sigurna_vrijednost(ragas_score, 'answer_relevancy'):.4f}")
    print(f"  Faithfulness:      {sigurna_vrijednost(ragas_score, 'faithfulness'):.4f}")
    print(f"  Context Precision: {sigurna_vrijednost(ragas_score, 'context_precision'):.4f}")
    print(f"  Context Recall:    {sigurna_vrijednost(ragas_score, 'context_recall'):.4f}")

    # 5. Spremi rezultate
    output_dir = Path(__file__).parent / "results"
    spremi_rezultate(ragas_score, rezultati, output_dir)
    spremi_sirove_rezultate(rezultati, output_dir)


if __name__ == "__main__":
    main()