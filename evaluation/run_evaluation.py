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

from backend.rag_pipeline import get_answer_with_sources, GENERATOR_MODEL
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

from typing import Optional
from langchain_core.outputs import LLMResult


# ---------------------------------------------------------------------------
# JSON middle-man za Gemmu 3
#
# Gemma 3 4B nije pouzdano JSON-kompatibilna pa RAGAS metrikama (koje očekuju
# strogi JSON) ponekad vrati prozu → iznimka → NaN. Ovaj omotač prvo pokuša
# izvući JSON iz odgovora; ako ga nema, šalje few-shot upit koji modelu pokaže
# točan format i traži rekonstrukciju. Ako ni to ne uspije, ostavlja se izvorni
# tekst → RAGAS baca iznimku → postojeći NaN fallback (MAX_METRIC_RETRIES).
#
# VAŽNO: hook je na _generate / _agenerate, a NE na _call. OllamaLLM iz
# langchain-ollama 0.3.3 nasljeđuje BaseLLM i implementira _generate/_agenerate
# izravno (preko _stream_with_aggregation), pa _call nikad nije na putanji
# poziva. RAGAS metrike kroz single_turn_score idu asinkrono → _agenerate.
#
# NAPOMENA: few-shot prefiks se gradi konkatenacijom, NE str.format()-om, jer
# sadrži vitičaste zagrade ({ }) iz JSON primjera koje bi format pogrešno
# protumačio kao polja.
# ---------------------------------------------------------------------------

JSON_FIX_PREFIX = """Your previous response was not valid JSON. Extract or reconstruct the JSON object from the text below. Return ONLY the JSON object — no explanation, no markdown, no code blocks.

Example 1 (faithfulness — statement extraction):
Text: Odgovor iznosi dvije tvrdnje: prva je da nacionalni cilj udjela obnovljivih izvora energije iznosi 42,5% do 2030. godine, a druga da je to regulirano člancima 7–9 Zakona o OIE.
Output: {"statements": ["Nacionalni cilj udjela obnovljivih izvora energije iznosi 42,5% do 2030. godine.", "Cilj je reguliran člancima 7–9 Zakona o OIE."]}

Example 2 (faithfulness — NLI verdict):
Text: Tvrdnja o cilju od 42,5% podržana je kontekstom koji eksplicitno navodi tu vrijednost, presuda je 1. Tvrdnja o člancima 7–9 također se nalazi u kontekstu, presuda je 1.
Output: {"statements": [{"statement": "Nacionalni cilj udjela obnovljivih izvora energije iznosi 42,5% do 2030. godine.", "reason": "Kontekst eksplicitno navodi vrijednost od 42,5%.", "verdict": 1}, {"statement": "Cilj je reguliran člancima 7–9 Zakona o OIE.", "reason": "Kontekst upućuje na članke 7–9 za metodologiju izračuna.", "verdict": 1}]}

Example 3 (answer_relevancy):
Text: Na temelju odgovora o tržišnoj premiji i zajamčenoj otkupnoj cijeni, pitanje koje bi generiralo taj odgovor je: Što je tržišna premija i kako se izračunava za postrojenja na obnovljive izvore energije? Odgovor je konkretan i ne izbjegava temu.
Output: {"question": "Što je tržišna premija i kako se izračunava za postrojenja na obnovljive izvore energije?", "noncommittal": 0}

Example 4 (context_precision):
Text: Dohvaćeni kontekst iz Zakona o OIE opisuje mehanizam tržišne premije u člancima 16–26, što je izravno relevantno za odgovor na pitanje o izračunu premije. Kontekst je koristan.
Output: {"reason": "Kontekst iz članaka 16–26 izravno opisuje mehanizam izračuna premije.", "verdict": 1}

Example 5 (context_recall):
Text: Referentni odgovor spominje tri stvari: cilj od 42,5%, ulogu HROTE-a i metodu izračuna premije. Prve dvije se nalaze u kontekstu, ali izračun premije nije eksplicitno naveden.
Output: {"classifications": [{"statement": "Nacionalni cilj iznosi 42,5%.", "reason": "Eksplicitno navedeno u dohvaćenom kontekstu.", "attributed": 1}, {"statement": "HROTE upravlja isplatama premija.", "reason": "Kontekst spominje ulogu HROTE-a.", "attributed": 1}, {"statement": "Premija se izračunava kao razlika između zajamčene i tržišne cijene.", "reason": "Nije pronađeno u dohvaćenom kontekstu.", "attributed": 0}]}

Text: """


class OllamaMiddleMan(OllamaLLM):
    """OllamaLLM koji jamči da izlaz sadrži valjan JSON objekt.

    Hook je na _generate / _agenerate (NE _call) jer OllamaLLM zaobilazi _call.
    Vidi komentar iznad za detalje.
    """

    @staticmethod
    def _izvuci_json(tekst: str) -> Optional[str]:
        # Izvlači JSON objekt iz teksta — od prvog "{" do zadnjeg "}".
        pocetak = tekst.find("{")
        kraj = tekst.rfind("}")
        if pocetak != -1 and kraj != -1 and kraj > pocetak:
            return tekst[pocetak:kraj + 1]
        return None

    def _generate(self, prompts, stop=None, run_manager=None, **kwargs) -> LLMResult:
        rezultat = super()._generate(prompts, stop=stop, run_manager=run_manager, **kwargs)
        for gen_list in rezultat.generations:
            gen = gen_list[0]
            json_tekst = self._izvuci_json(gen.text)
            if json_tekst:
                gen.text = json_tekst
                continue
            # Few-shot popravak (sinkroni put)
            popravak = super()._generate([JSON_FIX_PREFIX + gen.text], stop=stop, **kwargs)
            popravljeni = self._izvuci_json(popravak.generations[0][0].text)
            if popravljeni:
                gen.text = popravljeni
            # inače: ostavi original → RAGAS baca iznimku → NaN fallback
        return rezultat

    async def _agenerate(self, prompts, stop=None, run_manager=None, **kwargs) -> LLMResult:
        rezultat = await super()._agenerate(prompts, stop=stop, run_manager=run_manager, **kwargs)
        for gen_list in rezultat.generations:
            gen = gen_list[0]
            json_tekst = self._izvuci_json(gen.text)
            if json_tekst:
                gen.text = json_tekst
                continue
            # Few-shot popravak (asinkroni put — ovaj RAGAS stvarno koristi)
            popravak = await super()._agenerate([JSON_FIX_PREFIX + gen.text], stop=stop, **kwargs)
            popravljeni = self._izvuci_json(popravak.generations[0][0].text)
            if popravljeni:
                gen.text = popravljeni
        return rezultat


# Lokalni modeli za RAGAS evaluaciju
# OllamaMiddleMan = OllamaLLM s JSON zaštitom (sloj prije NaN fallbacka)
ragas_llm = LangchainLLMWrapper(
    OllamaMiddleMan(model="gemma3:4b", temperature=0.0) # sudac; temp 0.0 (≠ produkcija 0.2)
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
# Koliko se puta pokušava izračunati pojedina metrika prije nego se zabilježi NaN.
# 2 = ukupno dva pokušaja po metrici (model je stohastičan pa drugi pokušaj
# često vrati valjani JSON tamo gdje je prvi vratio prozu).
MAX_METRIC_RETRIES = 2

MODEL_TAG = GENERATOR_MODEL.replace(":", "-")

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

        if not odgovor:
            print("  UPOZORENJE: pipeline vratio prazan odgovor")

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
            vrijednost = float("nan")
            for pokusaj in range(1, MAX_METRIC_RETRIES + 1):
                try:
                    vrijednost = metric.single_turn_score(uzorak)
                    break  # uspjeh — prekidamo petlju pokušaja
                except Exception as e:
                    if pokusaj < MAX_METRIC_RETRIES:
                        print(f"\n    {name} — pokušaj {pokusaj}/{MAX_METRIC_RETRIES} "
                              f"neuspješan ({type(e).__name__}), ponavljam...", flush=True)
                    else:
                        print(f"\n    {name} — neuspješno nakon {MAX_METRIC_RETRIES} "
                              f"pokušaja ({type(e).__name__}) → NaN", flush=True)
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
def spremi_rezultate(agregirano: dict, scores: dict, rezultati: list, output_dir: Path, timestamp: str):
    output_dir.mkdir(parents=True, exist_ok=True)
    # timestamp se prima iz main() radi konzistentnosti svih datoteka istog pokretanja

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
def spremi_sirove_rezultate(rezultati: list, output_dir: Path, timestamp: str) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    # timestamp se prima iz main() radi konzistentnosti svih datoteka istog pokretanja
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

    output_dir = Path(__file__).parent / "results"

    # Jedan timestamp za cijelo pokretanje — sve izlazne datoteke (raw_results,
    # ragas_metrics, ragas_details) dijele isti sufiks pa ih je moguće povezati.
    timestamp =  f"{MODEL_TAG}_" + datetime.now().strftime("%Y%m%d_%H%M%S")

    # 1. Prikupi odgovore iz RAG pipeline-a
    rezultati = prikupi_odgovore(EVAL_DATASET)

    # 1b. ODMAH spremi sirove rezultate
    print("\nSPREMANJE SIROVIH REZULTATA (prije metrika)")
    spremi_sirove_rezultate(rezultati, output_dir, timestamp)

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

    # 4. Spremi agregatne i detaljne rezultate
    print("\nSPREMANJE REZULTATA")
    spremi_rezultate(agregirano, scores, rezultati, output_dir, timestamp)

if __name__ == "__main__":
    main()