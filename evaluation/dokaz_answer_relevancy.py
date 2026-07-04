"""
dokaz_answer_relevancy.py

Cilj: pokazati IZRAVNO, na vlastitim podacima, dvije stvari koje objašnjavaju
nulte vrijednosti metrike Answer Relevancy (RAGAS 0.2.15):

  (1) na kojem se JEZIKU generiraju "unatrag" pitanja koja metrika koristi
      (očekivano: engleski, jer su uputa i primjeri u RAGAS-u na engleskome);
  (2) da model-procjenitelj SADRŽAJNO ODREĐENE (committal) odgovore ponekad
      pogrešno označi kao noncommittal, čime rezultat pada na točno 0,0
      (dovoljna je 1 od 3 procjene, jer score = sim * int(not any_noncommittal)).

Pokreni u istom okruženju (venv310) u kojem radi RAGAS evaluacija.
Skripta koristi RAGAS-ov IZVORNI predložak (ResponseRelevancePrompt), pa je
vjerna onome što je metrika stvarno radila.
"""

import asyncio
import json

# --- 1) MODEL-PROCJENITELJ ---------------------------------------------------
# Koristi se ISTI model-procjenitelj kao u evaluaciji. 
from langchain_ollama import OllamaLLM
from ragas.llms import LangchainLLMWrapper
from run_evaluation import OllamaMiddleMan

llm = LangchainLLMWrapper(OllamaMiddleMan(model="gemma3:4b", temperature=0.0))

# --- 2) RAGAS-ov izvorni predložak za Answer Relevancy -----------------------
from ragas.metrics._answer_relevance import (
    ResponseRelevancePrompt,
    ResponseRelevanceInput,
)

prompt = ResponseRelevancePrompt()

# --- 3) Učitaj Mistralove/Gemmine odgovore iz sirovih rezultata ----------------------
RAW = "evaluation/raw_results_mistral_20260702_233554.json"
raw = json.load(open(RAW, encoding="utf-8"))
by_id = {d["id"]: d for d in raw}

# Pet "committal-nula" (očekujemo EN pitanja i poneki pogrešan noncommittal=1)
COMMITTAL_ZERO = ["ZOI-02", "PRI-04", "MET-01", "JPO-01", "JPO-04"]
# Pet "hedging-nula" (očekujemo noncommittal=1 s razlogom)
HEDGING_ZERO = ["ZOI-01", "ZOI-04", "MET-03", "VEN-04", "JPO-03"]

STRICTNESS = 3  # RAGAS-ova zadana postavka (3 generirana pitanja po odgovoru)


async def probe(answer: str, k: int = STRICTNESS):
    trips = []
    for _ in range(k):
        res = await prompt.generate(data=ResponseRelevanceInput(response=answer), llm=llm)
        trips.append((res.question, res.noncommittal))
    return trips


async def run(ids, naslov):
    print("\n" + "#" * 82)
    print("#", naslov)
    print("#" * 82)
    for qid in ids:
        ans = by_id[qid]["generated_answer"]
        trips = await probe(ans)
        any_nc = any(nc for _, nc in trips)
        print("=" * 82)
        print(f"{qid}   (poništeno na 0,0? {'DA' if any_nc else 'NE'})")
        for i, (q, nc) in enumerate(trips, 1):
            print(f"  [{i}] noncommittal={nc} | generirano pitanje: {q}")


async def main():
    await run(COMMITTAL_ZERO, "SADRŽAJNO ODREĐENI ODGOVORI (nula = artefakt procjenitelja)")
    await run(HEDGING_ZERO, "IZBJEGAVAJUĆI ODGOVORI (nula = opravdana)")


if __name__ == "__main__":
    asyncio.run(main())
