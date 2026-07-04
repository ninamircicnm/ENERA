"""
precision_filter_demo.py

Demonstracija učinka metadata filtra (kategorija) na preciznost dohvata.

Za nekoliko reprezentativnih pitanja uspoređuje dva načina dohvata:
  (A) čisto semantičko pretraživanje (top-k)               — kao u rag_pipeline.py
  (B) semantičko pretraživanje + filter po kategoriji        — hibridni pristup

Mjeri se "kategorijska čistoća" (category purity) dohvaćenog skupa:
    purity = (broj dohvaćenih odlomaka iz ciljane kategorije) / k

Napomena o metodologiji: purity NIJE isto što i RAGAS Context Precision.
Purity je strukturni proxy koji izravno mjeri međukategorijski šum u dohvatu
— upravo onu pojavu koja obara Context Precision. Za formalni
RAGAS broj, skripta sprema oba dohvaćena skupa (A i B) u JSON, pa ih je moguće
naknadno provući kroz RAGAS Context Precision bez ponovnog pozivanja modela.

Pokretanje (iz mape projekta, uz pokrenutu Ollamu):
    py evaluation/precision_filter_demo.py
"""

import json
from pathlib import Path
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma

BASE_DIR = Path(__file__).parent.parent
CHROMA_DIR = BASE_DIR / "chroma_db"
OUT_JSON = Path(__file__).parent / "results" / "precision_filter_demo.json"

K = 3

# Testna pitanja + ciljana kategorija (egzaktan string iz metapodaci_dokumenti.xlsx;
# Chroma filter zahtijeva podudaranje slovo-u-slovo, uključujući velika slova i dijakritiku).
TESTNA = [
    {
        "id": "MET-pregled",
        "pitanje": "Što je energetski pregled zgrade i koji su njegovi ciljevi?",
        "kategorija": "Energetski pregled zgrada",   # jedinstvena -> Metodologija
    },
    {
        "id": "VEN-pregledi",
        "pitanje": "Koliko često se moraju provoditi redoviti pregledi sustava ventilacije i klimatizacije?",
        "kategorija": "Mjere obnove zgrada",          # dijele Metodologija + Tehnički propis
    },
    {
        "id": "PRI-povlasteni",
        "pitanje": "Koji je postupak stjecanja statusa povlaštenog proizvođača električne energije?",
        "kategorija": "Pravni okvir i postupci",       # dijele Priručnik + Zakon
    },
]


def _stranica(meta):
    s = meta.get("tiskana_stranica")
    return s if s is not None else meta.get("page", "?")


def _red(rank, doc, score, ciljana):
    meta = doc.metadata
    kat = meta.get("kategorija", "?")
    return {
        "rank": rank,
        "dokument": meta.get("naziv_dokumenta", "?"),
        "kategorija": kat,
        "stranica": _stranica(meta),
        "status": meta.get("status_dokumenta", ""),
        "score": round(float(score), 4),
        "u_ciljanoj": (kat == ciljana),
        "tekst": doc.page_content,
    }


def _ispis(naslov, redovi, ciljana):
    print(f"  {naslov}")
    if not redovi:
        print("    (ništa dohvaćeno)")
        return
    for r in redovi:
        flag = "✓" if r["u_ciljanoj"] else "✗ DRUGA KAT."
        neakt = "  [NEAKTIVAN]" if str(r["status"]).lower().startswith("neaktiv") else ""
        dok = r["dokument"][:38]
        print(f"    {r['rank']}. {dok:38} | {r['kategorija'][:26]:26} "
              f"| str.{str(r['stranica']):>4} | d={r['score']:.3f} | {flag}{neakt}")


def purity(redovi):
    if not redovi:
        return 0.0
    return sum(1 for r in redovi if r["u_ciljanoj"]) / len(redovi)


def main():
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    vectorstore = Chroma(persist_directory=str(CHROMA_DIR), embedding_function=embeddings)

    # Validacija da ciljane kategorije postoje u bazi (zaštita od tipfelera)
    postojece = set()
    try:
        for m in (vectorstore.get(include=["metadatas"])["metadatas"] or []):
            if m and "kategorija" in m:
                postojece.add(m["kategorija"])
    except Exception:
        pass
    print("Kategorije u bazi:", sorted(postojece) or "(nepoznato)")
    print("=" * 92)

    sazetak = []
    spremi = []

    for t in TESTNA:
        q, kat = t["pitanje"], t["kategorija"]
        if postojece and kat not in postojece:
            print(f"\n[{t['id']}] UPOZORENJE: kategorija {kat!r} ne postoji u bazi — provjeri string.")
        print(f"\n[{t['id']}]  {q}")
        print(f"  Ciljana kategorija: {kat!r}")

        # (A) bez filtra
        bez = vectorstore.similarity_search_with_score(q, k=K)
        red_bez = [_red(i + 1, d, s, kat) for i, (d, s) in enumerate(bez)]

        # (B) s filtrom po kategoriji
        # (Ako tvoja verzija langchain-chroma traži operator: {"kategorija": {"$eq": kat}})
        sfil = vectorstore.similarity_search_with_score(q, k=K, filter={"kategorija": kat})
        red_fil = [_red(i + 1, d, s, kat) for i, (d, s) in enumerate(sfil)]

        print()
        _ispis("(A) BEZ filtra — čisto semantički:", red_bez, kat)
        print()
        _ispis("(B) S filtrom po kategoriji:", red_fil, kat)

        p_bez, p_fil = purity(red_bez), purity(red_fil)
        print(f"\n  Kategorijska čistoća:  bez filtra = {p_bez:.2f}   |   s filtrom = {p_fil:.2f}")
        print("-" * 92)

        sazetak.append((t["id"], p_bez, p_fil))
        spremi.append({
            "id": t["id"], "pitanje": q, "ciljana_kategorija": kat,
            "purity_bez_filtra": p_bez, "purity_s_filtrom": p_fil,
            "dohvat_bez_filtra": red_bez, "dohvat_s_filtrom": red_fil,
        })

    print("\nSAŽETAK (kategorijska čistoća po pitanju):")
    print(f"  {'ID':16} {'bez filtra':>10} {'s filtrom':>10}")
    for pid, pb, pf in sazetak:
        print(f"  {pid:16} {pb:>10.2f} {pf:>10.2f}")

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(spremi, f, ensure_ascii=False, indent=2)
    print(f"\nDetaljan ispis (oba dohvaćena skupa) spremljen u: {OUT_JSON}")
    print("→ za formalni RAGAS Context Precision: učitaj 'dohvat_bez_filtra' i 'dohvat_s_filtrom'")
    print("  kao 'contexts' i pokreni RAGAS na isti način kao u run_evaluation.py.")


if __name__ == "__main__":
    main()