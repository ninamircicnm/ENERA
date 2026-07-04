"""
repeated_query_test.py — Testiranje stabilnosti i robusnosti odgovora

Pokriva dimenziju "Stabilnost i robusnost" iz evaluacijskog okvira
(Strahonja & Oreški):
  - Konzistentnost odgovora (repeated query testing)
  - Detekcija halucinacija (SelfCheckGPT-inspirirana metodologija)
  - Error Detection / Learning to refuse (out-of-domain pitanja)

Pristup:
  1. REPEATED QUERY TEST — odabrani podskup pitanja iz EVAL_DATASET
     postavlja se N puta zaredom (isti upit, ista konfiguracija modela).
     Bilježi se varira li sadržaj odgovora (npr. model jednom odgovori,
     drugi put odbije) — ovo je upravo opaženo ponašanje Gemma 3 4B
     modela ("lost in the middle" / inkonzistentno odbijanje).

  2. SELFCHECKGPT-STIL PROVJERA — za svaki upit iz repeated testa,
     odgovori se uspoređuju međusobno.
     Odgovori koji se NE pojavljuju konzistentno kroz većinu generiranja
     označavaju se kao potencijalno nepouzdani (halucinacije ili
     nestabilni zaključci). Ovo je pojednostavljena ručna verzija punog
     SelfCheckGPT pristupa (Manakul et al., 2023), prilagođena opsegu
     prototipa.

  3. OUT-OF-DOMAIN TEST (Error Detection) — pitanja izvan domene
     dokumenata; provjerava se odbija li sustav ispravno odgovoriti umjesto da halucinira.

Korištenje (iz korijena projekta, s aktivnim venv (kod mene se moralo na Python 3.10 jer je stabilniji)):
    py evaluation/repeated_query_test.py
"""

import json
import csv
import statistics
from pathlib import Path
from datetime import datetime

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.rag_pipeline import get_answer_with_sources
from evaluation.eval_dataset import EVAL_DATASET

# Konfiguracija testa 

N_PONAVLJANJA = 4

# Podskup pitanja za repeated query test — po jedno iz svake kategorije je dovoljno za prototip 
# # ID-evi se referenciraju na EVAL_DATASET
ODABRANA_PITANJA_ID = ["ZOI-01", "PRI-03", "MET-03", "VEN-01", "JPO-02"]

# Pitanja izvan domene — za Error Detection / Learning to refuse
OUT_OF_DOMAIN = [
    {
        "id": "OOD-01",
        "question": "Koji su porezni propisi za isplatu dividende dioničarima u Hrvatskoj?",
        "razlog": "Porezno pravo nije pokriveno domenskim dokumentima",
    },
    {
        "id": "OOD-02",
        "question": "Kako instalirati solarne panele na jedrilici za pogon motora?",
        "razlog": "Pomorska/brodska primjena nije tema dokumenata",
    },
    {
        "id": "OOD-03",
        "question": "Kolika je trenutna burzovna cijena električne energije na HUPX-u danas?",
        "razlog": "Sustav nema pristup live tržišnim podacima, samo statičkim dokumentima",
    },
    {
        "id": "OOD-04",
        "question": "Koji je najbolji recept za mafine?",
        "razlog": "Potpuno izvan domene — kontrolno pitanje",
    },
]

# Riječi/fraze koje upućuju na to da je model odbio odgovoriti
ODBIJANJE_OKIDACI = [
    # Originalne fraze
    "ne mogu odgovoriti",
    "nemam dovoljno informacija",
    "nije navedeno u kontekstu",
    "kontekst ne sadrži",
    "ne mogu pronaći",
    "nije obuhvaćeno",
    "izvan dosega",
    "ne odnosi se na",
    # Fraze koje Gemma 3 4B stvarno koristi (utvrđeno OOD testom)
    "nema informacija",
    "ne postoji informacija",
    "nema relevantnih informacija",
    "nema relevantnih izvora",
    "kontekst se fokusira",
    "dokumentacija se odnosi",
    "dokumentacija se fokusira",
]

def je_odbijanje(odgovor: str) -> bool:
    odgovor_lower = odgovor.lower()
    return any(okidac in odgovor_lower for okidac in ODBIJANJE_OKIDACI)


# 1. Repeated query test

def pokreni_repeated_query_test(eval_dataset: list, odabrani_id: list, n: int) -> list:
    pitanja = [item for item in eval_dataset if item["id"] in odabrani_id]

    print(f"REPEATED QUERY TEST — {len(pitanja)} pitanja x {n} ponavljanja")

    rezultati = []
    for item in pitanja:
        print(f"\n--- {item['id']}: {item['question'][:60]}...")
        odgovori = []
        for i in range(n):
            try:
                res = get_answer_with_sources(item["question"])
                odgovor = res.get("answer", "")
            except Exception as e:
                odgovor = f"[GREŠKA: {e}]"
            odgovori.append(odgovor)
            odbijen = je_odbijanje(odgovor)
            print(f"  Pokušaj {i+1}/{n}: {'ODBIJANJE' if odbijen else 'ODGOVOR'} "
                  f"({len(odgovor)} znakova)")

        n_odbijanja = sum(1 for o in odgovori if je_odbijanje(o))
        konzistentnost = "STABILNO" if n_odbijanja in (0, n) else "NESTABILNO"

        rezultati.append({
            "id": item["id"],
            "question": item["question"],
            "odgovori": odgovori,
            "n_odbijanja": n_odbijanja,
            "n_ukupno": n,
            "konzistentnost": konzistentnost,
        })
        print(f"  >> {konzistentnost} ({n_odbijanja}/{n} odbijanja)")

    return rezultati


# 2. SelfCheckGPT-stil provjera (pojednostavljena)

# Prag koeficijenta varijacije iznad kojeg se duljina smatra varijabilnom.
# CV = std / mean; vrijednost 0.25 znači da su odgovori međusobno različiti za više od 25% prosječne duljine — signal sadržajne nestabilnosti.
CV_PRAG = 0.25

def izracunaj_cv_duljine(odgovori: list) -> float:
    """
    Koeficijent varijacije (CV) duljina odgovora za jedno pitanje.
    CV = std(duljine) / mean(duljine)
    Vrijednosti bliže 0 znače ujednačene odgovore; visok CV upućuje
    na značajnu varijabilnost opsega (potencijalni znak nestabilnosti).
    Vraća 0.0 ako postoji samo jedan odgovor ili su svi iste duljine.
    """
    duljine = [len(o) for o in odgovori]
    if len(duljine) < 2 or statistics.mean(duljine) == 0:
        return 0.0
    return statistics.stdev(duljine) / statistics.mean(duljine)


def selfcheck_analiza(repeated_rezultati: list) -> list:
    """
    Pojednostavljena SelfCheckGPT analiza s dvije dimenzije konzistentnosti:

    1. KONZISTENTNOST TIPA ODGOVORA — je li skup odgovora konzistentan
       (svi odgovaraju ILI svi odbijaju). Mješoviti ishod ukazuje na
       nestabilnost modela pri istom upitu i kontekstu.

    2. KONZISTENTNOST DULJINE ODGOVORA — koeficijent varijacije (CV)
       duljina odgovora. Visok CV (> 0.25) upućuje na varijabilnost
       opsega što može odražavati sadržajnu nestabilnost — model u
       nekim pokušajima daje detaljniji, u drugima kraći odgovor na
       isto pitanje s istim dohvaćenim kontekstom.

    Ovo NE zamjenjuje punu SelfCheckGPT implementaciju (koja bi rečenicu-
    po-rečenicu uspoređivala semantičku podudarnost), nego daje grubu,
    ali za prototip dovoljno informativnu, mjeru nestabilnosti na razini
    cijelog odgovora. Semantička podudaranost se provodi ručno. Puna implementacija navedena je kao smjernica za
    budući razvoj u ograničenjima sustava.
    """
    analiza = []
    for r in repeated_rezultati:
        rizik_tipa = "VISOK" if r["konzistentnost"] == "NESTABILNO" else "NIZAK"

        cv = izracunaj_cv_duljine(r["odgovori"])
        duljine = [len(o) for o in r["odgovori"]]
        rizik_duljine = "VARIJABILNO" if cv > CV_PRAG else "UJEDNAČENO"

        # Ukupni rizik: visok ako je nestabilan tip ILI visoka varijabilnost duljine
        rizik_ukupni = "VISOK" if (rizik_tipa == "VISOK" or rizik_duljine == "VARIJABILNO") else "NIZAK"

        if rizik_tipa == "VISOK":
            obrazlozenje_tipa = (
                f"{r['n_odbijanja']}/{r['n_ukupno']} pokušaja rezultiralo "
                f"odbijanjem — mješoviti ishod ukazuje na nestabilnost modela."
            )
        else:
            obrazlozenje_tipa = (
                f"Svi pokušaji ({r['n_ukupno']}/{r['n_ukupno']}) dali "
                f"dosljedan tip odgovora."
            )

        obrazlozenje_duljine = (
            f"Duljine odgovora: {duljine} znakova | "
            f"CV={cv:.2f} → {'varijabilno (>' + str(CV_PRAG) + ')' if rizik_duljine == 'VARIJABILNO' else 'ujednačeno'}"
        )

        analiza.append({
            "id": r["id"],
            "question": r["question"],
            "rizik_halucinacije": rizik_tipa,
            "cv_duljine": round(cv, 3),
            "varijabilnost_duljine": rizik_duljine,
            "rizik_ukupni": rizik_ukupni,
            "obrazlozenje": f"{obrazlozenje_tipa} {obrazlozenje_duljine}",
        })
    return analiza


# 3. Out-of-domain / Error Detection test 

def pokreni_ood_test(ood_pitanja: list) -> list:
    print(f"OUT-OF-DOMAIN TEST (Error Detection) — {len(ood_pitanja)} pitanja")

    rezultati = []
    for item in ood_pitanja:
        print(f"\n--- {item['id']}: {item['question'][:60]}...")
        try:
            res = get_answer_with_sources(item["question"])
            odgovor = res.get("answer", "")
        except Exception as e:
            odgovor = f"[GREŠKA: {e}]"

        odbijen = je_odbijanje(odgovor)
        ishod = "ISPRAVNO ODBIJENO" if odbijen else "NIJE ODBIJENO (rizik halucinacije)"
        print(f"  >> {ishod}")

        rezultati.append({
            "id": item["id"],
            "question": item["question"],
            "razlog_izvan_domene": item["razlog"],
            "odgovor": odgovor,
            "ispravno_odbijeno": odbijen,
        })

    return rezultati


# Spremanje rezultata 

def spremi_rezultate(repeated, selfcheck, ood, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # JSON s potpunim podacima (uključujući sve varijante odgovora)
    json_path = output_dir / f"stability_test_{timestamp}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({
            "repeated_query_test": repeated,
            "selfcheck_analiza": selfcheck,
            "out_of_domain_test": ood,
        }, f, ensure_ascii=False, indent=2)
    print(f"\nPotpuni rezultati: {json_path}")

    # CSV sažetak
    csv_path = output_dir / f"stability_summary_{timestamp}.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Tip testa", "ID", "Pitanje", "Rezultat", "Detalj"])
        for r in repeated:
            writer.writerow([
                "Repeated query", r["id"], r["question"],
                r["konzistentnost"], f"{r['n_odbijanja']}/{r['n_ukupno']} odbijanja"
            ])
        for s in selfcheck:
            writer.writerow([
                "SelfCheck rizik", s["id"], s["question"],
                s["rizik_ukupni"],
                f"Tip={s['rizik_halucinacije']} | CV={s['cv_duljine']:.3f} Duljina={s['varijabilnost_duljine']} | {s['obrazlozenje']}"
            ])
        for o in ood:
            writer.writerow([
                "Out-of-domain", o["id"], o["question"],
                "ISPRAVNO" if o["ispravno_odbijeno"] else "PROPUST",
                o["razlog_izvan_domene"]
            ])
    print(f"Sažetak za rad (CSV): {csv_path}")

    return json_path, csv_path


def ispisi_sazetak(repeated, selfcheck, ood):
    print("\nSAŽETAK — STABILNOST I ROBUSNOST")

    n_nestabilno = sum(1 for r in repeated if r["konzistentnost"] == "NESTABILNO")
    print(f"Repeated query test:     {n_nestabilno}/{len(repeated)} pitanja pokazalo nestabilnost tipa")

    print("\nKoeficijent varijacije duljine odgovora (CV):")
    for s in selfcheck:
        marker = " ← VARIJABILNO" if s["varijabilnost_duljine"] == "VARIJABILNO" else ""
        print(f"  {s['id']}: CV={s['cv_duljine']:.3f}{marker}")

    n_varijabilno = sum(1 for s in selfcheck if s["varijabilnost_duljine"] == "VARIJABILNO")
    print(f"Varijabilnost duljine:   {n_varijabilno}/{len(selfcheck)} pitanja iznad praga CV>{CV_PRAG}")

    n_visok_rizik = sum(1 for s in selfcheck if s["rizik_ukupni"] == "VISOK")
    print(f"Ukupni visoki rizik:     {n_visok_rizik}/{len(selfcheck)} pitanja")

    n_ispravno = sum(1 for o in ood if o["ispravno_odbijeno"])
    print(f"Out-of-domain detekcija: {n_ispravno}/{len(ood)} ispravno odbijeno")


def main():
    print("RAG ASISTENT — TEST STABILNOSTI I ROBUSNOSTI")

    repeated = pokreni_repeated_query_test(EVAL_DATASET, ODABRANA_PITANJA_ID, N_PONAVLJANJA)
    selfcheck = selfcheck_analiza(repeated)
    ood = pokreni_ood_test(OUT_OF_DOMAIN)

    ispisi_sazetak(repeated, selfcheck, ood)

    output_dir = Path(__file__).parent / "results"
    spremi_rezultate(repeated, selfcheck, ood, output_dir)


if __name__ == "__main__":
    main()