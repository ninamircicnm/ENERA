"""
citation_check.py — Provjera točnosti citiranja izvora (Attribution Accuracy)

Pokriva dimenziju "Utemeljenost i atribucija" iz evaluacijskog okvira
(Strahonja & Oreški), konkretno metriku Attribution Accuracy koja u RAGAS-u nije pokrivena.

Pristup je polu-automatski:
  1. Automatska provjera (programska) — je li format "Izvor: ..." prisutan
     u odgovoru, i poklapa li se barem jedan navedeni dokument s nekim od
     stvarno dohvaćenih izvora (sources_meta) za to pitanje.
  2. Ručna ocjena (popunjava se naknadno) — je li navedena stranica zaista
     sadržajno točna, ne samo formalno prisutna.

Korištenje (iz korijena projekta, s aktivnim venv):
    py evaluation/citation_check.py evaluation/results/raw_results_<timestamp>.json

Ulaz: raw_results_<timestamp>.json koji generira run_evaluation.py
      (sadrži question, generated_answer, sources_meta po pitanju)

Izlaz: evaluation/results/citation_check_<timestamp>.csv
       — tablica spremna za ručno dopunjavanje stupca "rucna_ocjena"
"""

import sys
import csv
import json
import re
from pathlib import Path
from datetime import datetime


# Rubrika za ručnu ocjenu — koristi se kao legenda pri popunjavanju CSV-a
CITATION_RUBRIC = {
    "tocno":       "Naveden dokument i stranica odgovaraju stvarnom sadržaju odgovora",
    "djelomicno":  "Dokument je točan, ali stranica nije precizna ili nedostaje",
    "netocno":     "Naveden izvor ne podržava sadržaj odgovora (krivi dokument)",
    "nema_citata": "Odgovor ne sadrži naveden izvor u traženom formatu",
}

# Regex za prepoznavanje formata "Izvor: [naziv], str. [broj]" iz system prompta
IZVOR_PATTERN = re.compile(r"Izvor:\s*(.+?),?\s*\bstr\.?\s*(\d+(?:\s*[–-]\s*\d+)?)", re.IGNORECASE)

def ucitaj_rezultate(path: Path) -> list:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def izvuci_navedene_izvore(generated_answer: str) -> list:
    """
    Parsira tekst odgovora i izvlači sve parove (naziv_dokumenta, stranica)
    koje je model naveo u formatu "Izvor: ..., str. ...".
    Model može navesti više izvora u istom odgovoru.
    """
    nadeni = IZVOR_PATTERN.findall(generated_answer)
    return [{"naziv": naziv.strip().strip("[]").strip(), "stranica": str(stranica).strip()} for naziv, stranica in nadeni]

def provjeri_poklapanje(navedeni_izvori: list, sources_meta: list) -> str:
    """
    Automatska provjera (gruba, programska razina):
      - "nema_citata"   ako model nije naveo nijedan izvor u tekstu
      - "tocno"         ako se barem jedan navedeni dokument+stranica
                        poklapa s nečim što je stvarno dohvaćeno
      - "djelomicno"    ako se dokument poklapa, ali stranica ne
      - "netocno"       ako se nijedan navedeni dokument ne poklapa
                        s dohvaćenim izvorima

    Napomena: ovo je AUTOMATSKA pred-ocjena radi uštede vremena.
    Konačna ocjena za rad treba biti ručno potvrđena (vidi rucna_ocjena
    stupac u izlaznom CSV-u), jer programska usporedba naziva dokumenata
    može biti netočna zbog kraćenja/varijacija naziva.
    """
    if not navedeni_izvori:
        return "nema_citata"

    dokumenti_dohvaceni = {s.get("naziv_dokumenta", "").lower() for s in sources_meta}
    stranice_po_dokumentu = {}
    for s in sources_meta:
        dok = s.get("naziv_dokumenta", "").lower()
        stranice_po_dokumentu.setdefault(dok, set()).add(str(s.get("stranica", "")).lower())

    dokument_pogodak = False
    stranica_pogodak = False

    for izvor in navedeni_izvori:
        naziv_lower = izvor["naziv"].lower()
        # Djelomično podudaranje naziva (model ponekad krati naziv dokumenta)
        for dok in dokumenti_dohvaceni:
            if naziv_lower in dok or dok in naziv_lower:
                dokument_pogodak = True
                if izvor["stranica"].lower() in stranice_po_dokumentu.get(dok, set()):
                    stranica_pogodak = True

    if dokument_pogodak and stranica_pogodak:
        return "tocno"
    elif dokument_pogodak:
        return "djelomicno"
    else:
        return "netocno"


def pokreni_citation_check(rezultati: list) -> list:
    """Vraća listu redaka spremnih za CSV, s automatskom pred-ocjenom."""
    redovi = []
    for r in rezultati:
        odgovor = r.get("generated_answer", "")
        sources_meta = r.get("sources_meta", [])

        navedeni = izvuci_navedene_izvore(odgovor)
        auto_ocjena = provjeri_poklapanje(navedeni, sources_meta)

        redovi.append({
            "id": r["id"],
            "question": r["question"],
            "navedeni_izvori": "; ".join(
                f"{i['naziv']}, str. {i['stranica']}" for i in navedeni
            ) or "(nema)",
            "dohvaceni_izvori": "; ".join(
                f"{s.get('naziv_dokumenta', '')}, str. {s.get('stranica', '')}"
                for s in sources_meta
            ) or "(nema)",
            "auto_ocjena": auto_ocjena,
            "rucna_ocjena": "",  # popunjava se naknadno
            "napomena": "",      # prostor za komentar pri ručnoj provjeri
        })
    return redovi


def spremi_csv(redovi: list, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = output_dir / f"citation_check_{timestamp}.csv"

    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, delimiter=";", fieldnames=[
            "id", "question", "navedeni_izvori", "dohvaceni_izvori",
            "auto_ocjena", "rucna_ocjena", "napomena"
        ])
        writer.writeheader()
        writer.writerows(redovi)

    return path


def ispisi_sazetak(redovi: list):
    ukupno = len(redovi)
    brojac = {"tocno": 0, "djelomicno": 0, "netocno": 0, "nema_citata": 0}
    for r in redovi:
        brojac[r["auto_ocjena"]] += 1

    print("AUTOMATSKA PRED-OCJENA CITIRANJA (Attribution Accuracy)")
    for kljuc, opis in CITATION_RUBRIC.items():
        udio = brojac[kljuc] / ukupno * 100 if ukupno else 0
        print(f"  {kljuc:13s}: {brojac[kljuc]:2d}/{ukupno}  ({udio:5.1f}%)  — {opis}")
    print(f"\nNAPOMENA: ovo je automatska pred-ocjena. Za rad je potrebno")
    print(f"ručno potvrditi/ispraviti stupac 'rucna_ocjena' u CSV-u, jer")
    print(f"programsko poklapanje naziva dokumenata može biti netočno.")


def main():
    if len(sys.argv) < 2:
        print("Korištenje: py evaluation/citation_check.py <putanja_do_raw_results.json>")
        sys.exit(1)

    input_path = Path(sys.argv[1])
    if not input_path.exists():
        print(f"Datoteka ne postoji: {input_path}")
        sys.exit(1)

    rezultati = ucitaj_rezultate(input_path)
    redovi = pokreni_citation_check(rezultati)

    output_dir = Path(__file__).parent / "results"
    csv_path = spremi_csv(redovi, output_dir)

    ispisi_sazetak(redovi)
    print(f"\nIzvještaj spremljen: {csv_path}")
    print("Otvori CSV i popuni stupac 'rucna_ocjena' za konačnu evaluaciju.")


if __name__ == "__main__":
    main()
