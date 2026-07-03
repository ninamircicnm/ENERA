
# skripta za indeksiranje baze

import pandas as pd
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma

# Putanje
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
CHROMA_DIR = BASE_DIR / "chroma_db"
METADATA_FILE = BASE_DIR / "data" / "metapodaci_dokumenti.xlsx"

# Učitaj metapodatke iz Excela
df = pd.read_excel(METADATA_FILE)

# Konfiguriraj splitter
splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=100,
    separators=["\n\n", "\n", ".", " ", ""]
)

# Konfiguriraj embeddings
embeddings = OllamaEmbeddings(model="nomic-embed-text")

# Pripremi sve dokumente s metapodacima
all_chunks = []

for _, row in df.iterrows():
    pdf_path = DATA_DIR / row["naziv_datoteke"]

    if not pdf_path.exists():
        print(f"UPOZORENJE: {pdf_path} ne postoji, preskačem.")
        continue

    # pdf_od / pdf_do = stvaran redni broj stranice u PDF datoteci (1-based).
    # stranica_od / stranica_do = broj OTISNUT na stranici (ono što vidi korisnik).
    # Offset je razlika između te dvije numeracije za ovaj raspon
    pdf_od = int(row["pdf_od"])
    pdf_do = int(row["pdf_do"])
    tiskana_od = int(row["stranica_od"])
    offset = pdf_od - tiskana_od  

    print(f"Učitavam: {row['naziv_datoteke']} "
          f"(PDF {pdf_od}–{pdf_do} = tiskane {tiskana_od}–{int(row['stranica_do'])}, offset {offset})")

    # Učitaj PDF
    loader = PyPDFLoader(str(pdf_path))
    pages = loader.load()

    # Filtriraj po FIZIČKIM PDF stranicama (PyPDFLoader 'page' je 0-based).
    relevant_pages = [
        p for p in pages
        if pdf_od <= (p.metadata.get("page", 0) + 1) <= pdf_do
    ]

    # Podijeli na chunkove
    chunks = splitter.split_documents(relevant_pages)

    # Dodaj metapodatke iz Excel tablice na svaki chunk
    for chunk in chunks:
        pdf_stranica = chunk.metadata.get("page", 0) + 1  # fizička PDF stranica (1-based)
        tiskana_stranica = pdf_stranica - offset          # broj otisnut na stranici
        chunk.metadata.update({
            "id": str(row["id"]),
            "naziv_dokumenta": row["naziv_dokumenta"],
            "naziv_datoteke": row["naziv_datoteke"],
            "vrsta_dokumenta": row["vrsta_dokumenta"],
            "kategorija": row["kategorija"],
            "ciljana_skupina": row["ciljana_skupina"],
            "status_dokumenta": row["status_dokumenta"],
            "pdf_stranica": int(pdf_stranica),          # za internu provjeru / debug
            "tiskana_stranica": int(tiskana_stranica),  # ZA CITAT prema korisniku
        })

    all_chunks.extend(chunks)
    print(f"  → {len(chunks)} chunkova")

print(f"\nUkupno chunkova za indeksiranje: {len(all_chunks)}")

# Pohrani u ChromaDB
print("Pohranjujem u ChromaDB...")
vectorstore = Chroma.from_documents(
    documents=all_chunks,
    embedding=embeddings,
    persist_directory=str(CHROMA_DIR)
)

print(f"Indeksiranje završeno. Ukupno zapisa u bazi: {vectorstore._collection.count()}")
