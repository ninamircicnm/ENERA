from pathlib import Path
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma

BASE_DIR = Path(__file__).parent.parent
CHROMA_DIR = BASE_DIR / "chroma_db"

embeddings = OllamaEmbeddings(model="nomic-embed-text")
vectorstore = Chroma(
    persist_directory=str(CHROMA_DIR),
    embedding_function=embeddings
)

print(f"Ukupno zapisa u ChromaDB: {vectorstore._collection.count()}\n")

# Test 1: Semantičko pretraživanje
print("=== TEST 1: Semantičko pretraživanje ===")
upit = "uvjeti za ugradnju fotonaponskih elektrana"
rezultati = vectorstore.similarity_search(upit, k=3)
for i, r in enumerate(rezultati, 1):
    print(f"\n[{i}] Dokument: {r.metadata.get('naziv_dokumenta')}")
    print(f"    Kategorija: {r.metadata.get('kategorija')}")
    print(f"    Tekst: {r.page_content[:200]}...")

# Test 2: Metadata filtering
print("\n=== TEST 2: Metadata filtering ===")
rezultati_filter = vectorstore.similarity_search(
    "minimalni uvjeti i zahtjevi",
    k=3,
    filter={"kategorija": "Obnovljivi izvori energije"}  # prilagodi prema svojoj tablici
)
for i, r in enumerate(rezultati_filter, 1):
    print(f"\n[{i}] Dokument: {r.metadata.get('naziv_dokumenta')}")
    print(f"    Kategorija: {r.metadata.get('kategorija')}")
    print(f"    Tekst: {r.page_content[:200]}...")