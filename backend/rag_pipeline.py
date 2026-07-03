from pathlib import Path
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

BASE_DIR = Path(__file__).parent.parent
CHROMA_DIR = BASE_DIR / "chroma_db"

# Embeddings i vektorska baza
embeddings = OllamaEmbeddings(model="nomic-embed-text")
vectorstore = Chroma(
    persist_directory=str(CHROMA_DIR),
    embedding_function=embeddings
)

# Retriever — dohvaća top-3 najsličnija odlomka
retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3}
)

# System prompt
SYSTEM_PROMPT = """Ti si stručni AI asistent za podršku poslovnom odlučivanju u domeni obnovljivih izvora energije u Republici Hrvatskoj.

Odgovaraj ISKLJUČIVO na temelju priloženog konteksta iz domenskih dokumenata.
Ako odgovor nije eksplicitno u kontekstu, izvedi zaključak na temelju dostupnih informacija i jasno naznači da se radi o zaključku temeljenom na dostupnoj dokumentaciji.
Ne koristiti opće znanje izvan priloženog konteksta.
Ne odgovaraj na pitanja koja nisu vezana uz obnovljive izvore energije, energetske preglede ili povezanu regulativu.

Ako je dohvaćeni izvor u kontekstu označen kao [NEAKTIVAN / rok istekao], jasno upozori korisnika da taj natječaj ili dokument više nije na snazi te da navedene uvjete i rokove treba provjeriti u aktualnoj verziji. Svejedno smiješ odgovoriti na temelju njegova sadržaja (npr. opisati kakvi su uvjeti bili), ali uz to upozorenje.

Uvijek odgovaraj na hrvatskom jeziku, bez obzira na jezik upita.
Odgovaraj jasno i stručno. Ako odgovor uključuje više koraka ili uvjeta, navedi ih kao numerirani popis.

Pri svakom odgovoru na kraju navedi izvore u formatu:
Izvor: [naziv dokumenta], str. [broj stranice]

Kontekst:
{context}"""

prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("human", "{question}")
])

GENERATOR_MODEL = "gemma3:4b"

# LLM model
llm = ChatOllama(
    model=GENERATOR_MODEL,
    temperature=0.2,
    keep_alive="10m",
    # num_predict=512,
)

# Broj stranice za citat: tiskana (otisnuta u dokumentu) stranica, koju indexer
# upisuje u metapodatke. Fallback na PyPDF 'page' (PDF indeks) samo ako u bazi
# ostane stari chunk od prije re-indeksiranja.
def stranica_za_citat(doc):
    s = doc.metadata.get("tiskana_stranica")
    if s is None:
        s = doc.metadata.get("page", "?")
    return s


# Pomoćna funkcija za formatiranje dohvaćenih odlomaka
def format_context(docs):
    dijelovi = []
    for doc in docs:
        naziv = doc.metadata.get("naziv_dokumenta", "Nepoznat dokument")
        stranica = stranica_za_citat(doc)
        status = (doc.metadata.get("status_dokumenta") or "")
        oznaka = " [NEAKTIVAN / rok istekao]" if status.lower().startswith("neaktiv") else ""
        dijelovi.append(f"[{naziv}, str. {stranica}{oznaka}]\n{doc.page_content}")
    return "\n\n---\n\n".join(dijelovi)

# Glavna funkcija — retriever se poziva JEDNOM, rezultat se dijeli između
# formatiranja konteksta i pripreme izvora (nema duplog pretraživanja)
def get_answer_with_sources(question: str) -> dict:
    # Jedan retrieval poziv
    docs = retriever.invoke(question)

    # Formatiraj kontekst i generiraj odgovor
    context = format_context(docs)
    chain = prompt | llm | StrOutputParser()
    answer = chain.invoke({"context": context, "question": question})

    # Pripremi izvore za prikaz (deduplicirani)
    sources = []
    seen = set()
    for doc in docs:
        naziv = doc.metadata.get("naziv_dokumenta", "Nepoznat")
        stranica = stranica_za_citat(doc)
        key = f"{naziv}_{stranica}"
        if key not in seen:
            seen.add(key)
            sources.append({
                "naziv_dokumenta": naziv,
                "stranica": stranica,
                "kategorija": doc.metadata.get("kategorija", ""),
            })

    # Sirovi tekst dohvaćenih odlomaka — POTREBNO za RAGAS evaluaciju
    # (Faithfulness, Context Precision i Context Recall trebaju stvarni
    # tekst koji je model vidio, ne samo metapodatke o izvoru)
    retrieved_chunks = [doc.page_content for doc in docs]

    return {
        "answer": answer,
        "sources": sources,
        "retrieved_chunks": retrieved_chunks
    }