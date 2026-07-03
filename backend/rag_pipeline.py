from pathlib import Path
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

BASE_DIR = Path(__file__).parent.parent
CHROMA_DIR = BASE_DIR / "chroma_db"

# Embeddings i vektorska baza
embeddings = OllamaEmbeddings(model="nomic-embed-text")
vectorstore = Chroma(
    persist_directory=str(CHROMA_DIR),
    embedding_function=embeddings
)

# Retriever — dohvaća top-4 najsličnija odlomka
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

# LLM model
llm = ChatOllama(
    model="gemma3:4b",
    temperature=0.2,
    keep_alive="10m",
    num_predict=512,
)

# Pomoćna funkcija za formatiranje dohvaćenih odlomaka
def format_context(docs):
    dijelovi = []
    for doc in docs:
        naziv = doc.metadata.get("naziv_dokumenta", "Nepoznat dokument")
        stranica = doc.metadata.get("page", "?")
        dijelovi.append(f"[{naziv}, str. {stranica}]\n{doc.page_content}")
    return "\n\n---\n\n".join(dijelovi)

# Cjelovit RAG pipeline
rag_chain = (
    {
        "context": retriever | format_context,
        "question": RunnablePassthrough()
    }
    | prompt
    | llm
    | StrOutputParser()
)

# Funkcija za dohvat izvora uz odgovor
def get_answer_with_sources(question: str) -> dict:
    # Dohvati relevantne odlomke
    docs = retriever.invoke(question)
    
    # Generiraj odgovor
    answer = rag_chain.invoke(question)
    
    # Pripremi izvore za prikaz
    sources = []
    seen = set()
    for doc in docs:
        naziv = doc.metadata.get("naziv_dokumenta", "Nepoznat")
        stranica = doc.metadata.get("page", "?")
        key = f"{naziv}_{stranica}"
        if key not in seen:
            seen.add(key)
            sources.append({
                "naziv_dokumenta": naziv,
                "stranica": stranica,
                "kategorija": doc.metadata.get("kategorija", ""),
            })
    
    return {
        "answer": answer,
        "sources": sources
    }

