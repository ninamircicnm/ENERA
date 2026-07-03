import streamlit as st
import requests

# Konfiguracija stranice
st.set_page_config(
    page_title="RAG Asistent — Obnovljivi izvori energije",
    layout="centered",
)

# Konstante
BACKEND_URL = "http://127.0.0.1:8000"

# CSS — minimalni custom stil
st.markdown("""
<style>
    /* Naslov sekcije s izvorima */
    .sources-header {
        font-size: 0.82rem;
        font-weight: 600;
        color: #555;
        margin-top: 0.6rem;
        margin-bottom: 0.25rem;
    }
    /* Jedan izvor */
    .source-chip {
        display: inline-block;
        background: #f0f4ff;
        border: 1px solid #c9d6f7;
        border-radius: 6px;
        padding: 3px 10px;
        font-size: 0.78rem;
        color: #2c4a9e;
        margin: 2px 4px 2px 0;
    }
    /* Kategorija unutar chipa */
    .source-cat {
        color: #888;
        font-size: 0.74rem;
    }
    /* Informacijska poruka */
    .info-box {
        background: #fffbe6;
        border-left: 4px solid #f5c518;
        padding: 0.5rem 0.8rem;
        border-radius: 4px;
        font-size: 0.85rem;
        color: #555;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Provjera dostupnosti backenda
@st.cache_data(ttl=300)
def backend_zdrav() -> bool:
    try:
        r = requests.get(f"{BACKEND_URL}/health", timeout=3)
        return r.status_code == 200
    except Exception:
        return False

# Inicijalizacija session state
if "poruke" not in st.session_state:
    st.session_state.poruke = []   # lista dict: {"role": "user"|"assistant", "content": str, "sources": list}

# Zaglavlje
st.title("RAG Asistent")
st.caption("Poslovni AI asistent za domenu obnovljivih izvora energije · Republika Hrvatska")

# Status backenda
if backend_zdrav():
    st.success("Backend aktivan")
else:
    st.error(
        "Backend nije dostupan. Pokrenite server naredbom:\n\n"
        "```\nuvicorn backend.main:app --reload\n```"
    )

st.divider()

# Prikaz povijesti razgovora
def prikazi_izvore(sources: list) -> str:
    """Gradi HTML za prikaz izvora ispod odgovora asistenta."""
    if not sources:
        return ""
    chips = ""
    for s in sources:
        naziv = s.get("naziv_dokumenta", "Nepoznat dokument")
        stranica = s.get("stranica", "?")
        kategorija = s.get("kategorija", "")
        cat_span = f' <span class="source-cat">· {kategorija}</span>' if kategorija else ""
        chips += f'<span class="source-chip"> {naziv}, str. {stranica}{cat_span}</span>'
    return f'<div class="sources-header">Korišteni izvori:</div>{chips}'


for poruka in st.session_state.poruke:
    with st.chat_message(poruka["role"]):
        st.markdown(poruka["content"])
        if poruka.get("sources"):
            st.markdown(prikazi_izvore(poruka["sources"]), unsafe_allow_html=True)

# Unos pitanja
upit = st.chat_input(
    "Postavite pitanje o obnovljivim izvorima energije, dozvolama, natječajima...",
    disabled=not backend_zdrav(),
)

if upit:
    # Prikaži korisnikovu poruku
    st.session_state.poruke.append({"role": "user", "content": upit, "sources": []})
    with st.chat_message("user"):
        st.markdown(upit)

    # Pozovi backend i prikaži odgovor
    with st.chat_message("assistant"):
        with st.spinner("Pretražujem dokumentaciju..."):
            try:
                response = requests.post(
                    f"{BACKEND_URL}/ask",
                    json={"question": upit},
                    timeout=180,
                )
                response.raise_for_status()
                data = response.json()

                odgovor = data.get("answer", "Nema odgovora.")
                sources = data.get("sources", [])

                st.markdown(odgovor)
                if sources:
                    st.markdown(prikazi_izvore(sources), unsafe_allow_html=True)

                # Spremi u povijest
                st.session_state.poruke.append({
                    "role": "assistant",
                    "content": odgovor,
                    "sources": sources,
                })

            except requests.exceptions.Timeout:
                st.error("Zahtjev je predugo trajao. Model možda procesira dugi kontekst — pokušajte ponovno.")
            except requests.exceptions.ConnectionError:
                st.error("Nije moguće spojiti se na backend. Provjerite je li Uvicorn pokrenut.")
            except Exception as e:
                st.error(f"Greška: {e}")

# Bočna traka — upute i primjeri upita
with st.sidebar:
    st.header("O sustavu")
    st.markdown("""
    Ovaj AI asistent odgovara **isključivo na temelju** sljedećih domenskih dokumenata:
    
    - Zakon o OIE i visokoučinkovitoj kogeneraciji  
    - Priručnik o postupcima ishođenja dozvola  
    - Metodologija energetskog pregleda zgrada  
    - Tehnički propis o sustavima ventilacije  
    - Javni poziv EnU-6/25 (fotonaponske elektrane)
    """)

    st.divider()
    st.subheader("Primjeri pitanja")
    primjeri = [
        "Koji su uvjeti za prijavu na javni poziv za ugradnju fotonaponskih elektrana?",
        "Kako se izračunava tržišna premija za obnovljive izvore energije?",
        "Koji je postupak ishođenja građevinske dozvole za sunčanu elektranu?",
        "Što obuhvaća energetski pregled zgrade i koji su mu ciljevi?",
        "Koje su obveze redovitih pregleda sustava ventilacije?",
        "Koji je nacionalni cilj udjela OIE do 2030. godine?",
    ]
    for primjer in primjeri:
        st.markdown(f"› *{primjer}*")

    st.divider()
    if st.button("Očisti razgovor"):
        st.session_state.poruke = []
        st.rerun()
