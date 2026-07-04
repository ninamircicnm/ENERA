import streamlit as st
import requests
import base64
from pathlib import Path

# Pomocna funkcija za ucitavanje slika
def ucitaj_sliku_b64(naziv: str) -> str:
    putanja = Path(__file__).parent / "assets" / naziv
    with open(putanja, "rb") as f:
        return base64.b64encode(f.read()).decode()

# Konfiguracija stranice
st.set_page_config(
    page_title="ENERA — Energetski Ekspert i RAG Asistent",
    page_icon=Path(__file__).parent / "assets" / "novi_logo.JPG",
    layout="centered",
)

# CSS
st.markdown("""
<style>
    /* ── Zaglavlje ── */
    .enera-header {
        display: flex;
        align-items: center;
        gap: 16px;
        padding: 1.1rem 1.4rem;
        background: linear-gradient(135deg, #1b3f6e 0%, #1a4a6e 60%, #1a5c6e 100%);
        border-radius: 12px;
        margin-bottom: 0.5rem;
    }
    .enera-header img {
        width: 56px;
        height: 56px;
        border-radius: 10px;
        object-fit: cover;
        flex-shrink: 0;
    }
    .enera-title {
        font-size: 1.6rem;
        font-weight: 700;
        color: #ffffff;
        letter-spacing: 0.04em;
        margin: 0;
        line-height: 1.2;
    }
    .enera-subtitle {
        font-size: 0.8rem;
        color: #a8cce0;
        margin: 3px 0 0 0;
    }

    /* ── Disclaimer ── */
    .enera-disclaimer {
        background: #1e2a38;
        border: 1px solid #2d3f52;
        border-radius: 8px;
        padding: 0.45rem 1rem;
        font-size: 0.65rem;
        color: #8aa4b8;
        text-align: center;
        margin-bottom: 1rem;
    }

    /* ── Welcome screen ── */
    .welcome-card {
        background: linear-gradient(135deg, #1b3f6e 0%, #1a5c6e 100%);
        border-radius: 14px;
        padding: 2rem 2rem 1.6rem 2rem;
        text-align: center;
        margin: 1rem 0 1.5rem 0;
        border: 1px solid #2d5a7a
    }
    .welcome-card h2 {
        color: #e8f4fd;
        font-size: 1.25rem;
        margin: 0 0 0.6rem 0;
    }
    .welcome-card p {
        color: #b8d8ea;
        font-size: 0.88rem;
        line-height: 1.65;
        margin: 0 0 1.2rem 0;
    }
    .welcome-pills {
        display: flex;
        flex-wrap: wrap;
        gap: 8px;
        justify-content: center;
    }
    .welcome-pill {
        background: rgba(255, 255, 255, 0.1);
        border: 1px solid rgba(255, 255, 255, 0.2);
        border-radius: 20px;
        padding: 5px 14px;
        font-size: 0.78rem;
        color: #d0ecf8;
    }

    /* ── Chip za izvore ── */
    .sources-header {
        font-size: 0.75rem;
        font-weight: 600;
        color: var(--text-color, #777);
        opacity: 0.7;
        margin-top: 0.9rem;
        margin-bottom: 0.3rem;
        text-transform: uppercase;
        letter-spacing: 0.06em;
    }
    .source-chip {
        display: inline-block;
        border: 1px solid rgba(100, 150, 200, 0.35);
        border-radius: 6px;
        padding: 3px 10px;
        font-size: 0.75rem;
        color: #4a90c4;
        margin: 2px 4px 2px 0;
        background: rgba(74, 144, 196, 0.08);
    }
    .source-cat {
        font-size: 0.71rem;
        opacity: 0.65;
    }

    /* ── Sidebar logo ── */
    .sidebar-logo {
        text-align: center;
        padding: 0.3rem 0 0.8rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Konstante
BACKEND_URL = "http://127.0.0.1:8000"

# Provjera backenda
@st.cache_data(ttl=300)
def backend_zdrav() -> bool:
    try:
        r = requests.get(f"{BACKEND_URL}/health", timeout=3)
        return r.status_code == 200
    except Exception:
        return False

# Inicijalizacija session state
if "poruke" not in st.session_state:
    st.session_state.poruke = []

# Zaglavlje
logo_b64 = ucitaj_sliku_b64("logo.jpg")
st.markdown(f"""
<div class="enera-header">
    <img src="data:image/jpeg;base64,{logo_b64}" alt="ENERA logo">
    <div>
        <p class="enera-title">ENERA</p>
        <p class="enera-subtitle">ENergetski Ekspert i RAG Asistent &nbsp;·&nbsp; Obnovljivi izvori energije &nbsp;·&nbsp; RH</p>
    </div>
</div>
""", unsafe_allow_html=True)

# Disclaimer
st.markdown("""
<div class="enera-disclaimer">
    ENERA je AI asistent — odgovori se temelje isključivo na indeksiranim dokumentima i mogu sadržavati pogreške.
    Uvijek provjerite informacije u izvornoj dokumentaciji prije donošenja poslovnih odluka.
</div>
""", unsafe_allow_html=True)

# Status backenda
if not backend_zdrav():
    st.error(
        "Backend nije dostupan. Pokrenite server naredbom:\n\n"
        "```\nuvicorn backend.main:app --reload\n```"
    )

# Welcome screen
if not st.session_state.poruke:
    st.markdown("""
<div class="welcome-card">
    <h2>Dobrodošli u ENERA-u</h2>
    <p>
        Postavite pitanje o obnovljivim izvorima energije, energetskim pregledima,<br>
        postupcima ishođenja dozvola ili dostupnim programima financiranja.<br>
        Odgovori se temelje isključivo na verificiranoj domenskoj dokumentaciji.
    </p>
    <div class="welcome-pills">
        <span class="welcome-pill">Zakon o OIE</span>
        <span class="welcome-pill">Dozvole i procedure</span>
        <span class="welcome-pill">Energetski pregledi</span>
        <span class="welcome-pill">Tehnički propisi</span>
        <span class="welcome-pill">Javni pozivi</span>
    </div>
</div>
""", unsafe_allow_html=True)

# Pomocna funkcija — prikaz izvora
KATEGORIJA_BOJA = {
    "Zakonodavstvo":            "#1a4a6e",
    "Pravni okvir i postupci":  "#1a4a6e",
    "Procedure i dozvole":      "#2d5a3a",
    "Financiranje i potpore":   "#4a3a0f",
    "Energetska učinkovitost":  "#2a3a5a",
    "Mjere obnove zgrada":      "#2a3a5a",
    "Tehnički propisi":         "#3a2a5a",
}

# Pomocna funkcija — prikaz izvora
def prikazi_izvore(sources: list) -> str:
    if not sources:
        return ""
    chips = ""
    for s in sources:
        naziv = s.get("naziv_dokumenta", "Nepoznat dokument")
        stranica = s.get("stranica", "?")
        kategorija = s.get("kategorija", "")
        boja = KATEGORIJA_BOJA.get(kategorija, "#1e2a38")
        cat_span = f' <span class="source-cat">&nbsp;· {kategorija}</span>' if kategorija else ""
        chips += (
            f'<span class="source-chip" style="border-color:{boja}; background:{boja}22;">'
            f'{naziv}, str. {stranica}{cat_span}</span>'
        )    
    return f'<div class="sources-header">Korišteni izvori</div>{chips}'

# Prikaz povijesti razgovora
for poruka in st.session_state.poruke:
    with st.chat_message(poruka["role"]):
        st.markdown(poruka["content"])
        if poruka.get("sources"):
            st.markdown(prikazi_izvore(poruka["sources"]), unsafe_allow_html=True)

# Unos pitanja
upit = st.chat_input(
    "Postavite pitanje o OIE, dozvolama, energetskim pregledima...",
    disabled=not backend_zdrav(),
)

if upit:
    # Dodaj u session_state PRIJE renderiranja — welcome screen nestaje odmah
    st.session_state.poruke.append({"role": "user", "content": upit, "sources": []})

    with st.chat_message("user"):
        st.markdown(upit)

    with st.chat_message("assistant"):
        with st.spinner("ENERA pretražuje dokumentaciju..."):
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

                st.session_state.poruke.append({
                    "role": "assistant",
                    "content": odgovor,
                    "sources": sources,
                })

            except requests.exceptions.Timeout:
                st.error("Zahtjev je predugo trajao. Model procesira dugi kontekst — pokušajte ponovno.")
            except requests.exceptions.ConnectionError:
                st.error("Nije moguće spojiti se na backend. Provjerite je li Uvicorn pokrenut.")
            except Exception as e:
                st.error(f"Greška: {e}")

# Sidebar
with st.sidebar:
    try:
        novi_logo_b64 = ucitaj_sliku_b64("novi_logo.jpeg")
        st.markdown(
            f'<div class="sidebar-logo">'
            f'<img src="data:image/jpeg;base64,{novi_logo_b64}" '
            f'width="90" style="border-radius:14px;"></div>',
            unsafe_allow_html=True
        )
    except FileNotFoundError:
        pass

    st.markdown("### ENERA")
    st.caption("ENergetski Ekspert i RAG Asistent")
    st.divider()

    st.markdown("**Indeksirani dokumenti:**")
    st.markdown("""
- Zakon o OIE i visokoučinkovitoj kogeneraciji
- Priručnik o postupcima ishođenja dozvola
- Metodologija energetskog pregleda zgrada
- Tehnički propis o sustavima ventilacije
- Javni poziv EnU-6/25
""")

    st.divider()
    st.markdown("**Primjeri pitanja:**")
    primjeri = [
        "Koji su uvjeti za prijavu na javni poziv za ugradnju fotonaponskih elektrana?",
        "Kako se izračunava tržišna premija za OIE?",
        "Koji je postupak ishođenja dozvole za sunčanu elektranu?",
        "Što obuhvaća energetski pregled zgrade?",
        "Koje su obveze redovitih pregleda sustava ventilacije?",
        "Koji je nacionalni cilj udjela OIE do 2030.?",
    ]
    for primjer in primjeri:
        st.markdown(f"› *{primjer}*")

    st.divider()
    if st.button("Očisti razgovor", use_container_width=True):
        st.session_state.poruke = []
        st.rerun()