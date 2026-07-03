# ENERA — Energetski Ekspert i RAG Asistent

**ENERA** is a locally-executed, retrieval-augmented generation (RAG) assistant that answers domain-specific questions about Croatian renewable-energy regulation, technical standards, and funding programmes. Every answer is grounded in a curated set of authoritative documents and returned together with its sources, so each claim can be verified against the original text.

The system runs entirely on open-source components, with no external commercial API calls — a deliberate choice motivated by **data privacy** (regulatory-sensitive documents never leave the local machine) and **independence from per-query cost and service availability**.

This repository also contains the full **evaluation suite** used to assess the system's reliability across three dimensions — *answer quality*, *grounding and source attribution*, and *stability and robustness* — comparing two local generator models, **Gemma 3 4B** and **Mistral 7B**.

> Developed as an undergraduate (bachelor's) thesis at the Faculty of Organization and Informatics (FOI), University of Zagreb. The evaluation directly implements and validates the three-dimensional reliability framework proposed by Strahonja & Oreški (2026).

---

## Table of contents

- [Key features](#key-features)
- [Architecture](#architecture)
- [Technology stack](#technology-stack)
- [Repository structure](#repository-structure)
- [Knowledge base](#knowledge-base)
- [Installation](#installation)
- [Running the system](#running-the-system)
- [Evaluation](#evaluation)
- [Results at a glance](#results-at-a-glance)
- [Limitations](#limitations)
- [License](#license)
- [Acknowledgements](#acknowledgements)

---

## Key features

- **Grounded answers with citations.** The assistant answers strictly from retrieved context and appends its sources in a fixed format (`Izvor: <document>, str. <page>`), enabling direct verification.
- **Fully local execution.** Language models and embeddings run on-device via Ollama; no data leaves the machine.
- **Metadata-aware retrieval.** Each chunk carries structured metadata (document type, category, target audience, status), allowing filtering beyond plain semantic similarity.
- **Document-status handling.** Expired documents (e.g. a closed public call) are retained in the index and flagged to the user via the system prompt rather than hard-filtered out.
- **Reproducible evaluation.** A controlled comparison of two generator models, holding every other component constant, with pinned dependency versions.

---

## Architecture

The system is organised around two logically separate phases, connected by a local ChromaDB vector store.

**Indexing phase (run once).** Content-relevant pages are loaded from the domain PDFs according to a metadata table, split into chunks, embedded, and stored — together with their metadata — in ChromaDB.

**Query phase (run per question).** The user's question flows through the chat UI and backend into the RAG pipeline, where it is embedded and matched against the stored chunks. The three most similar chunks (`top_k = 3`) are combined with the system prompt and the question to form the context from which the language model generates its answer. The answer is returned with its cited sources.

```
                        ┌─────────────────────┐
   PDF documents ─────► │   Indexing phase    │
   + metadata table     │  (indexer.py)       │
                        │  load → split →     │
                        │  embed → store      │
                        └──────────┬──────────┘
                                   ▼
                          ┌──────────────────┐
                          │    ChromaDB      │
                          │  (vector store)  │
                          └──────────┬───────┘
                                     ▲
                                     │ retrieve top_k = 3
   ┌───────────┐   POST /ask   ┌─────┴──────────┐   ┌──────────────┐
   │ Streamlit │ ────────────► │  FastAPI       │ ► │ RAG pipeline │ ► answer + sources
   │ frontend  │ ◄──────────── │  backend       │   │ + Ollama LLM │
   │ (app.py)  │   JSON reply  │  (main.py)     │   └──────────────┘
   └───────────┘               └────────────────┘
```

Key parameters: chunk size **800** characters with **100**-character overlap; retrieval `top_k = 3`; generation temperature **0.2** (evaluation judge runs at **0.0** for determinism); model kept warm with `keep_alive = 10m`. The active generator model is defined in a single `GENERATOR_MODEL` constant read by both the pipeline and the evaluation scripts, guaranteeing that the model in configuration always matches the model label in results.

---

## Technology stack

| Component | Tool | Role |
|---|---|---|
| Language | Python 3.14 (app) / 3.10 (evaluation) | Single-language stack |
| Vector store | ChromaDB | Local semantic search + metadata filtering |
| Orchestration | LangChain | RAG pipeline (loading, splitting, retrieval, generation) |
| Model runtime | Ollama | Local execution of LLMs and the embedding model |
| Embeddings | `nomic-embed-text` | Vectorisation of documents and queries |
| Generator LLM | Gemma 3 4B (default) / Mistral 7B (comparison) | Answer generation |
| Backend | FastAPI + Uvicorn | REST API (`/health`, `/ask`) |
| Frontend | Streamlit | Chat interface |
| Evaluation | RAGAS | Automated RAG metrics |

---

## Repository structure

```
ENERA/
├── data/                       # Domain PDF documents (see note below — not redistributed)
│   ├── metapodaci_dokumenti.xlsx   # Metadata table driving indexing
├── chroma_db/                  # Persisted vector store (generated by indexing)
├── backend/
│   ├── main.py                 # FastAPI service: /health and /ask endpoints
│   ├── indexer.py              # Builds the knowledge base
│   ├── rag_pipeline.py         # Retrieval + generation pipeline
│   ├── test_retrieval.py       # Sanity checks for semantic search and metadata filtering
├── frontend/
│   ├── app.py                  # Streamlit chat interface
│   ├── assets/                 # Visual identity
├── evaluation/
│   ├── eval_dataset.py         # 20-question evaluation set with ground-truth answers
│   ├── run_evaluation.py       # RAGAS metrics (with the OllamaMiddleMan JSON fix)
│   ├── citation_check.py       # Semi-automatic source-attribution check
│   ├── repeated_query_test.py  # Stability, SelfCheck-style variability, out-of-domain tests
│   ├── precision_filter_demo.py# Effect of metadata filtering on retrieval precision
│   └── results/                # Raw model outputs and evaluation results
├── requirements.txt            # App environment (Python 3.14)
└── requirements_venv310.txt    # Evaluation environment (Python 3.10, pinned)
```

---

## Knowledge base

The assistant is built over five Croatian domain documents, chosen for thematic variety (methodology, technical regulation, law, manual, public call) and audience coverage (auditors, designers/contractors, citizens, investors):

1. **Metodologija provođenja energetskog pregleda zgrada (2021)** — building energy-audit methodology
2. **Tehnički propis o sustavima ventilacije, djelomične klimatizacije i klimatizacije zgrada** — ventilation/HVAC technical regulation
3. **Zakon o obnovljivim izvorima energije i visokoučinkovitoj kogeneraciji** — the core renewable-energy law
4. **Priručnik o postupcima ishođenja dozvola za OIE projekte** — permitting-procedure manual
5. **Javni poziv EnU-6/25** — public co-financing call for residential photovoltaic systems (retained as an expired-document test case)

> **The domain documents are official public publications and are *not* redistributed in this repository.** They remain the property of their respective issuers. Please obtain them from their original official sources. The MIT license below applies to the **source code only**.

---

## Installation

The project uses **two separate Python environments**. The application (backend + frontend) runs on Python 3.14; the RAGAS evaluation runs on a dedicated Python 3.10 environment, because RAGAS's async executor does not work correctly on Python 3.14 (Windows).

**Prerequisites:** [Ollama](https://ollama.com) installed and running, with the required models pulled:

```bash
ollama pull gemma3:4b
ollama pull mistral:7b       # only needed for the comparative evaluation
ollama pull nomic-embed-text
```

**Application environment (Python 3.14):**

```bash
python -m venv venv
# Windows: venv\Scripts\activate   |   Unix: source venv/bin/activate
pip install -r requirements.txt
```

**Evaluation environment (Python 3.10):**

```bash
python3.10 -m venv venv310
# Windows: venv310\Scripts\activate   |   Unix: source venv310/bin/activate
pip install -r requirements_venv310.txt
```

The evaluation environment pins a known-good dependency combination (`ragas==0.2.15` with matched `langchain` ecosystem releases) to avoid breaking changes in newer versions.

---

## Running the system

**1. Build the knowledge base** (once, after placing the PDFs in `data/`):

```bash
python indexer.py
```

**2. Start the backend** (FastAPI on Python 3.14):

```bash
uvicorn backend.main:app --reload
```

Interactive API docs are then available at `http://127.0.0.1:8000/docs` (Swagger UI).

**3. Start the frontend** (Streamlit, in a separate terminal):

```bash
streamlit run frontend/app.py
```

Open the chat interface, ask a domain question, and the assistant answers with cited sources.

---

## Evaluation

Reliability is assessed with a three-dimensional framework, each dimension operationalised by dedicated scripts (run from the Python 3.10 environment):

| Dimension | What it measures | Scripts / metrics |
|---|---|---|
| **Answer quality** | Relevance and correctness | RAGAS *Answer Relevancy* + manual *Answer Correctness* |
| **Grounding & attribution** | Faithfulness to context, source accuracy | RAGAS *Faithfulness*, *Context Precision*, *Context Recall* + `citation_check.py` |
| **Stability & robustness** | Consistency, hallucination tendency, out-of-domain detection | `repeated_query_test.py` (repeated queries, SelfCheck-style variability, OOD tests) |

The comparison is **controlled**: only the generator model changes between the Gemma and Mistral runs; the retrieval layer, embedding model, evaluation set, and system prompt are held constant, and the judge model stays fixed (Gemma 3 4B at temperature 0.0). Mistral's weaker adherence to the shared system prompt is therefore itself a comparative finding, not a confound.

```bash
python evaluation/run_evaluation.py        # RAGAS metrics
python evaluation/citation_check.py        # attribution check (semi-automatic)
python evaluation/repeated_query_test.py   # stability + OOD
python evaluation/precision_filter_demo.py # metadata-filter effect on precision
```

---

## Results at a glance

- Both models stay reliably **within the retrieved context** (Faithfulness ≈ 0.87 for Gemma, ≈ 0.90 for Mistral) and cite sources correctly (**85%** correct attribution for both).
- Despite this, roughly **40–45%** of answers are factually incorrect — almost never due to fabrication, but due to **wrong retrieval**: the model faithfully reports retrieved content, but when the wrong pages are retrieved, the answer is grounded yet incorrect.
- These failures are invisible to automated grounding metrics and only surface through **separate human correctness judgement**, validating the combined automatic-plus-manual methodology.
- Under the development constraint of **8 GB RAM**, and given its stronger stability and out-of-domain refusal behaviour, **Gemma 3 4B is preferred** — with the caveat that the system prompt was tuned for it, so part of its advantage stems from that alignment rather than the model alone.

The central takeaway: the reliability of a RAG system cannot be reduced to a single metric or to model choice; it emerges from the balance of the three dimensions, with the **retrieval layer** identified as the dominant, shared source of error.

---

## Limitations

This is a research prototype. The evaluation set of 20 questions is indicative but not statistically representative; retrieval uses a fixed `top_k = 3` with no fine-tuning of the embedding model; the RAGAS *Answer Relevancy* metric is unreliable in this bilingual setting (queries are reconstructed in English against Croatian outputs); and correctness/attribution judgements partly rely on manual assessment. See the thesis for a full discussion and directions for future work.

---

## License

This project's **source code** is released under the [MIT License](LICENSE). The domain documents used to build the knowledge base are **not** covered by this license and are not redistributed here — see [Knowledge base](#knowledge-base).

---

## Acknowledgements

Developed as a master's thesis at the **Faculty of Organization and Informatics (FOI), University of Zagreb**, under the supervision of V. Strahonja and D. Oreški. The reliability evaluation implements and validates the three-dimensional framework proposed in:

> Strahonja, V., & Oreški, D. (2026). *Methods and instruments for determining the reliability for LLM and RAG systems in renewable energy domain.* 45th International Conference on Organizational Science Development. University of Maribor Press. https://doi.org/10.18690/um.fov.3.2026.59
