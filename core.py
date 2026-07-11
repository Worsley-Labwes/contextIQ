"""
Core RAG (Retrieval-Augmented Generation) logic shared by the CLI (app.py)
and web (app_web.py) front ends.

Pipeline:
  1. Load .txt files from docs/
  2. Split them into overlapping chunks
  3. Embed the chunks (OpenAI or Google Gemini, whichever key is configured)
  4. Store/retrieve embeddings from a local Chroma vector store (persisted to disk)
  5. On a question: retrieve the most relevant chunks, then ask the LLM to
     answer using ONLY that retrieved context (classic RAG)

Provider selection:
  - Set RAG_PROVIDER=openai, RAG_PROVIDER=google, or RAG_PROVIDER=mistral in
    .env to force a provider.
  - If not set, it auto-detects in this order: OPENAI_API_KEY -> OpenAI,
    MISTRAL_API_KEY -> Mistral, GOOGLE_API_KEY/GEMINI_API_KEY -> Google Gemini.
  Mistral is offered because it has a genuinely free "Experiment" tier on La
  Plateforme (rate-limited, no billing required) - a good option if your
  OpenAI account has run out of quota and you'd rather not add billing, or if
  Google's Gemini key migration bug (see below) is blocking you.
  This exists because Google's Gemini API is mid-migration to a new "auth key"
  format (as of mid-2026) that has a known, currently-unresolved bug rejecting
  requests from some accounts (ACCESS_TOKEN_TYPE_UNSUPPORTED). OpenAI and
  Mistral are offered as reliable alternatives so you're not blocked by that.
"""

import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader, PyPDFLoader, Docx2txtLoader, CSVLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage

load_dotenv()

# Configurable so a mounted persistent volume (e.g. a Railway Volume) can be
# pointed at these paths - without that, anything written here (uploads, the
# vector index) is lost on every redeploy/restart, since container
# filesystems are ephemeral by default on most hosting platforms.
DOCS_DIR = Path(os.getenv("DOCS_DIR", "docs"))
PERSIST_DIR = os.getenv("CHROMA_DIR", "chroma_db")
COLLECTION_NAME = "docs_qa"

# Document types that get parsed, chunked, and embedded into the knowledge base.
DOCUMENT_LOADERS = {
    ".txt": lambda p: TextLoader(str(p), encoding="utf-8").load(),
    ".md": lambda p: TextLoader(str(p), encoding="utf-8").load(),
    ".pdf": lambda p: PyPDFLoader(str(p)).load(),
    ".docx": lambda p: Docx2txtLoader(str(p)).load(),
    ".csv": lambda p: CSVLoader(str(p)).load(),
}
SUPPORTED_DOCUMENT_EXTENSIONS = set(DOCUMENT_LOADERS.keys())

# Image types handled via a vision model call instead of embeddings (see
# answer_image_question below) - these are never added to the vector store.
SUPPORTED_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".gif"}

SUPPORTED_UPLOAD_EXTENSIONS = SUPPORTED_DOCUMENT_EXTENSIONS | SUPPORTED_IMAGE_EXTENSIONS

GOOGLE_EMBEDDING_MODEL = "models/gemini-embedding-001"
GOOGLE_CHAT_MODEL = "gemini-2.5-flash"

OPENAI_EMBEDDING_MODEL = "text-embedding-3-small"
OPENAI_CHAT_MODEL = "gpt-4.1-mini"

MISTRAL_EMBEDDING_MODEL = "mistral-embed"
MISTRAL_CHAT_MODEL = "mistral-small-latest"

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
RETRIEVE_K = 4

WEB_SEARCH_ENABLED = os.getenv("ENABLE_WEB_SEARCH", "true").strip().lower() != "false"
WEB_SEARCH_RESULTS = 4

QA_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful assistant answering questions using the context "
            "provided below, which may include excerpts from the user's own "
            "documents and/or live web search results.\n\n"
            "Rules:\n"
            "- Prefer the user's documents when they contain the answer.\n"
            "- Use the web search results to answer general-knowledge or "
            "current-events questions the documents don't cover, or to fill "
            "gaps in what the documents say.\n"
            "- Answer directly and naturally, as if it's simply your own "
            "knowledge. Do NOT mention, cite, or link to your sources inside "
            "the answer (no 'according to...', no '(Source: ...)', no URLs, "
            "no file names) - just give the content of the answer itself.\n"
            "- Do not use markdown formatting like asterisks for bold/italic; "
            "write in plain prose (short paragraphs or plain '-' bullet lines "
            "are fine, but no **bold** or *italic* markers).\n"
            "- If neither source has the answer, say so plainly instead of "
            "guessing.\n\n"
            "Context:\n{context}",
        ),
        ("human", "{question}"),
    ]
)


class RAGError(RuntimeError):
    """Raised for configuration problems (e.g. missing API key)."""


def resolve_provider() -> str:
    """
    Decide which LLM/embedding provider to use.
    RAG_PROVIDER in .env overrides auto-detection.
    """
    forced = os.getenv("RAG_PROVIDER", "").strip().lower()
    if forced in {"openai", "google", "mistral"}:
        return forced

    if os.getenv("OPENAI_API_KEY"):
        return "openai"
    if os.getenv("MISTRAL_API_KEY"):
        return "mistral"
    if os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY"):
        return "google"

    raise RAGError(
        "No API key found. Add one of the following to your .env file:\n"
        "  OPENAI_API_KEY=... (get one at https://platform.openai.com/api-keys)\n"
        "  MISTRAL_API_KEY=... (free tier at https://console.mistral.ai/api-keys)\n"
        "  GOOGLE_API_KEY=... (get one at https://aistudio.google.com/app/apikey)"
    )


def load_documents(doc_dir: Path = DOCS_DIR):
    """Load all supported documents (.txt, .md, .pdf, .docx, .csv) from a directory."""
    docs = []
    if not doc_dir.exists():
        return docs
    for path in sorted(doc_dir.iterdir()):
        if not path.is_file():
            continue
        loader = DOCUMENT_LOADERS.get(path.suffix.lower())
        if loader is None:
            continue
        try:
            docs.extend(loader(path))
        except Exception as e:
            print(f"Skipping '{path.name}' - couldn't read it: {e}")
    return docs


def split_documents(documents):
    """Split documents into overlapping chunks for embedding."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
    )
    return splitter.split_documents(documents)


def get_embeddings(provider: str):
    if provider == "openai":
        from langchain_openai import OpenAIEmbeddings

        return OpenAIEmbeddings(model=OPENAI_EMBEDDING_MODEL)

    if provider == "mistral":
        from langchain_mistralai import MistralAIEmbeddings

        return MistralAIEmbeddings(model=MISTRAL_EMBEDDING_MODEL)

    from langchain_google_genai import GoogleGenerativeAIEmbeddings

    return GoogleGenerativeAIEmbeddings(model=GOOGLE_EMBEDDING_MODEL)


def get_llm(provider: str):
    if provider == "openai":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(model=OPENAI_CHAT_MODEL, temperature=0.2)

    if provider == "mistral":
        from langchain_mistralai import ChatMistralAI

        return ChatMistralAI(model=MISTRAL_CHAT_MODEL, temperature=0.2)

    from langchain_google_genai import ChatGoogleGenerativeAI

    return ChatGoogleGenerativeAI(model=GOOGLE_CHAT_MODEL, temperature=0.2)


def build_or_load_vectorstore(force_rebuild: bool = False) -> "Chroma | None":
    """
    Build a fresh Chroma vector store from docs/ if one doesn't already exist
    on disk (or if force_rebuild=True), otherwise load the persisted one.
    Returns None if there are no documents to index yet.

    The collection is namespaced by provider (docs_qa_openai / docs_qa_google)
    so switching providers never mixes incompatible embedding vectors.
    """
    provider = resolve_provider()
    embeddings = get_embeddings(provider)
    collection_name = f"{COLLECTION_NAME}_{provider}"

    # Instantiating Chroma here does NOT call the embeddings API - it's just
    # opening/creating the local sqlite-backed client, so this is always safe.
    store = Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=PERSIST_DIR,
    )

    existing_count = len(store.get(limit=1)["ids"])
    if existing_count > 0 and not force_rebuild:
        return store

    documents = load_documents()
    if not documents:
        return None

    chunks = split_documents(documents)
    try:
        store.add_documents(chunks)
    except Exception as e:
        raise RAGError(
            f"Failed to build the vector store using the '{provider}' provider "
            f"(check your API key and internet connection): {e}"
        ) from e
    return store


def add_document_to_vectorstore(vectorstore, file_path: Path):
    """
    Load a single uploaded document, split it, and add it to the given
    vectorstore (creating one first if none exists yet). Returns the
    vectorstore to use going forward (same object if one was passed in).

    Supports .txt, .md, .pdf, .docx, and .csv. Raises RAGError with a clear
    message for anything else (including images - see answer_image_question
    for those instead) so the UI can surface it.
    """
    loader = DOCUMENT_LOADERS.get(file_path.suffix.lower())
    if loader is None:
        raise RAGError(
            f"'{file_path.suffix}' documents aren't supported for the knowledge "
            f"base. Supported: {', '.join(sorted(SUPPORTED_DOCUMENT_EXTENSIONS))}"
        )

    try:
        documents = loader(file_path)
    except Exception as e:
        raise RAGError(f"Couldn't read '{file_path.name}': {e}") from e

    chunks = split_documents(documents)
    if not chunks:
        raise RAGError(f"'{file_path.name}' appears to be empty - nothing to add.")

    provider = resolve_provider()

    if vectorstore is None:
        embeddings = get_embeddings(provider)
        vectorstore = Chroma(
            collection_name=f"{COLLECTION_NAME}_{provider}",
            embedding_function=embeddings,
            persist_directory=PERSIST_DIR,
        )

    try:
        vectorstore.add_documents(chunks)
    except Exception as e:
        raise RAGError(
            f"Failed to add '{file_path.name}' to the knowledge base "
            f"(check your API key and internet connection): {e}"
        ) from e

    return vectorstore


def search_web(query: str, max_results: int = WEB_SEARCH_RESULTS) -> list[dict]:
    """
    Run a live DuckDuckGo web search. Returns [] on any failure (rate limits,
    network issues, etc.) rather than raising - web search is a nice-to-have
    enrichment, not something that should take down document Q&A if it fails.
    """
    if not WEB_SEARCH_ENABLED:
        return []

    try:
        from ddgs import DDGS

        with DDGS() as ddgs:
            raw_results = list(ddgs.text(query, max_results=max_results))
    except Exception:
        return []

    results = []
    for r in raw_results:
        results.append(
            {
                "title": r.get("title", "").strip() or "Untitled",
                "url": r.get("href") or r.get("url") or "",
                "snippet": r.get("body", "").strip(),
            }
        )
    return [r for r in results if r["url"]]


def answer_question(vectorstore, question: str) -> dict:
    """
    Retrieve relevant document chunks AND live web search results for
    `question`, then ask the LLM to answer using whichever combination of
    those is most relevant. Returns the answer plus document sources and web
    sources separately, so the UI can show where the answer came from.
    """
    retrieved = []
    if vectorstore is not None:
        try:
            retrieved = vectorstore.similarity_search(question, k=RETRIEVE_K)
        except Exception as e:
            raise RAGError(f"Failed to search your documents: {e}") from e

    web_results = search_web(question)

    if not retrieved and not web_results:
        return {
            "answer": "I couldn't find anything relevant to that question in your documents or on the web.",
            "sources": [],
            "web_sources": [],
        }

    context_sections = []
    if retrieved:
        doc_context = "\n\n---\n\n".join(doc.page_content for doc in retrieved)
        context_sections.append(f"### From your documents:\n{doc_context}")
    if web_results:
        web_context = "\n\n---\n\n".join(
            f"{r['title']}\n{r['snippet']}\n(Source: {r['url']})" for r in web_results
        )
        context_sections.append(f"### From a live web search:\n{web_context}")
    context = "\n\n".join(context_sections)

    provider = resolve_provider()
    llm = get_llm(provider)
    chain = QA_PROMPT | llm
    try:
        response = chain.invoke({"context": context, "question": question})
    except Exception as e:
        raise RAGError(f"Failed to get an answer from the '{provider}' model: {e}") from e

    doc_sources = sorted({Path(doc.metadata.get("source", "unknown")).name for doc in retrieved})
    web_sources = [{"title": r["title"], "url": r["url"]} for r in web_results]

    return {"answer": response.content, "sources": doc_sources, "web_sources": web_sources}


def answer_image_question(question: str, image_b64: str, mime_type: str) -> dict:
    """
    Answer a question about an attached image using a vision-capable call to
    the current provider's chat model. This bypasses document retrieval and
    web search entirely - it's a single-turn multimodal Q&A about the image.

    Not all providers/models handle images equally well; if the call fails,
    the error suggests trying a different provider (RAG_PROVIDER in .env)
    rather than silently guessing which ones currently support vision.
    """
    if not question.strip():
        question = "Describe what's in this image."

    provider = resolve_provider()
    llm = get_llm(provider)

    message = HumanMessage(
        content=[
            {"type": "text", "text": question},
            {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{image_b64}"}},
        ]
    )

    try:
        response = llm.invoke([message])
    except Exception as e:
        raise RAGError(
            f"The '{provider}' model couldn't process that image ({e}). "
            "Try switching providers (set RAG_PROVIDER=openai or RAG_PROVIDER=google "
            "in .env) if this one doesn't support image input."
        ) from e

    return {"answer": response.content, "sources": [], "web_sources": []}
