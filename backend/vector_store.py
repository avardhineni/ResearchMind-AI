"""ResearchMind AI: session-isolated Qdrant storage for research documents.

Supports uploaded files, arXiv papers, and Semantic Scholar open-access PDFs.
Existing Papeer collections remain readable without migration.
"""

import os

from dotenv import load_dotenv
from langchain_classic.embeddings import CacheBackedEmbeddings
from langchain_classic.storage import LocalFileStore
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

load_dotenv()

EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIM = 1536
NEW_COLLECTION_PREFIX = "researchmind_"
LEGACY_COLLECTION_PREFIX = "papeer_"

base_embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
embedding_file_store = LocalFileStore("./embedding_cache/")
embeddings = CacheBackedEmbeddings.from_bytes_store(
    base_embeddings,
    embedding_file_store,
    namespace=base_embeddings.model,
    query_embedding_cache=True,
    key_encoder="blake2b",
)

qdrant_client = QdrantClient(
    url=os.environ["QDRANT_URL"],
    api_key=os.environ["QDRANT_API_KEY"],
    timeout=120,
)


def _session_suffix(session_id: str) -> str:
    """Match the original Papeer session-ID normalization."""
    if not isinstance(session_id, str) or not session_id.strip():
        raise ValueError("A non-empty session_id is required.")
    return session_id.replace("-", "_")


def get_collection_name(session_id: str) -> str:
    """Return the active collection for a session, preferring legacy data.

    A pre-existing Papeer collection continues receiving new documents for that
    session. New sessions use the ResearchMind AI prefix. This avoids splitting
    one session's documents across two collections during the rename.
    """
    suffix = _session_suffix(session_id)
    legacy = f"{LEGACY_COLLECTION_PREFIX}{suffix}"
    if qdrant_client.collection_exists(legacy):
        return legacy
    return f"{NEW_COLLECTION_PREFIX}{suffix}"


def _existing_collection_name(session_id: str) -> str | None:
    """Resolve an existing collection without creating an empty one."""
    suffix = _session_suffix(session_id)
    legacy = f"{LEGACY_COLLECTION_PREFIX}{suffix}"
    current = f"{NEW_COLLECTION_PREFIX}{suffix}"
    if qdrant_client.collection_exists(legacy):
        return legacy
    if qdrant_client.collection_exists(current):
        return current
    return None


def get_vectorstore(session_id: str) -> QdrantVectorStore:
    """Get or create the session's vector store."""
    collection_name = get_collection_name(session_id)
    if not qdrant_client.collection_exists(collection_name):
        qdrant_client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE),
        )
    return QdrantVectorStore(
        client=qdrant_client,
        collection_name=collection_name,
        embedding=embeddings,
    )


def add_paper(docs: list[Document], session_id: str) -> None:
    """Store chunked research documents and preserve their source metadata.

    Metadata such as title, source, url, DOI, authors, and publication year is
    passed through unchanged when supplied by the loader. Semantic Scholar
    PDF ingestion therefore needs no separate Qdrant collection.
    """
    if not docs:
        return
    get_vectorstore(session_id).add_documents(docs)


def list_papers(session_id: str) -> list[str]:
    """List distinct document titles in the session's collection."""
    collection_name = _existing_collection_name(session_id)
    if collection_name is None:
        return []

    seen: set[str] = set()
    titles: list[str] = []
    offset = None
    while True:
        points, offset = qdrant_client.scroll(
            collection_name=collection_name,
            with_payload=True,
            limit=100,
            offset=offset,
        )
        for point in points:
            metadata = (point.payload or {}).get("metadata") or {}
            title = metadata.get("title")
            if title and title not in seen:
                seen.add(title)
                titles.append(title)
        if offset is None:
            break
    return titles


def search(query: str, session_id: str, k: int = 4) -> list[Document]:
    """Search only the current session; return [] if nothing was ingested."""
    if not query.strip():
        return []
    if _existing_collection_name(session_id) is None:
        return []
    return get_vectorstore(session_id).similarity_search(query, k=k)
