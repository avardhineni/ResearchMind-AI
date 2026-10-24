import json
import os
import re
import logging
import tempfile
import urllib.parse
import urllib.request
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from pathlib import Path

from langchain_community.document_loaders import PyMuPDFLoader, TextLoader, WebBaseLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)

_ARXIV_ID_RE = re.compile(r"(\d{4}\.\d{4,5}(?:v\d+)?)")

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP, add_start_index=True
)
_md_splitter = RecursiveCharacterTextSplitter.from_language(
    "markdown", chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP, add_start_index=True
)


def _stamp_title(docs: list[Document], title: str) -> list[Document]:
    for doc in docs:
        doc.metadata["title"] = title
    return docs


def load_pdf(file_path: str) -> list[Document]:
    docs = PyMuPDFLoader(file_path).load()
    return _stamp_title(_splitter.split_documents(docs), Path(file_path).stem)


def load_text(file_path: str) -> list[Document]:
    docs = TextLoader(file_path, encoding="utf-8").load()
    return _stamp_title(_splitter.split_documents(docs), Path(file_path).stem)


def load_markdown(file_path: str) -> list[Document]:
    docs = TextLoader(file_path, encoding="utf-8").load()
    return _stamp_title(_md_splitter.split_documents(docs), Path(file_path).stem)


def load_webpage(url: str) -> list[Document]:
    docs = WebBaseLoader(url, requests_kwargs={"timeout": 30}).load()
    title = (docs[0].metadata.get("title") or url) if docs else url
    return _stamp_title(_splitter.split_documents(docs), title)


def _extract_arxiv_id(query: str) -> str | None:
    """Return bare ArXiv ID (no version suffix) if one appears in the query."""
    m = _ARXIV_ID_RE.search(query)
    if m:
        return re.sub(r"v\d+$", "", m.group(1))
    return None


def _arxiv_api_lookup(arxiv_id: str) -> str:
    """Fetch paper title by ID from the ArXiv Atom API."""
    url = f"https://export.arxiv.org/api/query?id_list={arxiv_id}"
    with urllib.request.urlopen(url, timeout=10) as resp:
        xml = resp.read().decode()
    titles = re.findall(r"<title>(.*?)</title>", xml, re.DOTALL)
    return titles[1].strip() if len(titles) > 1 else arxiv_id


def _arxiv_search(query: str) -> str:
    """Search ArXiv Atom API by title phrase and return the top result's bare paper ID."""
    phrase = query.strip('"')
    search_query = urllib.parse.quote(f'ti:"{phrase}"')
    url = f"https://export.arxiv.org/api/query?search_query={search_query}&max_results=1&sortBy=relevance"
    with urllib.request.urlopen(url, timeout=15) as resp:
        xml = resp.read().decode()
    m = re.search(r"<id>https?://arxiv\.org/abs/(\d{4}\.\d{4,5}(?:v\d+)?)</id>", xml)
    if not m:
        raise ValueError(f"No ArXiv paper found for: {query}")
    return re.sub(r"v\d+$", "", m.group(1))


def _load_arxiv_by_id(arxiv_id: str) -> list[Document]:
    """Download and chunk an ArXiv paper PDF by its bare ID."""
    pdf_url = f"https://arxiv.org/pdf/{arxiv_id}"
    with urllib.request.urlopen(pdf_url, timeout=60) as resp:
        pdf_bytes = resp.read()
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(pdf_bytes)
            tmp_path = tmp.name
        docs = PyMuPDFLoader(tmp_path).load()
        if not docs:
            raise ValueError(f"Could not load PDF for ArXiv ID: {arxiv_id}")
        title = (docs[0].metadata.get("title") or "").strip() or _arxiv_api_lookup(arxiv_id)
        return _stamp_title(_splitter.split_documents(docs), title)
    finally:
        if tmp_path:
            Path(tmp_path).unlink(missing_ok=True)


def load_arxiv(query: str) -> list[Document]:
    arxiv_id = _extract_arxiv_id(query) or _arxiv_search(query)
    return _load_arxiv_by_id(arxiv_id)


# ── Semantic Scholar full-text ingestion ────────────────────────────────────────

_S2_API = "https://api.semanticscholar.org/graph/v1/paper"
_S2_FIELDS = "paperId,title,abstract,year,authors,url,openAccessPdf,externalIds,citationCount,publicationDate"
_S2_ID_RE = re.compile(r"^[0-9a-fA-F]{40}$")
_MAX_PDF_BYTES = 40 * 1024 * 1024  # Limit remote downloads to 40 MiB.


def _s2_json(url: str) -> dict:
    """Fetch Semantic Scholar API JSON, optionally using an API key from .env."""
    headers = {"Accept": "application/json", "User-Agent": "ResearchMindAI/1.0"}
    api_key = os.getenv("SEMANTIC_SCHOLAR_API_KEY", "").strip()
    if api_key:
        headers["x-api-key"] = api_key
    request = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=25) as response:
            return json.load(response)
    except HTTPError as exc:
        if exc.code == 429:
            raise RuntimeError("Semantic Scholar rate limit reached. Please try again later.") from exc
        if exc.code == 404:
            raise ValueError("Paper not found in Semantic Scholar.") from exc
        raise RuntimeError(f"Semantic Scholar API returned HTTP {exc.code}.") from exc
    except (URLError, TimeoutError) as exc:
        raise RuntimeError("Could not connect to Semantic Scholar.") from exc


def search_semantic_scholar_papers(query: str, limit: int = 5) -> list[dict]:
    """Find paper metadata; this does not ingest abstracts as full-text papers."""
    if not query.strip():
        raise ValueError("Enter a paper title or research topic.")
    params = urllib.parse.urlencode({
        "query": query.strip(), "limit": max(1, min(int(limit), 10)), "fields": _S2_FIELDS
    })
    return _s2_json(f"{_S2_API}/search?{params}").get("data", [])


def get_semantic_scholar_paper(paper_id: str) -> dict:
    """Fetch a paper by Semantic Scholar ID, DOI:..., or ARXIV:... identifier."""
    identifier = paper_id.strip()
    if not identifier:
        raise ValueError("A Semantic Scholar paper identifier is required.")
    encoded = urllib.parse.quote(identifier, safe=":")
    return _s2_json(f"{_S2_API}/{encoded}?fields={urllib.parse.quote(_S2_FIELDS)}")


def _semantic_scholar_identifier(query: str) -> str | None:
    """Recognize Semantic Scholar paper links, DOI URLs and explicit paper IDs."""
    q = query.strip()
    if _S2_ID_RE.fullmatch(q) or q.upper().startswith(("DOI:", "ARXIV:")):
        return q
    parsed = urlparse(q)
    host = (parsed.hostname or "").lower()
    if host in ("www.semanticscholar.org", "semanticscholar.org"):
        parts = [p for p in parsed.path.split("/") if p]
        if parts and _S2_ID_RE.fullmatch(parts[-1]):
            return parts[-1]
    if host in ("doi.org", "www.doi.org") and parsed.path.strip("/"):
        return "DOI:" + urllib.parse.unquote(parsed.path.strip("/"))
    return None


def _download_open_access_pdf(pdf_url: str) -> bytes:
    """Download a public HTTPS PDF, enforcing a response size and PDF signature."""
    parsed = urlparse(pdf_url)
    if parsed.scheme != "https" or not parsed.hostname:
        raise ValueError("Semantic Scholar did not provide a valid HTTPS PDF URL.")
    request = urllib.request.Request(pdf_url, headers={"User-Agent": "ResearchMindAI/1.0"})
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            length = response.headers.get("Content-Length")
            if length and int(length) > _MAX_PDF_BYTES:
                raise ValueError("The open-access PDF is larger than the 40 MB download limit.")
            data = response.read(_MAX_PDF_BYTES + 1)
    except (HTTPError, URLError, TimeoutError) as exc:
        raise RuntimeError("The open-access PDF could not be downloaded from its publisher/repository.") from exc
    if len(data) > _MAX_PDF_BYTES:
        raise ValueError("The open-access PDF is larger than the 40 MB download limit.")
    if not data.lstrip().startswith(b"%PDF-"):
        raise ValueError("The supplied open-access link did not return a valid PDF.")
    return data


def load_semantic_scholar(query: str, paper_id: str | None = None) -> list[Document]:
    """Load a Semantic Scholar open-access paper PDF into chunked Documents.

    Pass a selected paper_id from search results for unambiguous ingestion.
    When only a title is provided, the first search result is selected; callers
    should show results and let the user choose for better accuracy.
    """
    identifier = paper_id or _semantic_scholar_identifier(query)
    if identifier:
        paper = get_semantic_scholar_paper(identifier)
    else:
        matches = search_semantic_scholar_papers(query, limit=5)
        if not matches:
            raise ValueError(f"No Semantic Scholar papers found for: {query}")
        paper = get_semantic_scholar_paper(matches[0]["paperId"])

    title = paper.get("title") or "Untitled research paper"
    pdf_url = (paper.get("openAccessPdf") or {}).get("url")
    if not pdf_url:
        raise ValueError(
            f"'{title}' has no open-access PDF listed in Semantic Scholar. "
            "Its abstract can be used for discovery, but it cannot be ingested as full text."
        )
    pdf_bytes = _download_open_access_pdf(pdf_url)
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp:
            temp.write(pdf_bytes)
            temp_path = temp.name
        docs = PyMuPDFLoader(temp_path).load()
        if not docs:
            raise ValueError(f"No readable pages were found in '{title}'.")
        metadata = {
            "title": title,
            "source": paper.get("url") or pdf_url,
            "source_type": "semantic_scholar",
            "paper_id": paper.get("paperId"),
            "pdf_url": pdf_url,
            "year": paper.get("year"),
            "authors": ", ".join(a.get("name", "") for a in paper.get("authors") or []),
            "doi": (paper.get("externalIds") or {}).get("DOI"),
            "citation_count": paper.get("citationCount"),
        }
        for doc in docs:
            doc.metadata.update({k: v for k, v in metadata.items() if v is not None})
        return _splitter.split_documents(docs)
    finally:
        if temp_path:
            Path(temp_path).unlink(missing_ok=True)


def load_document(source: str) -> list[Document]:
    """Dispatch to the appropriate loader based on URL prefix or file extension."""
    if _semantic_scholar_identifier(source):
        return load_semantic_scholar(source)
    if source.startswith(("http://", "https://")):
        return load_webpage(source)
    ext = Path(source).suffix.lower()
    if ext == ".pdf":
        return load_pdf(source)
    if ext == ".txt":
        return load_text(source)
    if ext in (".md", ".markdown"):
        return load_markdown(source)
    raise ValueError(f"Unsupported file type: {ext}")
