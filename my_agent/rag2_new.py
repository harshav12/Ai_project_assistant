import hashlib
import json
import os
from functools import lru_cache
from html.parser import HTMLParser
from pathlib import Path

import chromadb
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent
KNOWLEDGE_BASE = BASE_DIR / "Knowledge_base"
CHROMA_PATH = BASE_DIR / "chroma_db"
ISSUES_FILE = KNOWLEDGE_BASE / "issues.json"

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
COLLECTION_NAME = "project_issues_rag2_v2"
GEMINI_MODEL_NAME = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")

client = chromadb.PersistentClient(path=str(CHROMA_PATH))
collection = client.get_or_create_collection(
    name=COLLECTION_NAME,
    metadata={"hnsw:space": "cosine"},
)


class _TextExtractor(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts = []

    def handle_data(self, data):
        self.parts.append(data)


def _plain_text(value):
    if value is None:
        return ""

    parser = _TextExtractor()
    parser.feed(str(value))
    return " ".join(" ".join(parser.parts).split())


def load_file():
    with open(ISSUES_FILE, encoding="utf-8") as file:
        issues = json.load(file)

    if isinstance(issues, dict):
        issues = [issues]
    if not isinstance(issues, list) or not all(
        isinstance(issue, dict) for issue in issues
    ):
        raise ValueError(f"Expected a list of issue objects in {ISSUES_FILE}")

    return issues


def create_document(doc):
    fields = (
        ("Issue ID", doc.get("_id")),
        ("Title", doc.get("Title")),
        ("Description", doc.get("Description")),
        ("Reproducible steps", doc.get("ReproduceableSteps")),
        ("Related task ID", doc.get("RelatedTaskID")),
        ("Status", doc.get("Status")),
        ("Priority", doc.get("Priority")),
        ("Severity", doc.get("Severity")),
        ("Created on", doc.get("CreatedOn")),
        ("Resolved on", doc.get("ResolvedOn")),
        ("Closed on", doc.get("ClosedOn")),
        ("Workspace ID", doc.get("WorkspaceGUID")),
        ("Organization ID", doc.get("OrgGUID")),
        ("Section ID", doc.get("SectionGUID")),
        ("Space ID", doc.get("SpaceGUID")),
    )
    return "\n".join(
        f"{label}: {text}"
        for label, value in fields
        if (text := _plain_text(value))
    )


def converting_docs(data):
    documents = []

    for index, issue in enumerate(data):
        text = create_document(issue)
        if not text:
            continue

        issue_id = str(issue.get("_id") or f"record_{index}")
        metadata = {
            "issue_id": issue_id,
            "WorkspaceGUID": str(issue.get("WorkspaceGUID") or ""),
            "OrgGUID": str(issue.get("OrgGUID") or ""),
            "SectionGUID": str(issue.get("SectionGUID") or ""),
            "SpaceGUID": str(issue.get("SpaceGUID") or ""),
            "content_hash": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        }
        documents.append({
            "id": f"issue_{issue_id}",
            "document": text,
            "metadata": metadata,
        })

    return documents


@lru_cache(maxsize=1)
def get_embedding_model():
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(EMBEDDING_MODEL_NAME)


def create_embeddings(documents):
    if not documents:
        return []

    embeddings = get_embedding_model().encode(
        [document["document"] for document in documents],
        batch_size=32,
        show_progress_bar=False,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    return embeddings.tolist()


def create_collection(documents, embeddings):
    if not documents:
        return

    collection.upsert(
        ids=[document["id"] for document in documents],
        documents=[document["document"] for document in documents],
        embeddings=embeddings,
        metadatas=[document["metadata"] for document in documents],
    )


def index_issues():
    documents = converting_docs(load_file())
    existing = collection.get(include=["metadatas"])
    existing_hashes = {
        record_id: (metadata or {}).get("content_hash")
        for record_id, metadata in zip(existing["ids"], existing["metadatas"])
    }

    current_ids = {document["id"] for document in documents}
    changed_documents = [
        document
        for document in documents
        if existing_hashes.get(document["id"])
        != document["metadata"]["content_hash"]
    ]

    if changed_documents:
        create_collection(changed_documents, create_embeddings(changed_documents))

    stale_ids = set(existing["ids"]) - current_ids
    if stale_ids:
        collection.delete(ids=sorted(stale_ids))


def ask_chromadb(query, n_results=5, where=None):
    if not query or not query.strip():
        raise ValueError("Query must not be empty.")
    if n_results <= 0:
        raise ValueError("n_results must be greater than zero.")

    index_issues()
    total_documents = collection.count()
    if total_documents == 0:
        return {
            "ids": [[]],
            "documents": [[]],
            "metadatas": [[]],
            "distances": [[]],
        }

    query_embedding = get_embedding_model().encode(
        [query],
        show_progress_bar=False,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )[0]
    query_options = {
        "query_embeddings": [query_embedding.tolist()],
        "n_results": min(n_results, total_documents),
        "include": ["documents", "metadatas", "distances"],
    }
    if where is not None:
        query_options["where"] = where

    return collection.query(**query_options)


def search_issues(query, n_results=5, where=None):
    if not query or not query.strip():
        return []

    results = ask_chromadb(query, n_results=n_results, where=where)
    if not results["ids"] or not results["ids"][0]:
        return []

    return [
        {
            "id": results["ids"][0][index],
            "text": results["documents"][0][index],
            "metadata": results["metadatas"][0][index],
            "distance": results["distances"][0][index],
        }
        for index in range(len(results["ids"][0]))
    ]


def build_context(query, n_results=5, where=None):
    matches = search_issues(query, n_results=n_results, where=where)
    return "\n\n---\n\n".join(
        f"Issue {match['metadata'].get('issue_id', match['id'])}:\n"
        f"{match['text']}"
        for match in matches
    )


@lru_cache(maxsize=1)
def get_gemini_client():
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY environment variable is not set.")

    from google import genai

    return genai.Client(api_key=api_key)


def generate_answer(query, n_results=5, where=None):
    if not query or not query.strip():
        return "Please provide a question."

    matches = search_issues(query, n_results=n_results, where=where)
    if not matches:
        return "I couldn't find relevant issue records for that question."

    context = "\n\n---\n\n".join(
        f"Issue {match['metadata'].get('issue_id', match['id'])}:\n"
        f"{match['text']}"
        for match in matches
    )
    prompt = f"""Answer the user's question using only the retrieved issue records below.
Do not invent facts or interpret numeric status, priority, or severity codes beyond
what the records explicitly say. If the records do not contain enough information,
say what is missing. Cite relevant issue IDs in the answer.

Retrieved issue records:
{context}

User question:
{query}
"""
    response = get_gemini_client().models.generate_content(
        model=GEMINI_MODEL_NAME,
        contents=prompt,
    )
    if not response.text:
        raise RuntimeError("Gemini returned an empty response.")

    return response.text
