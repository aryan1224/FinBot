import os
import chromadb
from modules.embedding import embed_texts

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHROMA_PATH = os.path.join(_PROJECT_ROOT, "data", "chroma")
COLLECTION_NAME = "bank_policies"

def _get_collection():
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    try:
        return client.get_collection(COLLECTION_NAME)
    except Exception:
        raise RuntimeError("no db exists")


def retrieve_top_k(query: str, k: int = 5) -> list[str]:
    """Return up to k most relevant chunk texts for `query`. Raises
    RuntimeError("no db exists") if the collection hasn't been created
    yet (i.e. no policy PDF has ever been ingested)."""
    if not query or not query.strip():
        return []

    collection = _get_collection()
    query_embedding = embed_texts([query])[0]

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=k,
    )
    documents = results.get("documents") or [[]]
    return documents[0]
