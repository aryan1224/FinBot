from sentence_transformers import SentenceTransformer

_model: SentenceTransformer | None = None


def get_embedding_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer("all-MiniLM-L6-v2")
    return _model


def preload_embedding_model() -> None:
    """Force the model to load now. Safe to call more than once (it's a
    no-op after the first call, same singleton as get_embedding_model)."""
    get_embedding_model()


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed a list of strings, returning a list of plain float vectors
    (Chroma wants plain lists, not numpy arrays)."""
    model = get_embedding_model()
    vectors = model.encode(texts, convert_to_numpy=True)
    return vectors.tolist()