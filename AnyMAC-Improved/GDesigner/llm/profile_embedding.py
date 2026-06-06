import logging
from sentence_transformers import SentenceTransformer

# Suppress the noisy "embeddings.position_ids | UNEXPECTED" load report
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)

# Cache the model so it's loaded once for the entire process
_model: SentenceTransformer | None = None

# Embedding cache: sentence -> numpy array
_embedding_cache: dict = {}


def get_sentence_embedding(sentence):
    global _model
    if sentence in _embedding_cache:
        return _embedding_cache[sentence]
    if _model is None:
        _model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    emb = _model.encode(sentence)
    _embedding_cache[sentence] = emb
    return emb


def clear_embedding_cache():
    _embedding_cache.clear()
