import os
from functools import lru_cache

from chromadb import Documents, EmbeddingFunction, Embeddings


@lru_cache(maxsize=1)
def _load_model():
    from sentence_transformers import SentenceTransformer

    model_path = os.environ.get("EMBEDDING_MODEL_PATH") or os.environ.get(
        "EMBEDDING_MODEL_NAME", "BAAI/bge-small-zh-v1.5"
    )
    return SentenceTransformer(model_path)


class BGEEmbeddingFunction(EmbeddingFunction):
    # Chroma validates the parameter is named exactly `input`
    def __call__(self, input: Documents) -> Embeddings:
        model = _load_model()
        vecs = model.encode(
            list(input),
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return vecs.tolist()


@lru_cache(maxsize=1)
def get_embedding_function() -> BGEEmbeddingFunction:
    return BGEEmbeddingFunction()
