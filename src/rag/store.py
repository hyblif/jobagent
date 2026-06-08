import os
from functools import lru_cache

import chromadb

from src.rag.embed import get_embedding_function

COLLECTION_NAME = "baguwen"


def get_persist_dir() -> str:
    return os.environ.get("CHROMA_PERSIST_DIR", ".chroma/jobagent")


@lru_cache(maxsize=1)
def get_client() -> "chromadb.api.ClientAPI":
    return chromadb.PersistentClient(path=get_persist_dir())


def get_collection(reset: bool = False):
    client = get_client()
    if reset:
        try:
            client.delete_collection(COLLECTION_NAME)
        except Exception:
            pass
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=get_embedding_function(),
        metadata={"hnsw:space": "cosine"},
    )
