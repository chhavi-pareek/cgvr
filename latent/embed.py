import hashlib
import pathlib

import numpy as np

CACHE = pathlib.Path(__file__).parent / "cache"
MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def embed(texts, model_name=MODEL):
    key = hashlib.sha1((model_name + "\n" + "\n".join(texts)).encode()).hexdigest()[:16]
    path = CACHE / f"emb_{key}.npy"
    if path.exists():
        return np.load(path)
    import torch
    from sentence_transformers import SentenceTransformer

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    e = SentenceTransformer(model_name, device=dev).encode(
        texts, batch_size=128, normalize_embeddings=True, convert_to_numpy=True
    ).astype(np.float32)
    CACHE.mkdir(exist_ok=True)
    np.save(path, e)
    return e


if __name__ == "__main__":
    from latent.corpus import load

    e = embed(load()[0])
    print(e.shape, e.dtype)
