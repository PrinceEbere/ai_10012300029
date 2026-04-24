from sentence_transformers import SentenceTransformer
import numpy as np

# Lazy load model (VERY IMPORTANT)
_model = None

def get_model():
    global _model
    if _model is None:
        _model = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')
    return _model


def create_embeddings(chunks, batch_size=16):
    """
    Create embeddings in batches to reduce memory usage.
    """
    model = get_model()

    embeddings = []

    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        batch_embeddings = model.encode(batch, show_progress_bar=False)
        embeddings.append(batch_embeddings)

    # Combine all batches
    embeddings = np.vstack(embeddings)

    return embeddings
