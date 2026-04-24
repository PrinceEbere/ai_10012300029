import numpy as np


class Retriever:
    def __init__(self, index, chunks):
        self.index = index
        self.chunks = chunks

    # ----------------------------
    # NORMALIZE VECTOR (IMPORTANT FOR COSINE SIMILARITY)
    # ----------------------------
    def _normalize(self, vec):
        norm = np.linalg.norm(vec)
        return vec if norm == 0 else vec / norm

    # ----------------------------
    # SEARCH FUNCTION (IMPROVED)
    # ----------------------------
    def search(self, query_embedding, k=5, threshold=0.3):
        # Normalize query for cosine similarity
        query_embedding = self._normalize(query_embedding)

        query_embedding = np.array([query_embedding]).astype('float32')

        distances, indices = self.index.search(query_embedding, k)

        distances = distances[0]
        indices = indices[0]

        results = []
        scores = []

        for idx, score in zip(indices, distances):
            if 0 <= idx < len(self.chunks):
                # Filter weak matches
                if score >= threshold:
                    results.append(self.chunks[idx])
                    scores.append(float(score))

        return results, scores

    # ----------------------------
    # RERANK (IMPROVED VERSION)
    # ----------------------------
    def rerank(self, results, scores):
        # Combine relevance + length heuristic (better than plain sorting)
        ranked = sorted(
            zip(results, scores),
            key=lambda x: (x[1], len(x[0])),  # score + chunk richness
            reverse=True
        )

        reranked_results = [r for r, _ in ranked]
        reranked_scores = [s for _, s in ranked]

        return reranked_results, reranked_scores
