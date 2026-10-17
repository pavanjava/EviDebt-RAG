from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams
from sentence_transformers import SentenceTransformer

EMBED_DIM = 768


class QdrantIndex:
    def __init__(self, qdrant_cfg: dict, embedding_model: str):
        self.client = QdrantClient(
            host=qdrant_cfg["host"],
            port=qdrant_cfg["port"],
            api_key=qdrant_cfg["api_key"],
            https=False,
        )
        self.collection = qdrant_cfg["collection"]
        self.model = SentenceTransformer(embedding_model)

    def build(self, corpus: dict, batch_size: int = 64):
        """corpus: title -> paragraph text. Skips rebuild if collection already exists."""
        if self.client.collection_exists(self.collection):
            return

        self.client.create_collection(
            collection_name=self.collection,
            vectors_config=VectorParams(size=EMBED_DIM, distance=Distance.COSINE),
        )

        titles = list(corpus.keys())
        texts = [f"passage: {corpus[t]}" for t in titles]

        for i in range(0, len(texts), batch_size):
            batch_titles = titles[i : i + batch_size]
            batch_texts = texts[i : i + batch_size]
            vectors = self.model.encode(batch_texts, normalize_embeddings=True)
            points = [
                PointStruct(
                    id=i + j,
                    vector=vectors[j].tolist(),
                    payload={"title": batch_titles[j], "text": corpus[batch_titles[j]]},
                )
                for j in range(len(batch_titles))
            ]
            self.client.upsert(collection_name=self.collection, points=points)

    def retrieve(self, query: str, k: int):
        vector = self.model.encode(f"query: {query}", normalize_embeddings=True)
        hits = self.client.query_points(
            collection_name=self.collection, query=vector.tolist(), limit=k
        ).points
        return [{"title": h.payload["title"], "text": h.payload["text"]} for h in hits]
