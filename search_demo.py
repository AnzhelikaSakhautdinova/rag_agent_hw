from qdrant_client import QdrantClient
from fastembed import TextEmbedding

COLLECTION_NAME = "rag_mcp_hw"
QDRANT_PATH = "qdrant_data"


def search(query: str, k: int = 3):
    client = QdrantClient(path=QDRANT_PATH)
    embedding_model = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")

    query_vector = list(embedding_model.embed([query]))[0].tolist()

    response = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        limit=k,
        with_payload=True,
    )

    results = response.points

    for rank, item in enumerate(results, start=1):
        payload = item.payload

        print(f"\n{rank}. score={item.score:.4f}")
        print(f"document_id={payload['document_id']}")
        print(f"chunk_id={payload['chunk_id']}")
        print(f"source={payload['source']}")
        print(f"text={payload['text'][:500]}")


if __name__ == "__main__":
    search("What universities are located in South Korea?", k=3)
