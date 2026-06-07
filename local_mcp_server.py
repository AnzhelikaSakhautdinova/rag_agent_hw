from fastmcp import FastMCP
from qdrant_client import QdrantClient
from fastembed import TextEmbedding

COLLECTION_NAME = "rag_mcp_hw"
QDRANT_PATH = "qdrant_data"

mcp = FastMCP("local-qdrant-rag")

client = QdrantClient(path=QDRANT_PATH)
embedding_model = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")


@mcp.tool()
def qdrant_find(query: str, k: int = 3) -> str:
    """Search top-k relevant chunks in local Qdrant knowledge base."""

    query_vector = list(embedding_model.embed([query]))[0].tolist()

    response = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        limit=k,
        with_payload=True,
    )

    results = []

    for rank, item in enumerate(response.points, start=1):
        payload = item.payload

        results.append(
            f"""
Rank: {rank}
Score: {item.score:.4f}
document_id: {payload.get("document_id")}
chunk_id: {payload.get("chunk_id")}
source: {payload.get("source")}
text: {payload.get("text")}
"""
        )

    return "\n---\n".join(results)


if __name__ == "__main__":
    mcp.run(transport="stdio")