from pathlib import Path
from uuid import uuid4

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from fastembed import TextEmbedding


COLLECTION_NAME = "rag_mcp_hw"
DOCS_DIR = Path("docs")
QDRANT_PATH = "qdrant_data"

CHUNK_SIZE = 500
CHUNK_OVERLAP = 80


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP):
    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        start += chunk_size - overlap

    return chunks


def main():
    client = QdrantClient(path=QDRANT_PATH)

    embedding_model = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")
    vector_size = 384

    if client.collection_exists(COLLECTION_NAME):
        client.delete_collection(COLLECTION_NAME)

    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(
            size=vector_size,
            distance=Distance.COSINE,
        ),
    )

    points = []

    for doc_idx, path in enumerate(sorted(DOCS_DIR.glob("*.md")), start=1):
        text = path.read_text(encoding="utf-8")
        document_id = path.stem
        chunks = chunk_text(text)

        embeddings = list(embedding_model.embed(chunks))

        for chunk_idx, (chunk, vector) in enumerate(zip(chunks, embeddings), start=1):
            chunk_id = f"{document_id}_chunk_{chunk_idx:03d}"

            payload = {
                "text": chunk,
                "document_id": document_id,
                "chunk_id": chunk_id,
                "source": str(path),
                "title": path.stem,
            }

            points.append(
                PointStruct(
                    id=str(uuid4()),
                    vector=vector.tolist(),
                    payload=payload,
                )
            )

    client.upsert(
        collection_name=COLLECTION_NAME,
        points=points,
    )

    print(f"Indexed {len(points)} chunks into collection '{COLLECTION_NAME}'")
    print(f"Qdrant local path: {QDRANT_PATH}")


if __name__ == "__main__":
    main()