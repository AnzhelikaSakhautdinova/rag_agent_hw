import pandas as pd

from qdrant_client import QdrantClient
from fastembed import TextEmbedding

COLLECTION_NAME = "rag_mcp_hw"

client = QdrantClient(path="qdrant_data")
embedder = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")

df = pd.read_csv("eval.csv")

results = []

for _, row in df.iterrows():

    query = row["query"]

    query_vector = list(
        embedder.embed([query])
    )[0].tolist()

    response = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        limit=3,
        with_payload=True,
    )

    retrieved = [
        p.payload["chunk_id"]
        for p in response.points
    ]

    top_result = response.points[0]

    first_answer = top_result.payload["text"][:300]

    score = top_result.score

    expected = row["expected_source"]

    hit = expected in retrieved

    results.append({
        "query": query,
        "expected": expected,
        "retrieved": retrieved[0],
        "score": score,
        "first_answer": first_answer,
        "hit_top3": hit
    })


report = pd.DataFrame(results)

print(report)

accuracy = report["hit_top3"].mean()

print("\nTop-3 Accuracy:", accuracy)

report.to_excel("evaluation_report.xlsx", index=False)