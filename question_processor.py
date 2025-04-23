import asyncio
from sentence_transformers import SentenceTransformer
from qdrant_client import AsyncQdrantClient  


model = SentenceTransformer("intfloat/multilingual-e5-large")

async def async_search(client, collection, question_embedding):
    return await client.search(
        collection_name=collection,
        query_vector=question_embedding,
        limit=10,
        score_threshold=0.25,
        with_payload=["parent_id"]
    )


async def question_preparation(question):
    client = AsyncQdrantClient(host="localhost", port=6333, prefer_grpc=True)
    collections = await client.get_collections()
    collection_names = [c.name for c in collections.collections]
    

    question_embedding = model.encode(question).tolist()

    # сохраняем имя коллекции вместе с результатами
    tasks = [(col, async_search(client, col, question_embedding)) for col in collection_names]
    results = await asyncio.gather(*[task[1] for task in tasks])

    top_chunks = []
    seen_ids = set()

    for (collection_name, _), col_results in zip(tasks, results):
        for hit in col_results:
            if hit.payload["parent_id"] not in seen_ids:
                seen_ids.add(hit.payload["parent_id"])
                top_chunks.append((hit.score, hit.payload["parent_id"], collection_name))

    top_chunks.sort(reverse=True, key=lambda x: x[0])
    top5 = top_chunks[:5]


    retrieve_tasks = [
        client.retrieve(
            collection_name=item[2],
            ids=[item[1]],
            with_payload=["text"]
        ) for item in top5
    ]
    texts = await asyncio.gather(*retrieve_tasks)
    await client.close()
    
    return "\n".join([t[0].payload["text"] for t in texts])