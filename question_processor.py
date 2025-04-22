from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient



def question_preparation(question):
    client = QdrantClient(
        url="http://localhost:6333", 
        timeout=120.0,  # Увеличили таймаут
        prefer_grpc=True  # Используем gRPC для больших данных
    )

    collections  = client.get_collections().collections

    collection_names = [collection.name for collection in collections]
  
    top_of_results = 10
    all_results = {}
    for collection in collection_names:
        # texted_chunks = set()
        model_name="intfloat/multilingual-e5-large"
        score_threshold=0.25

        model = SentenceTransformer(model_name)
        question_embedding = model.encode(question).tolist()
        
        search_result = client.search(
            collection_name=collection,
            query_vector=question_embedding,
            limit=top_of_results,  
            with_payload=True,
            with_vectors=False,
            score_threshold=score_threshold  
        )
        chunk_list = {}
        for chunk in search_result:
            chunk_list.update({chunk.score: chunk.payload['parent_id']})
        all_results.update({collection: chunk_list})
 
    flattened = []
    for collection_name, scores in all_results.items():
        for score, parent_id in scores.items():
            flattened.append({
                'collection': collection_name,
                'score': score,
                'parent_id': parent_id
            })

    # Сортируем по убыванию score и берем топ-5
    top5 = sorted(flattened, key=lambda x: x['score'], reverse=True)[:5]
    name = f"top_chunks.txt"
    with open(name, 'w', encoding='utf-8') as f:
        seen_ids = set()
        for item in top5:
            if item['parent_id'] not in seen_ids:
                record = client.retrieve(
                    collection_name=item['collection'],
                    ids=[item['parent_id']],
                    with_payload=True,
                    with_vectors=False
                )
                seen_ids.add(item['parent_id'])
                chunk_text = record[0].payload.get('text')
                f.write(f"{item['parent_id']}\n\n{chunk_text}\n")
            else:
                continue
    with open(f"top_chunks.txt", 'r', encoding='utf-8') as f:
            text = f.read()
    return text


