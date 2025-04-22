from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct  # Добавить PointStruct в импорт
from itertools import islice



def send(filechunks,  filename, rewrite=False):
    # client = QdrantClient(url="http://localhost:6333", timeout=60.0)
    client = QdrantClient(
        url="http://localhost:6333", 
        timeout=120.0,  # Увеличили таймаут
        prefer_grpc=True  # Используем gRPC для больших данных
    )
    size=1024
    collection_name=filename

    if not rewrite and client.collection_exists(collection_name):
        print(f"Перезапись {collection_name} отключена.")
        return
    elif rewrite and client.collection_exists(collection_name):
        client.delete_collection(collection_name)

    #  создаем коллекцию 
    client.create_collection(  
        collection_name=collection_name,
        vectors_config=VectorParams(size=size, distance=Distance.DOT),
    )

    def batched(iterable, batch_size):
        iterator = iter(iterable)
        while batch := list(islice(iterator, batch_size)):
            yield batch

    # Для gRPC преобразуем батчи в список PointStruct
    for batch in batched(filechunks["Large"], 50):
        client.upsert(
            collection_name=collection_name,
            points=[PointStruct(**p) if isinstance(p, dict) else p for p in batch],  # Конвертируем при необходимости
            wait=True
        )

    for batch in batched(filechunks["Small"], 100):
        client.upsert(
            collection_name=collection_name,
            points=[PointStruct(**p) if isinstance(p, dict) else p for p in batch],
            wait=True
        )

    # client.upsert(  # создали вектора
    #     collection_name=collection_name,
    #     wait=True,
    #     points=filechunks["Large"]
    #     )
    
    # client.upsert(  
    #     collection_name=collection_name,
    #     wait=True,
    #     points=filechunks["Small"]
    #     )