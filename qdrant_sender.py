import asyncio
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from itertools import islice


def batched(iterable, batch_size):
    iterator = iter(iterable)
    while batch := list(islice(iterator, batch_size)):
        yield batch

async def async_send(filechunks, filename, rewrite=False):
    client = AsyncQdrantClient(
        host="localhost",
        port=6333,
        prefer_grpc=True,
        timeout=120 
    )

    size = 1024
    collection_name = filename
    
    try:
        existing_collections = await client.get_collections()
        exists = any(col.name == collection_name for col in existing_collections.collections)
        
        if exists:
            if rewrite:
                await client.delete_collection(collection_name)
            else:
                print(f"Collection {collection_name} exists. passing")
                return
                
        await client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=size, distance=Distance.DOT)
        )
        
        # Функция для вставки батча
        async def upsert_batch(points):
            converted = [
                PointStruct(**p) if isinstance(p, dict) else p
                for p in points
            ]
            await client.upsert(
                collection_name=collection_name,
                points=converted,
                wait=False
            )
        
        tasks = []
        for batch in batched(filechunks["Large"], 200):
            tasks.append(upsert_batch(batch))
            
        for batch in batched(filechunks["Small"], 500):
            tasks.append(upsert_batch(batch))
            
        await asyncio.gather(*tasks)
        
    except Exception as e:
        print(f"Error processing {collection_name}: {str(e)}")
    finally:
        await client.close()

