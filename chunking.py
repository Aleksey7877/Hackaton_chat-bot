import os
from functools import lru_cache
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from qdrant_client.models import PointStruct

@lru_cache(maxsize=None)
def get_model():
    return SentenceTransformer("intfloat/multilingual-e5-large")

def chunking(input_file, name):
    small_chunk_size = 300
    small_chunk_overlap = 80
    large_chunk_size = 2500
    large_chunk_overlap = 400

    # Задаем параметры крупного чанка для сохранения основного смысла раздела
    large_splitter = RecursiveCharacterTextSplitter(
        chunk_size=large_chunk_size,
        chunk_overlap=large_chunk_overlap,
        separators=['\n\nГлава ', '\n\nРаздел ', '\n\nСтатья ', '\nПункт ', '\n\n', '\n'])

    # Задаем параметры малого чанка, по которому будем искать лучшие совпадения
    small_splitter = RecursiveCharacterTextSplitter(
        chunk_size=small_chunk_size,
        chunk_overlap=small_chunk_overlap,
        separators=['\n\nГлава ', '\n\nРаздел ', '\n\nСтатья ', '\nПункт ', '\n\n', '\n'])

    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            text = f.read()
    except FileNotFoundError:
        print(f"Ошибка: файл {input_file} не найден!")
        return

    # Разбиваем текст на большие чанки
    large_chunks = large_splitter.split_text(text)

    points_small = []
    points_large = []
    model = get_model()
    i = 0
    for large_id, large_chunk in enumerate(large_chunks):
        if len(large_chunk) < small_chunk_size * 2:
            small_chunks = [large_chunk]
        else:
            small_chunks = small_splitter.split_text(large_chunk)
        large_embedding = model.encode(large_chunk).tolist()

        points_large.append(PointStruct(
            id=i,
            vector=large_embedding,
            payload={
                "document_name": name,
                "text": large_chunk,
                "parent_id": i    
            }
        ))
        large_id = i
        i += 1

        for small_chunk in small_chunks:
            small_embedding = model.encode(small_chunk).tolist()
            points_small.append(PointStruct(
                id=i,
                vector=small_embedding,
                payload={
                    "document_name": name,
                    "text": None,
                    "parent_id": large_id
                }
            ))
            i += 1

    return {"Large": points_large, "Small": points_small}


def knowledge_base_runner(directory):
    txt_files = {}
    base_dir = os.path.join(os.path.dirname(__file__), directory)
    if base_dir.endswith(".txt"):
        return {directory[:-4]: base_dir}
    else:
        for root, _, files in os.walk(base_dir):
            for file in files:
                if file.endswith(".txt"):
                    txt_files.update({file[:-4]: (os.path.join(root, file))})
    return txt_files


