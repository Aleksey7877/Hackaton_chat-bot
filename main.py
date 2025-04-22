import time, json
from chunking import chunking, knowledge_base_runner
from qdrant_sender import send
from question_processor import question_preparation
from question_synonimizer import result_question, lemmatize_ru, context, learning_model
from api import get_iam_token_via_oauth, send_to_yagpt



def main(question=None, base_directory=None):
    start_time = time.time()

    # перезапись в Qdrant базы знаний при обновлении по выбранной директории
    # перезапись нужна при обновлении
    if base_directory is not None:
        txt_files = knowledge_base_runner(base_directory)
        for file, dir in txt_files.items():
            chunk_time_start = time.time()
            filechunks = chunking(dir, file)
            send(filechunks, file, rewrite=True) # Перезапись включена
            print("Идет запись в Qdrant ", file, "время ", round((time.time()-chunk_time_start), 2))

    # создание токенизированного контекста для синонимизации, если не существует
    # Надо делать первый раз, а потом вызывать при необходимости
    if base_directory == None:
        with open("/context/context.json", 'r', encoding='utf-8') as f:
            full_context = json.load(f)
    else:
        full_context = context("knowledge_base")

    # переобучили модель на полном контексте
    if base_directory != None:
        learning_model(full_context)
    
    if question is not None:
        # лемматизировали вопрос и получили контекстные синонимы для вопроса
        question_lemma = " ".join(lemmatize_ru(question))
        question_top = result_question(question_lemma)

        # ищем лучшие совпадения в Qdrant
        prompt_text = question_preparation(question_top)

        system_prompt=f"Найди в тексте точный ответ на вопрос '{question}' и напиши краткий ответ. Если ответа нет - верни None"
        iam_token = get_iam_token_via_oauth()

        answer = send_to_yagpt(iam_token, prompt_text,
                    system_prompt=system_prompt, temperature=0.6, max_tokens=2000, folder_id="b1glp0iqac5h7ihhmh7b")
        
        print(answer)

    dif_time = time.time() - start_time
    print(f"Работа: {dif_time:.3f} секунд")


if __name__ == "__main__":
    
    question = "Что такое криптоключ и из чего он состоит?"

    # main(question, base_directory="knowledge_base")
    main(question)


# Инструкция

# Для тестового запуска: запустить докер, выполнить:
# docker run -p 6333:6333 -p 6334:6334     -v "$(pwd)/qdrant_storage:/qdrant/storage:z"     qdrant/qdrant
# Запустить программу.

# При первом запуске обязательно base_directory="knowledge_base" - на 6333 Qdrant
# это адрес папки в директории программы, в которой лежит вся база знаний от Дмитрия
# Программа обходит все папки и создает коллекции с векторными представлениями в БД по названию файла: 1 файл = 1 коллекция.
# Это может занять 20+ минут!!
# http://localhost:6333/dashboard#/collections/ - здесь можно увидеть все коллекции, созданные в Qdrant

# Далее на основании базы знаний создается JSON в папке /context/context.json - в нем содержатся токены слов по базе знаний:
# предложения токенизированы, выкинуты стоп-слова с помощью nltk. 
# При первом запуске создается папка nltk и скачиваются stopwords
 
# На основе токенизированного контекста обучается небольшая модель синонимов - FastText
# Она сохраняется в папке fasttext в текущей директории
# На этом этапе работа с базой знаний закончена (33 строка).
# При изменении файла в базе знаний нужно просто направить директорию папки, где обновился файл или ссылку на этот файл: base_directory=...
# ____________________________________________________________


# Теперь можно отправлять только вопрос.

# Далее работа с вопросом: он токенизируется, лемматизируется, к каждому токену добавляется топ2 слова из синонимов FastText
# После этого доработаный вопрос отправляется на векторизацию и ищутся лучшие совпадения чанков - для GPT топ5 Large чанков по совпадениям Small.
# (поиск совпадений по малым чанкам, они привязаны к большому. Большой чанк для сохранения смысла текста -~2500 символов)

# Лучший топ совпадений идет в GPT c системным промптом. 
# Временно включено сохранение топ чанков для GPT в top_chunks.txt в директории для наглядности, что было найдено.

# Работа с YaGPT:
# Аккаунт hackaton-team1@yandex.ru пароль: MIPT2025dev
# OAuth-токен - уже получен, актуальный для работы, вставлен по умолчанию в api.py
# Инструкция для подключения: https://yandex.cloud/ru/docs/iam/operations/iam-token/create#exchange-token

# folder_id уже актуальная, берется тут: https://console.yandex.cloud/folders/b1glp0iqac5h7ihhmh7b - b1glp0iqac5h7ihhmh7b
# В данном случае автосозданная дефолтная

# Зависимости с варсими в файле requirements.txt. Путон 3.10