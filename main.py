import time
import json
import asyncio
from psutil import cpu_count
from concurrent.futures import ProcessPoolExecutor
from chunking import chunking, knowledge_base_runner
from qdrant_sender import async_send
from question_processor import question_preparation
from question_synonimizer import result_question, lemmatize_ru, context, learning_model
from api import get_iam_token, send_to_yagpt



# async def monitor():
#     while True:
#         cpu_usage = psutil.cpu_percent(percpu=True)
#         avg = sum(cpu_usage) / len(cpu_usage)
#         print(f"\rCPU: {avg:.1f}%", end="", flush=True)
#         await asyncio.sleep(2)


async def async_process_file(file, dir):

    loop = asyncio.get_event_loop()

    # синхронная часть в отдельном потоке
    physical_cores = cpu_count(logical=False)
    max_workers = physical_cores
    # количество физических ядер
    with ProcessPoolExecutor(max_workers=max_workers) as pool:
        filechunks = await loop.run_in_executor(
            pool,
            chunking,
            dir,
            file
        )

    # Асинхронная отправка
    await async_send(filechunks, file, rewrite=True)
    print(f"Запись {file} завершена")


async def main(question=None, base_directory=None):
    # monitor_task = asyncio.create_task(monitor())
    if base_directory is not None:
        txt_files = knowledge_base_runner(base_directory)
        # Увеличиваем лимит одновременных задач, ограничиваем параллельные файлы (max_workers*sem)
        sem = asyncio.Semaphore(1)

        async def limited_process(file, dir):
            async with sem:
                await async_process_file(file, dir)

        tasks = [limited_process(file, dir)
                 for file, dir in txt_files.items()]
        await asyncio.gather(*tasks)

    # создание токенизированного контекста для синонимизации, если не существует
    # Надо делать первый раз, а потом вызывать при необходимости - временное решение с JSON, надо оптимизировать на БД (SQL и тд)
    if base_directory == None:
        with open("/context/context.json", 'r', encoding='utf-8') as f:
            full_context = json.load(f)
    else:
        full_context = context("knowledge_base")

    # переобучили модель на полном контексте
    if base_directory != None:
        learning_model(full_context)

    if question is not None:

        lemma_task = asyncio.to_thread(lemmatize_ru, question)
        iam_token_task = asyncio.to_thread(get_iam_token)

        question_lemma, iam_token = await asyncio.gather(
            lemma_task,
            iam_token_task
        )

        question_top = await asyncio.to_thread(result_question, " ".join(question_lemma))

        search_task = question_preparation(question_top)
        answer_task = asyncio.to_thread(
            send_to_yagpt,
            iam_token,
            await search_task,
            system_prompt=f"Найди в тексте точный ответ на вопрос '{question}' и напиши краткий ответ. Если ответа нет - верни None",
            temperature=0.0, #более строгий ответ в привязке к тексту
            max_tokens=2000,
            folder_id="b1glp0iqac5h7ihhmh7b"
        )

        answer = await answer_task
        print(answer)
      
    await asyncio.sleep(1)
    # monitor_task.cancel()


if __name__ == "__main__":
    start_time = time.time()
    # question = "Что такое криптоключ и из чего он состоит?"
    # question = "как войти в систему?"
    question = "как осуществляется выпуск ЦФА?"

    # asyncio.run(main(question))
    asyncio.run(main(question, base_directory="knowledge_base"))
    dif_time = time.time() - start_time
    print(f"Работа: {dif_time:.3f} секунд")


# Инструкция

# Зависимости с версиями в requirements.txt. Путон 3.10

# Для тестового запуска: запустить докер, выполнить:
# docker run -p 6333:6333 -p 6334:6334     -v "$(pwd)/qdrant_storage:/qdrant/storage:z"     qdrant/qdrant
# Запустить программу.

# При первом запуске обязательно base_directory="knowledge_base" - на 6333 Qdrant
# это адрес папки в директории программы, в которой лежит вся база знаний от Дмитрия - туда нужно ее сохранить
# Программа обходит все папки и создает коллекции с векторными представлениями в БД по названию файла: 1 файл = 1 коллекция.
# Это может занять 15+ минут.
# http://localhost:6333/dashboard#/collections/ - здесь можно увидеть все коллекции, созданные в Qdrant

# Далее на основании базы знаний создается JSON (это временное неоптимальное решение) в папке 
# /context/context.json - в нем содержатся токены слов по базе знаний:
# предложения токенизированы, выкинуты стоп-слова с помощью nltk.
# При первом запуске создается папка nltk и скачиваются stopwords

# На основе токенизированного контекста обучается небольшая модель синонимов - FastText
# Она сохраняется в папке fasttext в текущей директории
# На этом этапе работа с базой знаний закончена (33 строка).
# При изменении файла в базе знаний нужно просто направить директорию папки, где обновился файл или ссылку на этот файл: base_directory=...
# ____________________________________________________________


# Теперь можно отправлять только вопрос -> asyncio.run(main(question))

# Далее работа с вопросом: он токенизируется, лемматизируется, к каждому токену добавляется топ2 слова из синонимов FastText
# После этого доработаный вопрос отправляется на векторизацию и ищутся лучшие совпадения чанков - для GPT топ5 Large чанков по совпадениям Small.
# (поиск совпадений по малым чанкам, они привязаны к большому. Большой чанк для сохранения смысла текста -~2500 символов)

# Лучший топ совпадений идет в GPT c системным промптом.
# ____________________________________________________________
# Работа с YaGPT:
# Аккаунт hackaton-team1@yandex.ru пароль: MIPT2025dev, там активирован грант на 3000 рублей
# OAuth-токен - уже получен, актуальный для работы, вставлен по умолчанию в api.py
# Инструкция для подключения: https://yandex.cloud/ru/docs/iam/operations/iam-token/create#exchange-token

# folder_id уже актуальная, берется тут: https://console.yandex.cloud/folders/b1glp0iqac5h7ihhmh7b - b1glp0iqac5h7ihhmh7b
# В данном случае автосозданная дефолтная
# один запрос в яндекс lite стоит примерно 20 копеек

