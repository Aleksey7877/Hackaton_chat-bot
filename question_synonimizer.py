from pymorphy2 import MorphAnalyzer
import re
import os
import nltk
import json
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
from gensim.models import FastText
from nltk.data import find
from chunking import knowledge_base_runner


# Лемматизация текста
def lemmatize_ru(text):
    morph = MorphAnalyzer()
    words = re.findall(r'\b\w+\b', text.lower())
    lemmas = [morph.parse(word)[0].normal_form for word in words]
    return lemmas

# Предобработка текста


def preprocess(sentence):
    current_dir = os.path.dirname(__file__)
    nltk_dir = os.path.join(current_dir, "nltk")
    os.makedirs(nltk_dir, exist_ok=True)
    nltk.data.path.append(nltk_dir)

    # Инициализация stopwords один раз
    try:
        find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords', download_dir="nltk")
    stop_words = set(stopwords.words('russian'))

    words = lemmatize_ru(sentence)
    filtered_words = [word for word in words if word.isalpha()
                      and word not in stop_words]
    return filtered_words


# Обучение модели FastText
def learning_model(processed_sentences):
    os.makedirs("fasttext", exist_ok=True)
    model = FastText(
        vector_size=200,
        window=5,
        min_count=1,
        workers=4
    )
    model.build_vocab(processed_sentences)
    model.train(processed_sentences, total_examples=len(
        processed_sentences), epochs=10)
    model_path = os.path.join("fasttext", "fasttext.model")
    model.save(model_path)


def learning_synonims(file):
    with open(file, "r", encoding="utf-8") as file:
        text = file.read()
    sentences = sent_tokenize(text, language='russian')
    processed_sentences = [preprocess(sentence) for sentence in sentences]
    return processed_sentences

# Синонимизация вопроса
def synonimize_question(question, model):
    question = preprocess(question)
    synonymized_question = []
    for word in question:
        if word in model.wv:  # Проверяем, есть ли слово в модели
            synonyms = model.wv.most_similar(word, topn=2)  # Топ-2 синонима
            synonymized_question.append((word, synonyms))
        else:
            synonymized_question.append((word, []))  # Если слова нет в модели
    return synonymized_question


def result_question(question):
    questions = []
    model_path = os.path.join("fasttext", "fasttext.model")
    model = FastText.load(model_path)

    synonyms = synonimize_question(question, model)
    synonymized_question = []
    for word, word_synonyms in synonyms:
        synonymized_question.append(word)  # Добавляем исходное слово
        if word_synonyms:  # Если есть синонимы
            # Добавляем первые 2 синонима из word_synonyms, можно сократить до 1
            synonymized_question.extend([syn[0] for syn in word_synonyms[:2]])
    synonymized_question = list(set(synonymized_question))
    synonymized_question = " ".join(synonymized_question)
    questions.append(question + " " + synonymized_question)
    return synonymized_question


# question_lemma = " ".join(lemmatize_ru(
#     "какие бывают способы обеспечния имущества организации?"))
# print(result_question(question_lemma))


def context(base_directory):
    context_dir = os.path.join(os.getcwd(), "context")
    os.makedirs(context_dir, exist_ok=True)

    context_file = os.path.join(context_dir, "context.json")

    if base_directory == None and os.path.exists(context_file):
        with open("/context/context.json", 'r', encoding='utf-8') as f:
            full_context = json.load(f)
    else:
        context_files = knowledge_base_runner("knowledge_base")
        full_context = []
        for file, directory in context_files.items():
            print("Идет запись из файла: ", file)
            context = learning_synonims(directory)
            full_context += context
        with open(context_file, 'w', encoding='utf-8') as f:
            json.dump(full_context, f, ensure_ascii=False, indent=2)

    return full_context

# context("knowledge_base")
