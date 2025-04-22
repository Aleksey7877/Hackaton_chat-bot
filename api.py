import requests


def get_iam_token_via_oauth(oauth_token="y0__xD8_K6MCBjB3RMg9sPt8RIwyYivjAhazOSX7kyjRB8mzebu7l9FJAVbDQ"):
    # Обменивает OAuth-токен на IAM-токен
    url = "https://iam.api.cloud.yandex.net/iam/v1/tokens"
    headers = {"Content-Type": "application/json"}
    data = {"yandexPassportOauthToken": oauth_token}

    try:
        response = requests.post(url, headers=headers, json=data)
        response.raise_for_status()
        return response.json().get("iamToken")
    except requests.exceptions.RequestException as e:
        print(f"Ошибка при получении токена: {e}")
        return None


# Ваш OAuth-токен - уже получен актуальный для работы
# Инструкция для подключения: https://yandex.cloud/ru/docs/iam/operations/iam-token/create#exchange-token

oauth_token = "y0__xD8_K6MCBjB3RMg9sPt8RIwyYivjAhazOSX7kyjRB8mzebu7l9FJAVbDQ"
iam_token = get_iam_token_via_oauth(oauth_token)

# folder_id уже актуальная, берется тут: https://console.yandex.cloud/folders/b1glp0iqac5h7ihhmh7b - b1glp0iqac5h7ihhmh7b
# В данном случае автосозданная дефолтная


def send_to_yagpt(iam_token, prompt_text, system_prompt=None, temperature=0.6, max_tokens=20000, folder_id="b1glp0iqac5h7ihhmh7b"):
    """
    Отправляет запрос к Yandex GPT API

    :param folder_id: Идентификатор каталога Yandex Cloud
    :param iam_token: IAM-токен для аутентификации
    :param prompt_text: Текст запроса пользователя
    :param system_prompt: Системный промпт (опционально)
    :param temperature: Креативность ответа (0-1)
    :param max_tokens: Максимальное количество токенов в ответе
    :return: Ответ от API в виде строки
    """
    url = "https://llm.api.cloud.yandex.net/foundationModels/v1/completion"

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {iam_token}"
    }

    messages = []

    if system_prompt:
        messages.append({
            "role": "system",
            "text": system_prompt
        })

    messages.append({
        "role": "user",
        "text": prompt_text
    })

    data = {
        "modelUri": f"gpt://{folder_id}/yandexgpt-lite/latest",
        "completionOptions": {
            "stream": False,
            "temperature": temperature,
            "maxTokens": str(max_tokens),
            "reasoningOptions": {
                "mode": "DISABLED"
            }
        },
        "messages": messages
    }

    try:
        response = requests.post(url, headers=headers, json=data)
        response.raise_for_status()
        result = response.json()

        return result['result']['alternatives'][0]['message']['text']
    except requests.exceptions.RequestException as e:
        print(f"Ошибка при запросе к API: {e}")
        return None
    except (KeyError, IndexError) as e:
        print(f"Ошибка при обработке ответа: {e}")
        return None


# можно протестировать
# print(send_to_yagpt(iam_token, prompt_text="сколько будет 10!"))
