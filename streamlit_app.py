import json
import asyncio
from typing import Dict, Any, List, Coroutine
import streamlit as st
from openai import AsyncOpenAI
from anthropic import Anthropic
from statistics import mean
import logging
import plotly.express as px
import pandas as pd
import os

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


# Функция для получения API ключей
def get_api_keys() -> (str, str):
    st.sidebar.header("API Ключи")

    openai_key = st.sidebar.text_input("Введите ваш API ключ OpenAI:", type="password", key="openai_input")
    anthropic_key = st.sidebar.text_input("Введите ваш API ключ Anthropic:", type="password", key="anthropic_input")

    if st.sidebar.button("Сохранить ключи"):
        st.session_state["openai_key"] = openai_key
        st.session_state["anthropic_key"] = anthropic_key
        st.sidebar.success("Ключи сохранены!")

    return st.session_state.get("openai_key", openai_key), st.session_state.get("anthropic_key", anthropic_key)


# Настройка клиентов
@st.cache_resource
def get_openai_client(api_key: str) -> AsyncOpenAI:
    if api_key:
        return AsyncOpenAI(api_key=api_key)
    else:
        st.error("API ключ OpenAI не предоставлен.")
        return None


@st.cache_resource
def get_anthropic_client(api_key: str) -> Anthropic:
    if api_key:
        return Anthropic(api_key=api_key)
    else:
        st.error("API ключ Anthropic не предоставлен.")
        return None


# Асинхронные функции для работы с API
async def create_mnemonic(word: str, prompt: str, model: str, openai_client: AsyncOpenAI,
                          anthropic_client: Anthropic) -> Dict[str, Any]:
    try:
        if "gpt" in model.lower() and openai_client:
            response = await openai_client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": word}
                ]
            )
            content = response.choices[0].message.content
        elif "claude" in model.lower() and anthropic_client:
            if model.startswith("claude-3"):
                message = anthropic_client.messages.create(
                    model=model,
                    max_tokens=1000,
                    messages=[
                        {"role": "user", "content": f"{prompt}\n\nWord: {word}"}
                    ]
                )
                content = message.content[0].text
            else:
                response = anthropic_client.completions.create(
                    model=model,
                    prompt=f"{prompt}\n\nHuman: {word}\n\nAssistant:",
                    max_tokens_to_sample=300
                )
                content = response.completion
        else:
            raise ValueError(f"Неподдерживаемая модель или отсутствует клиент для модели: {model}")

        try:
            return json.loads(content)
        except json.JSONDecodeError:
            return {"association": content, "meaning": "", "prompt": ""}
    except Exception as e:
        logging.error(f"Ошибка при создании мнемоники для '{word}' с моделью {model}: {str(e)}")
        return {"association": f"Ошибка при создании мнемоники с моделью {model}: {str(e)}", "meaning": "",
                "prompt": ""}


async def rate_memory(association: str, model: str, rating_prompt: str, openai_client: AsyncOpenAI,
                      anthropic_client: Anthropic) -> int:
    try:
        if "gpt" in model.lower() and openai_client:
            response = await openai_client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": rating_prompt},
                    {"role": "user", "content": association}
                ]
            )
            content = response.choices[0].message.content
        elif "claude" in model.lower() and anthropic_client:
            if model.startswith("claude-3"):
                message = anthropic_client.messages.create(
                    model=model,
                    max_tokens=10,
                    messages=[
                        {"role": "user", "content": f"{rating_prompt}\n\nAssociation: {association}"}
                    ]
                )
                content = message.content[0].text
            else:
                response = anthropic_client.completions.create(
                    model=model,
                    prompt=f"{rating_prompt}\n\nHuman: {association}\n\nAssistant:",
                    max_tokens_to_sample=10
                )
                content = response.completion
        else:
            raise ValueError(f"Неподдерживаемая модель или отсутствует клиент для модели: {model}")

        return int(content)
    except ValueError as ve:
        logging.warning(f"Ошибка при оценке запоминаемости: {str(ve)}")
        return 0
    except Exception as e:
        logging.error(f"Ошибка при оценке запоминаемости с моделью {model}: {str(e)}")
        return 0


async def process_word(word: str, prompts: List[str], models: List[str], rating_model: str, openai_client: AsyncOpenAI,
                       anthropic_client: Anthropic, rating_prompt: str) -> Dict[str, Any]:
    word_results = []
    for prompt in prompts:
        for model in models:
            mnemonic = await create_mnemonic(word, prompt, model, openai_client, anthropic_client)
            association = mnemonic.get('association', '')
            score = await rate_memory(association, rating_model, rating_prompt, openai_client, anthropic_client)
            mnemonic['score'] = score
            mnemonic['model'] = model
            word_results.append(mnemonic)
    return {"word": word, "mnemonics": word_results}


def run_async(coroutine: Coroutine[Any, Any, Any]) -> Any:
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(coroutine)
    finally:
        loop.close()


@st.cache_data
def cached_process_words(words: List[str], prompts: List[str], models: List[str], rating_model: str,
                         _openai_client: AsyncOpenAI, _anthropic_client: Anthropic, rating_prompt: str) -> List[
    Dict[str, Any]]:
    return [
        run_async(process_word(word, prompts, models, rating_model, _openai_client, _anthropic_client, rating_prompt))
        for word in words]


# Функция для сохранения и загрузки истории запросов
def save_history(results: List[Dict[str, Any]], filename: str = "history.json"):
    if os.path.exists(filename):
        with open(filename, 'r', encoding='utf-8') as f:
            history = json.load(f)
    else:
        history = []

    history.extend(results)
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(history, f, ensure_ascii=False, indent=2)


def load_history(filename: str = "history.json") -> List[Dict[str, Any]]:
    if os.path.exists(filename):
        with open(filename, 'r', encoding='utf-8') as f:
            return json.load(f)
    return []


# Функция для визуализации результатов
def visualize_results(results: List[Dict[str, Any]]):
    data = []
    for result in results:
        for mnemonic in result['mnemonics']:
            data.append({
                "Word": result['word'],
                "Model": mnemonic['model'],
                "Prompt": mnemonic.get('prompt', 'Не указано'),
                "Score": mnemonic.get('score', 0)
            })

    df = pd.DataFrame(data)
    fig = px.box(df, x="Model", y="Score", color="Model", title="Оценки запоминаемости по моделям")
    st.plotly_chart(fig)


# Основной интерфейс приложения
def main():
    st.title('Мнемоническая ассоциация и оценка запоминаемости')

    # Ввод слов пользователем
    words_input = st.text_area("Введите слова (по одному на строку):", height=150)
    words = [word.strip() for word in words_input.split('\n') if word.strip()]

    # Предлагаемые модели
    suggested_models = [
        "claude-3-5-sonnet-20240620",
        "claude-3-opus-20240229",
        "claude-3-sonnet-20240229",
        "claude-3-haiku-20240307",
        "gpt-4o",
        "chatgpt-4o-latest",
        "gpt-4o-mini",
        "gpt-4",
        "gpt-4-turbo"
    ]

    # Ввод моделей пользователем
    st.subheader("Выберите модели для генерации мнемоник:")
    selected_models = st.multiselect(
        "Выберите модели из списка или введите свои:",
        options=suggested_models,
        default=[],
        key="model_selection"
    )

    # Дополнительное поле для ввода пользовательских моделей
    custom_models = st.text_area(
        "Введите дополнительные модели (по одной на строку):",
        height=100,
        key="custom_models"
    )

    # Объединение выбранных и пользовательских моделей
    selected_models += [model.strip() for model in custom_models.split('\n') if model.strip()]

    if not selected_models:
        st.warning("Пожалуйста, выберите хотя бы одну модель для продолжения.")
        st.stop()

    # Выбор модели для оценки
    st.subheader("Выберите модель для оценки:")
    rating_model = st.selectbox(
        "Выберите модель для оценки из списка или введите свою:",
        options=suggested_models + selected_models,
        index=0
    )

    # Редактирование стандартного промпта оценки
    default_rating_prompt = "You are a memory expert. Rate the following mnemonic association on a scale from 1 to 100 based on how easy it is to remember. Return only the numeric score."
    rating_prompt = st.text_area("Отредактируйте промпт для оценки запоминаемости:", value=default_rating_prompt,
                                 height=100)

    # Ввод промптов пользователем
    st.subheader("Введите промпты:")
    num_prompts = st.number_input("Количество промптов", min_value=1, value=1, step=1)
    prompts = []
    for i in range(num_prompts):
        prompt = st.text_area(f"Промпт {i + 1}", height=100, key=f"prompt_{i}")
        prompts.append(prompt)

    if st.button('Генерировать мнемоники и оценки'):
        if not words:
            st.error("Пожалуйста, введите хотя бы одно слово.")
        elif not all(prompts):
            st.error("Пожалуйста, заполните все промпты.")
        elif not (openai_api_key or anthropic_api_key):
            st.error("Пожалуйста, введите хотя бы один API ключ.")
        else:
            with st.spinner('Обработка...'):
                results = cached_process_words(words, prompts, selected_models, rating_model, openai_client,
                                               anthropic_client, rating_prompt)
                save_history(results)
                st.session_state['results'] = results
                prompt_scores: Dict[int, Dict[str, List[int]]] = {i: {model: [] for model in selected_models} for i in
                                                                  range(len(prompts))}

                for result in results:
                    for j, mnemonic in enumerate(result['mnemonics']):
                        prompt_index = j % len(prompts)
                        model = mnemonic['model']
                        prompt_scores[prompt_index][model].append(mnemonic['score'])

                # Вывод результатов
                for result in results:
                    st.subheader(f"Слово: {result['word']}")
                    for i, mnemonic in enumerate(result['mnemonics']):
                        st.write(f"Промпт {i % len(prompts) + 1}, Модель: {mnemonic['model']}")
                        st.write(f"Значение: {mnemonic.get('meaning', 'Не указано')}")
                        st.write(f"Ассоциация: {mnemonic.get('association', 'Не указано')}")
                        st.write(f"Визуальный промпт: {mnemonic.get('prompt', 'Не указано')}")
                        st.write(f"Оценка запоминаемости: {mnemonic.get('score', 'Не оценено')}")
                        st.write("---")

                # Визуализация результатов
                visualize_results(results)

                # Вывод средних оценок запоминаемости для каждого промпта и модели
                st.subheader("Средние оценки запоминаемости:")
                for i, model_scores in prompt_scores.items():
                    st.write(f"Промпт {i + 1}:")
                    for model, scores in model_scores.items():
                        avg_score = mean(scores) if scores else 0
                        st.write(f"  {model}: {avg_score:.2f}")

    # Сохранение результатов
    if 'results' in st.session_state:
        json_results = json.dumps(st.session_state['results'], ensure_ascii=False, indent=2)
        st.download_button("Скачать результаты", json_results, "results.json", mime="application/json")

    # История запросов
    if st.button('Показать историю'):
        history = load_history()
        if history:
            st.subheader("История запросов")
            visualize_results(history)
        else:
            st.info("История запросов пуста.")


if __name__ == "__main__":
    openai_api_key, anthropic_api_key = get_api_keys()
    openai_client = get_openai_client(openai_api_key)
    anthropic_client = get_anthropic_client(anthropic_api_key)
    main()
