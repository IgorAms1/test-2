import json
import asyncio
from typing import Dict, Any, List
import streamlit as st
from openai import AsyncOpenAI
from anthropic import Anthropic
from statistics import mean
import logging
import plotly.express as px
import pandas as pd
import os
import csv
import pickle
import hashlib
import threading

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Блокировка для кэширования
cache_lock = threading.Lock()

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

# Кэширование на уровне слов
def get_cached_result(word: str, prompt: str, model: str) -> Dict[str, Any]:
    cache_key = hashlib.md5(f"{word}:{prompt}:{model}".encode()).hexdigest()
    with cache_lock:
        return st.session_state.get(cache_key)

def set_cached_result(word: str, prompt: str, model: str, result: Dict[str, Any]):
    cache_key = hashlib.md5(f"{word}:{prompt}:{model}".encode()).hexdigest()
    with cache_lock:
        st.session_state[cache_key] = result

# Асинхронные функции для работы с API
async def create_mnemonic(word: str, prompt: str, model: str, openai_client: AsyncOpenAI,
                          anthropic_client: Anthropic) -> Dict[str, Any]:
    cached_result = get_cached_result(word, prompt, model)
    if cached_result:
        return cached_result

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
            result = json.loads(content)
        except json.JSONDecodeError:
            result = {"association": content, "meaning": "", "prompt": ""}

        set_cached_result(word, prompt, model, result)
        return result
    except Exception as e:
        logging.error(f"Ошибка при создании мнемоники для '{word}' с моделью {model}: {str(e)}")
        return {"association": f"Ошибка при создании мнемоники с моделью {model}: {str(e)}", "meaning": "",
                "prompt": ""}

async def rate_memory(association: str, model: str, rating_prompt: str, openai_client: AsyncOpenAI,
                      anthropic_client: Anthropic) -> Dict[str, int]:
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
                    max_tokens=100,
                    messages=[
                        {"role": "user", "content": f"{rating_prompt}\n\nAssociation: {association}"}
                    ]
                )
                content = message.content[0].text
            else:
                response = anthropic_client.completions.create(
                    model=model,
                    prompt=f"{rating_prompt}\n\nHuman: {association}\n\nAssistant:",
                    max_tokens_to_sample=100
                )
                content = response.completion
        else:
            raise ValueError(f"Неподдерживаемая модель или отсутствует клиент для модели: {model}")

        # Парсинг результатов
        ratings = {}
        for line in content.split('\n'):
            if ':' in line:
                key, value = line.split(':')
                try:
                    ratings[key.strip()] = int(value.strip())
                except ValueError:
                    logging.warning(f"Не удалось преобразовать значение в число: {line}")

        return ratings
    except Exception as e:
        logging.error(f"Ошибка при оценке запоминаемости с моделью {model}: {str(e)}")
        return {}

async def process_word(word: str, prompts: List[str], models: List[str], rating_model: str, openai_client: AsyncOpenAI,
                       anthropic_client: Anthropic, rating_prompt: str) -> Dict[str, Any]:
    word_results = []
    for i, prompt in enumerate(prompts):
        for model in models:
            mnemonic = await create_mnemonic(word, prompt, model, openai_client, anthropic_client)
            association = mnemonic.get('association', '')
            scores = await rate_memory(association, rating_model, rating_prompt, openai_client, anthropic_client)
            mnemonic.update(scores)
            mnemonic['model'] = model
            mnemonic['prompt_index'] = i
            word_results.append(mnemonic)
    return {"word": word, "mnemonics": word_results}

async def process_all_words(words: List[str], prompts: List[str], models: List[str], rating_model: str,
                            openai_client: AsyncOpenAI, anthropic_client: Anthropic, rating_prompt: str) -> List[
    Dict[str, Any]]:
    tasks = [process_word(word, prompts, models, rating_model, openai_client, anthropic_client, rating_prompt) for word
             in words]
    return await asyncio.gather(*tasks)

@st.cache_data
def cached_process_words(words: List[str], prompts: List[str], models: List[str], rating_model: str,
                         _openai_client: AsyncOpenAI, _anthropic_client: Anthropic, rating_prompt: str) -> List[
    Dict[str, Any]]:
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(
            process_all_words(words, prompts, models, rating_model, _openai_client, _anthropic_client, rating_prompt))
    finally:
        loop.close()

# Функция для сохранения и загрузки истории запросов
def save_history(results: List[Dict[str, Any]], prompts: List[str], filename: str = "history.json"):
    history = {"results": results, "prompts": prompts}
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(history, f, ensure_ascii=False, indent=2)

def load_history(filename: str = "history.json") -> (List[Dict[str, Any]], List[str]):
    if os.path.exists(filename):
        with open(filename, 'r', encoding='utf-8') as f:
            history = json.load(f)
            return history.get("results", []), history.get("prompts", [])
    return [], []

# Функция для визуализации результатов
def visualize_results(results: List[Dict[str, Any]], prompts: List[str]):
    data = []
    for result in results:
        for mnemonic in result['mnemonics']:
            data.append({
                "Word": result['word'],
                "Model": mnemonic['model'],
                "Prompt": f"Промпт {mnemonic['prompt_index'] + 1}",
                "Созвучие": mnemonic.get('Созвучие', 0),
                "Визуальная ассоциация": mnemonic.get('Визуальная ассоциация', 0),
                "Логическая связь": mnemonic.get('Логическая связь', 0),
                "Культурная и эмоциональная релевантность": mnemonic.get('Культурная и эмоциональная релевантность', 0)
            })

    df = pd.DataFrame(data)

    # Получаем цветовую схему
    color_sequence = getattr(px.colors.sequential, st.session_state.color_scheme)

    # Средние значения для каждого промпта
    avg_prompt_scores = df.groupby("Prompt")[["Созвучие", "Визуальная ассоциация", "Логическая связь",
                                              "Культурная и эмоциональная релевантность"]].mean().reset_index()

    fig_avg_prompt = px.bar(avg_prompt_scores, x="Prompt", y=["Созвучие", "Визуальная ассоциация", "Логическая связь",
                                                              "Культурная и эмоциональная релевантность"],
                            barmode="group",
                            title="Средние значения оценок параметров запоминаемости для каждого промпта",
                            color_discrete_sequence=color_sequence)

    st.plotly_chart(fig_avg_prompt)

    # Средние значения для каждой модели
    avg_model_scores = df.groupby("Model")[["Созвучие", "Визуальная ассоциация", "Логическая связь",
                                            "Культурная и эмоциональная релевантность"]].mean().reset_index()

    fig_avg_model = px.bar(avg_model_scores, x="Model", y=["Созвучие", "Визуальная ассоциация", "Логическая связь",
                                                           "Культурная и эмоциональная релевантность"],
                           barmode="group",
                           title="Средние значения оценок параметров запоминаемости для каждой модели",
                           color_discrete_sequence=color_sequence)

    st.plotly_chart(fig_avg_model)

    # Средние значения для каждого слова
    avg_word_scores = df.groupby("Word")[["Созвучие", "Визуальная ассоциация", "Логическая связь",
                                           "Культурная и эмоциональная релевантность"]].mean().reset_index()

    fig_avg_word = px.bar(avg_word_scores, x="Word", y=["Созвучие", "Визуальная ассоциация", "Логическая связь",
                                                        "Культурная и эмоциональная релевантность"],
                          barmode="group",
                          title="Средние значения оценок параметров запоминаемости для каждого слова",
                          color_discrete_sequence=color_sequence)

    st.plotly_chart(fig_avg_word)

def export_to_csv(results: List[Dict[str, Any]], filename: str = "results.csv"):
    with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Word', 'Model', 'Prompt', 'Association', 'Meaning', 'Созвучие', 'Визуальная ассоциация',
                         'Логическая связь', 'Культурная и эмоциональная релевантность'])

        for result in results:
            word = result['word']
            for mnemonic in result['mnemonics']:
                writer.writerow([
                    word,
                    mnemonic['model'],
                    f"Промпт {mnemonic['prompt_index'] + 1}",
                    mnemonic.get('association', ''),
                    mnemonic.get('meaning', ''),
                    mnemonic.get('Созвучие', ''),
                    mnemonic.get('Визуальная ассоциация', ''),
                    mnemonic.get('Логическая связь', ''),
                    mnemonic.get('Культурная и эмоциональная релевантность', '')
                ])

# Функции для управления сессией
def save_session(session_data: Dict[str, Any], filename: str = "session.pkl"):
    with open(filename, 'wb') as f:
        pickle.dump(session_data, f)

def load_session(filename: str = "session.pkl") -> Dict[str, Any]:
    if os.path.exists(filename):
        with open(filename, 'rb') as f:
            return pickle.load(f)
    return {}

# Основной интерфейс приложения
def main():
    st.title('Мнемоническая ассоциация и оценка запоминаемости')

    # Инициализация session_state
    if 'initialized' not in st.session_state:
        st.session_state.initialized = True
        st.session_state.words_input = ""
        st.session_state.selected_models = []
        st.session_state.custom_models = ""
        st.session_state.rating_model = ""
        st.session_state.rating_prompt = """Оцени, пожалуйста, следующую ассоциацию для запоминания иностранного слова по следующим параметрам:

1. Созвучие: Насколько хорошо части ассоциации звучат похоже на исходное иностранное слово? Оцени от 1 до 10.
2. Визуальная ассоциация: Насколько ярко и четко визуализируется образ, предложенный в ассоциации? Оцени от 1 до 10.
3. Логическая связь: Насколько логично и интуитивно понятно связаны элементы ассоциации с переводом или значением слова? Оцени от 1 до 10.
4. Культурная и эмоциональная релевантность: Насколько легко ассоциация воспринимается пользователями с точки зрения культурного контекста и вызывает ли она эмоциональный отклик? Оцени от 1 до 10.

Пожалуйста, верни результат в формате:
Созвучие: [оценка]
Визуальная ассоциация: [оценка]
Логическая связь: [оценка]
Культурная и эмоциональная релевантность: [оценка]
Возвращай только числовые значения."""
        st.session_state.num_prompts = 1
        st.session_state.prompts = [""]
        st.session_state.color_scheme = "Viridis"

    # Загрузка сессии
    if st.button('Загрузить последнюю сессию'):
        session_data = load_session()
        st.session_state.update(session_data)
        st.success("Сессия загружена успешно!")

    # Ввод слов пользователем
    st.session_state.words_input = st.text_area("Введите слова (по одному на строку):", height=150,
                                                value=st.session_state.words_input)
    words = [word.strip() for word in st.session_state.words_input.split('\n') if word.strip()]

    # Предлагаемые модели
    suggested_models = [
         "chatgpt-4o-latest",
        "claude-3-5-sonnet-20240620",
        "claude-3-opus-20240229",
        "claude-3-sonnet-20240229",
        "claude-3-haiku-20240307",
        "gpt-4o",
        "gpt-4o-mini",
        "gpt-4",
        "gpt-4-turbo"
    ]

    # Ввод моделей пользователем
    st.subheader("Выберите модели для генерации мнемоник:")
    st.session_state.selected_models = st.multiselect(
        "Выберите модели из списка или введите свои:",
        options=suggested_models,
        default=st.session_state.selected_models
    )

    # Дополнительное поле для ввода пользовательских моделей
    st.session_state.custom_models = st.text_area(
        "Введите дополнительные модели (по одной на строке):",
        height=100,
        value=st.session_state.custom_models
    )

    # Объединение выбранных и пользовательских моделей
    selected_models = st.session_state.selected_models + [model.strip() for model in
                                                          st.session_state.custom_models.split('\n') if model.strip()]

    if not selected_models:
        st.warning("Пожалуйста, выберите хотя бы одну модель для продолжения.")
        st.stop()

    # Выбор модели для оценки
    st.subheader("Выберите модель для оценки:")
    st.session_state.rating_model = st.selectbox(
        "Выберите модель для оценки из списка или введите свою:",
        options=suggested_models + selected_models,
        index=suggested_models.index(
            st.session_state.rating_model) if st.session_state.rating_model in suggested_models else 0
    )

    # Редактирование стандартного промпта оценки
    st.session_state.rating_prompt = st.text_area("Отредактируйте промпт для оценки запоминаемости:",
                                                  value=st.session_state.rating_prompt, height=300)

    # Ввод промптов пользователем
    st.subheader("Введите промпты:")
    st.session_state.num_prompts = st.number_input("Количество промптов", min_value=1,
                                                   value=st.session_state.num_prompts, step=1)

    st.session_state.prompts = st.session_state.prompts[:st.session_state.num_prompts]
    while len(st.session_state.prompts) < st.session_state.num_prompts:
        st.session_state.prompts.append("")

    prompts = []
    for i in range(st.session_state.num_prompts):
        prompt = st.text_area(f"Промпт {i + 1}", height=100, value=st.session_state.prompts[i], key=f"prompt_{i}")
        prompts.append(prompt)
    st.session_state.prompts = prompts

    # Выбор цветовой схемы для визуализации
    st.session_state.color_scheme = st.selectbox("Выберите цветовую схему:",
                                                 ["Viridis", "Plasma", "Inferno", "Magma"],
                                                 index=["Viridis", "Plasma", "Inferno", "Magma"].index(
                                                     st.session_state.color_scheme))

    if st.button('Генерировать мнемоники и оценки'):
        if not words:
            st.error("Пожалуйста, введите хотя бы одно слово.")
        elif not all(prompts):
            st.error("Пожалуйста, заполните все промпты.")
        elif not (openai_api_key or anthropic_api_key):
            st.error("Пожалуйста, введите хотя бы один API ключ.")
        else:
            if not openai_client and not anthropic_client:
                st.error("Не удалось инициализировать ни один из клиентов API. Пожалуйста, проверьте ваши API ключи.")
            else:
                with st.spinner('Обработка...'):
                    progress_bar = st.progress(0)
                    results = cached_process_words(words, prompts, selected_models, st.session_state.rating_model,
                                                   openai_client,
                                                   anthropic_client, st.session_state.rating_prompt)
                    save_history(results, prompts)
                    st.session_state['results'] = results

                    for i, result in enumerate(results):
                        progress_bar.progress((i + 1) / len(results))

                    # Вывод результатов
                    st.subheader("Результаты:")
                    results_df = pd.DataFrame([
                        {
                            "Слово": result['word'],
                            "Модель": mnemonic['model'],
                            "Промпт": f"Промпт {mnemonic['prompt_index'] + 1}",
                            "Ассоциация": mnemonic.get('association', ''),
                            "Значение": mnemonic.get('meaning', ''),
                            "Созвучие": mnemonic.get('Созвучие', ''),
                            "Визуальная ассоциация": mnemonic.get('Визуальная ассоциация', ''),
                            "Логическая связь": mnemonic.get('Логическая связь', ''),
                            "Культурная и эмоциональная релевантность": mnemonic.get(
                                'Культурная и эмоциональная релевантность', '')
                        }
                        for result in results
                        for mnemonic in result['mnemonics']
                    ])

                    # Фильтрация и сортировка результатов
                    st.subheader("Фильтрация и сортировка результатов:")
                    filter_word = st.text_input("Фильтр по слову:")
                    filter_model = st.selectbox("Фильтр по модели:", ["Все"] + list(results_df["Модель"].unique()))
                    sort_by = st.selectbox("Сортировать по:", results_df.columns)
                    sort_order = st.radio("Порядок сортировки:", ["По возрастанию", "По убыванию"])

                    filtered_df = results_df
                    if filter_word:
                        filtered_df = filtered_df[filtered_df["Слово"].str.contains(filter_word, case=False)]
                    if filter_model != "Все":
                        filtered_df = filtered_df[filtered_df["Модель"] == filter_model]

                    filtered_df = filtered_df.sort_values(by=sort_by, ascending=(sort_order == "По возрастанию"))
                    st.dataframe(filtered_df)

                    # Визуализация результатов
                    st.subheader("Визуализация результатов")
                    visualize_results(results, prompts)

    # Сохранение результатов
    if 'results' in st.session_state:
        st.subheader("Экспорт результатов")
        export_format = st.radio("Выберите формат экспорта:", ("JSON", "CSV"))
        if export_format == "JSON":
            json_results = json.dumps(st.session_state['results'], ensure_ascii=False, indent=2)
            st.download_button("Скачать результаты (JSON)", json_results, "results.json", mime="application/json")
        else:
            export_to_csv(st.session_state['results'])
            with open("results.csv", "rb") as file:
                st.download_button("Скачать результаты (CSV)", file, "results.csv", mime="text/csv")

    # История запросов
    if st.button('Показать историю'):
        history, history_prompts = load_history()
        if history:
            st.subheader("История запросов")
            visualize_results(history, history_prompts)
        else:
            st.info("История запросов пуста.")

    # Сохранение сессии
    if st.button('Сохранить текущую сессию'):
        session_data = {
            "words_input": st.session_state.words_input,
            "selected_models": st.session_state.selected_models,
            "custom_models": st.session_state.custom_models,
            "rating_model": st.session_state.rating_model,
            "rating_prompt": st.session_state.rating_prompt,
            "num_prompts": st.session_state.num_prompts,
            "prompts": st.session_state.prompts,
            "results": st.session_state.get('results', []),
            "color_scheme": st.session_state.color_scheme
        }
        save_session(session_data)
        st.success("Сессия сохранена успешно!")

if __name__ == "__main__":
    openai_api_key, anthropic_api_key = get_api_keys()
    openai_client = get_openai_client(openai_api_key)
    anthropic_client = get_anthropic_client(anthropic_api_key)
    main()
