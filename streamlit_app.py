import json
from typing import Dict, Any, List, Coroutine
import asyncio
import streamlit as st
from openai import AsyncOpenAI
from anthropic import Anthropic
from statistics import mean
import logging

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Функция для получения API ключей
def get_api_keys():
    st.sidebar.header("API Ключи")
    
    openai_key = st.sidebar.text_input("Введите ваш API ключ OpenAI:", type="password", key="openai_input")
    anthropic_key = st.sidebar.text_input("Введите ваш API ключ Anthropic:", type="password", key="anthropic_input")
    
    if st.sidebar.button("Сохранить ключи"):
        st.session_state["openai_key"] = openai_key
        st.session_state["anthropic_key"] = anthropic_key
        st.sidebar.success("Ключи сохранены!")
    
    return st.session_state.get("openai_key", openai_key), st.session_state.get("anthropic_key", anthropic_key)

# Получаем API ключи
openai_api_key, anthropic_api_key = get_api_keys()

# Настройка клиентов
@st.cache_resource
def get_openai_client(api_key):
    return AsyncOpenAI(api_key=api_key) if api_key else None

@st.cache_resource
def get_anthropic_client(api_key):
    return Anthropic(api_key=api_key) if api_key else None

openai_client = get_openai_client(openai_api_key)
anthropic_client = get_anthropic_client(anthropic_api_key)

# Выбор моделей
available_models = {
    "OpenAI": ["gpt-3.5-turbo", "gpt-4"],
    "Anthropic": ["claude-2", "claude-instant-1"]
}

selected_models = st.multiselect("Выберите модели для использования:", 
                                 [model for provider in available_models for model in available_models[provider]])

if not selected_models:
    st.warning("Пожалуйста, выберите хотя бы одну модель для продолжения.")
    st.stop()

async def create_mnemonic(word: str, prompt: str, model: str) -> Dict[str, Any]:
    try:
        if model in available_models["OpenAI"]:
            if not openai_client:
                raise ValueError("API ключ OpenAI не предоставлен")
            response = await openai_client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": word}
                ]
            )
            content = response.choices[0].message.content
        elif model in available_models["Anthropic"]:
            if not anthropic_client:
                raise ValueError("API ключ Anthropic не предоставлен")
            response = anthropic_client.completions.create(
                model=model,
                prompt=f"{prompt}\n\nHuman: {word}\n\nAssistant:",
                max_tokens_to_sample=300
            )
            content = response.completion
        else:
            raise ValueError(f"Неподдерживаемая модель: {model}")
        
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            return {"association": content, "meaning": "", "prompt": ""}
    except Exception as e:
        logging.error(f"Ошибка при создании мнемоники для '{word}' с моделью {model}: {str(e)}")
        return {"association": f"Ошибка при создании мнемоники с моделью {model}: {str(e)}", "meaning": "", "prompt": ""}

async def rate_memory(association: str, model: str) -> int:
    try:
        rating_prompt = "You are a memory expert. Rate the following mnemonic association on a scale from 1 to 100 based on how easy it is to remember. Return only the numeric score."
        
        if model in available_models["OpenAI"]:
            if not openai_client:
                raise ValueError("API ключ OpenAI не предоставлен")
            response = await openai_client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": rating_prompt},
                    {"role": "user", "content": association}
                ]
            )
            content = response.choices[0].message.content
        elif model in available_models["Anthropic"]:
            if not anthropic_client:
                raise ValueError("API ключ Anthropic не предоставлен")
            response = anthropic_client.completions.create(
                model=model,
                prompt=f"{rating_prompt}\n\nHuman: {association}\n\nAssistant:",
                max_tokens_to_sample=10
            )
            content = response.completion
        else:
            raise ValueError(f"Неподдерживаемая модель: {model}")
        
        return int(content)
    except ValueError as ve:
        logging.warning(f"Ошибка при оценке запоминаемости: {str(ve)}")
        return 0
    except Exception as e:
        logging.error(f"Ошибка при оценке запоминаемости с моделью {model}: {str(e)}")
        return 0

async def process_word(word: str, prompts: List[str], models: List[str]) -> Dict[str, Any]:
    word_results = []
    for prompt in prompts:
        for model in models:
            mnemonic = await create_mnemonic(word, prompt, model)
            association = mnemonic.get('association', '')
            score = await rate_memory(association, model)
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
def cached_process_words(words: List[str], prompts: List[str], models: List[str]) -> List[Dict[str, Any]]:
    return [run_async(process_word(word, prompts, models)) for word in words]

st.title('Мнемоническая ассоциация и оценка запоминаемости')

# Ввод слов пользователем
words_input = st.text_area("Введите слова (по одному на строку):", height=150)
words = [word.strip() for word in words_input.split('\n') if word.strip()]

# Ввод промптов пользователем
st.subheader("Введите промпты:")
num_prompts = st.number_input("Количество промптов", min_value=1, value=1, step=1)
prompts = []
for i in range(num_prompts):
    prompt = st.text_area(f"Промпт {i+1}", height=100, key=f"prompt_{i}")
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
            results = cached_process_words(words, prompts, selected_models)
            prompt_scores: Dict[int, Dict[str, List[int]]] = {i: {model: [] for model in selected_models} for i in range(len(prompts))}

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

            # Вывод средних оценок запоминаемости для каждого промпта и модели
            st.subheader("Средние оценки запоминаемости:")
            for i, model_scores in prompt_scores.items():
                st.write(f"Промпт {i+1}:")
                for model, scores in model_scores.items():
                    avg_score = mean(scores) if scores else 0
                    st.write(f"  {model}: {avg_score:.2f}")

            # Опция для сохранения результатов
            if st.button('Сохранить результаты'):
                with open('results.json', 'w', encoding='utf-8') as f:
                    json.dump(results, f, ensure_ascii=False, indent=2)
                st.success('Результаты сохранены в файл results.json')
