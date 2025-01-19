import re
from pylatexenc.latex2text import LatexNodes2Text
from sentence_transformers import SentenceTransformer
import faiss
import requests
import streamlit as st
import json

# Функция для очистки LaTeX
def clean_advanced_tex(latex_content):
    content = re.sub(r'%.*', '', latex_content)  # Удаление комментариев
    content = re.sub(r'\\begin{.*?}.*?\\end{.*?}', '', content, flags=re.DOTALL)  # Удаление окружений
    content = re.sub(r'\\[a-zA-Z]+{.*?}', '', content)  # Удаление команд с аргументами
    content = re.sub(r'\\[a-zA-Z]+', '', content)  # Удаление одиночных команд
    content = re.sub(r'[$_^&{}]', '', content)  # Удаление специальных символов
    content = re.sub(r'\s+', ' ', content).strip()  # Удаление лишних пробелов
    return content

# Разбиение текста на блоки
def chunk_text(text, chunk_size=200):
    sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', text)
    chunks, current_chunk, current_length = [], [], 0
    
    for sentence in sentences:
        current_length += len(sentence.split())
        current_chunk.append(sentence)
        if current_length >= chunk_size:
            chunks.append(" ".join(current_chunk))
            current_chunk, current_length = [], 0
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    return chunks

# Генерация ответа с помощью Ollama (с обработкой потока)
def generate_answer(prompt, context):
    url = "http://localhost:11434/api/generate"
    prompt = (
        f"Ты — помощник, который отвечает строго на основе заданного контекста. "
        f"Если ответ нельзя найти в контексте, напиши: 'Этого в учебнике нет.'\n\n"
        f"Контекст: {context}\n\n"
        f"Вопрос: {prompt}\n\n"
        f"Ответ: "
    )
    payload = {
        "model": "mistral",
        "prompt": prompt
    }

    with requests.post(url, json=payload, stream=True) as response:
        if response.status_code == 200:
            # Обработка потокового ответа построчно
            full_response = ""
            for line in response.iter_lines():
                if line:  # Пропустить пустые строки
                    try:
                        json_line = json.loads(line.decode("utf-8"))
                        full_response += json_line.get("response", "")  # Извлечение текстовой части ответа
                    except json.JSONDecodeError:
                        raise Exception(f"Ошибка декодирования JSON: {line.decode('utf-8')}")
            return full_response.strip()
        else:
            raise Exception(f"Ollama API Error: {response.status_code}, {response.text}")

# Поиск и генерация ответа
def search_and_generate(query, index, chunks, model, similarity_threshold=0.3):
    query_embedding = model.encode([query])
    distances, indices = index.search(query_embedding, k=5)

    # Рассчитываем среднюю релевантность (1 - дистанция, т.к. используется L2-норма)
    relevances = [1 - distances[0][i] for i in range(len(distances[0]))]
    average_relevance = -sum(relevances) / len(relevances)
    print(average_relevance)

    # Если релевантность ниже порога, возвращаем сообщение "Этого в учебнике нет."
    if average_relevance < similarity_threshold:
        return "Этого в учебнике нет."

    # Если релевантность достаточная, извлекаем контекст
    relevant_contexts = [chunks[i] for i in indices[0]]
    combined_context = "\n".join(relevant_contexts)

    return generate_answer(query, combined_context)

# Основная программа
def main():
    # Загрузка и очистка данных
    with open('../data/lecture_notes.tex', 'r', encoding='utf-8') as f:
        raw_content = f.read()
    cleaned_content = clean_advanced_tex(raw_content)
    chunks = chunk_text(cleaned_content)

    # Инициализация модели SentenceTransformer
    model = SentenceTransformer('all-MiniLM-L6-v2')

    # Создание эмбеддингов и индекса
    embeddings = model.encode(chunks, show_progress_bar=True)
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)

    # Интерфейс Streamlit
    st.title("RAG-пайплайн с Ollama Mistral")
    query = st.text_input("Введите вопрос:")

    if query:
        try:
            # Отображение результата потока
            st.write("Ответ:")
            response_area = st.empty()
            full_answer = ""
            for part in search_and_generate(query, index, chunks, model):
                full_answer += part
                response_area.write(full_answer)
        except Exception as e:
            st.error(f"Ошибка: {str(e)}")

if __name__ == "__main__":
    main()
