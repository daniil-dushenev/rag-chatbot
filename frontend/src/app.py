import os
import streamlit as st
import requests
import atexit
from typing import List
from chat_config import start_message, chunk_size, top_k


fastapi_host = f"http://fastapi:{os.getenv('BACK_PORT')}"


def get_vectorstore_from_files(file_path, chunk_size, access_token):
    print(file_path)
    headers = {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json"
    }
    response = requests.post(f"{fastapi_host}/vectorStorageFromFiles", json={"files": file_path, "chunk_size": chunk_size}, headers=headers)
    if response.status_code == 200:
        ind = response.json()['index']
    else:
        raise Exception(f"Failed to get embedding: {response.status_code} {response.text}")
    return ind



def get_response(user_input, vec_storage_ind):
    if 'chat_history' in st.session_state:
        chat_history = st.session_state.chat_history
    token = st.session_state.access_token

    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json"
    }

    response = requests.get(f"{fastapi_host}/get_response", json={"user_input": user_input, "chat_history": chat_history, 'vec_storage_ind': vec_storage_ind},
                            headers=headers
                            )
    if response.status_code == 200:
        response = response.json()['response']
    else:
        raise Exception(f"Failed to get embedding: {response.status_code} {response.text}")
        
    return response


DATA_DIR = "/app/data"

def save_uploaded_files(uploaded_files):
    """
    Сохраняет загруженные файлы в директорию /app/data.

    Args:
        uploaded_files (list): Список загруженных файлов.

    Returns:
        list: Список путей сохраненных файлов.
    """
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
    file_paths = []
    for uploaded_file in uploaded_files:
        file_path = os.path.join(DATA_DIR, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.read())
        file_paths.append(file_path)
    return file_paths

def cleanup_data_directory():
    """
    Удаляет все файлы и директорию /app/data.
    """
    if os.path.exists(DATA_DIR):
        for file_name in os.listdir(DATA_DIR):
            file_path = os.path.join(DATA_DIR, file_name)
            if os.path.isfile(file_path):
                os.unlink(file_path)
        os.rmdir(DATA_DIR)

# Register cleanup function to be called on exit
atexit.register(cleanup_data_directory)

# Основная часть страницы входа
def login_page():
    st.title("Login")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        try:
            response = requests.post(f"{fastapi_host}/token", 
                data={
                    "grant_type": "",
                    "username": username,
                    "password": password,
                    "scope": "",
                    "client_id": "",
                    "client_secret": ""
                },
                headers={
                    "Content-Type": "application/x-www-form-urlencoded"
                })
            if response.status_code == 200:
                access_token = response.json()["access_token"]
                st.session_state.access_token = access_token
                st.success("Logged in successfully!")
                return access_token  # Возврат значения True для перехода на страницу чата
            else:
                st.error("Authentication failed. Please check your credentials.")
        except requests.exceptions.RequestException as e:
            st.error(f"Error: {e}")
            st.error(f"URL: {e.request.url}")
            st.error(f"Request body: {e.request.body}")
    
    return False  # Возврат значения False для оставления на странице входа

def chat_page(access_token):
    st.set_page_config(page_title="Chat with Documents", page_icon="📄")
    st.title("Chat with Documents")
    with st.sidebar:
        st.header("Settings")
        document_files = st.file_uploader("Upload Documents", accept_multiple_files=True, type=["pdf"])

    if document_files is None:
        st.info("Please upload document files and provide the API URL for embeddings")
    else:
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = [
                {"role": "assistant", "content": start_message}
            ]

        if "vector_store" not in st.session_state:
            file_path = save_uploaded_files(document_files)
            st.session_state.vector_store = get_vectorstore_from_files(file_path, chunk_size, access_token)

        user_query = st.chat_input("Type your message ✍")
        if user_query is not None and user_query != "":
            response = get_response(user_query, st.session_state.vector_store)
            st.session_state.chat_history.append({"role": "user", "content": user_query})
            st.session_state.chat_history.append({"role": "assistant", "content": response})

        for message in st.session_state.chat_history:
            message_type = "AI" if message['role'] == 'assistant' else "Human"
            with st.chat_message(message_type):
                st.write(message['content'])


# Основная логика приложения
if "access_token" in st.session_state:
    # Если токен доступа есть, отображаем страницу чата
    chat_page(st.session_state.access_token)
else:
    # Если токена нет, отображаем страницу входа
    token = login_page()
    if token:
        st.experimental_rerun()
        # Если успешно залогинились, переключаемся на страницу чата
        chat_page(token)