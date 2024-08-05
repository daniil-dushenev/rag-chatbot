import os
from hugchat import hugchat
from hugchat.login import Login
from typing import Any, List

from sentence_transformers import SentenceTransformer


num_context_messages = 10


def load_bert():
    bert = SentenceTransformer('sergeyzh/LaBSE-ru-turbo')
    bert.to('cpu')
    
    return bert

def hugchat_client(login, password):
    cookie_path_dir = "./cookies/" # NOTE: trailing slash (/) is required to avoid errors
    sign = Login(login, password)
    cookies = sign.login(cookie_dir_path=cookie_path_dir, save_cookies=True)

    # Create your ChatBot
    chatbot = hugchat.ChatBot(cookies=cookies.get_dict())  # or cookie_path="usercookies/<email>.json"
    chatbot.switch_llm(1)
    return chatbot


def get_embeddings(texts, bert):
    try:
        return bert.encode(texts, device='cpu').tolist()
    except:
        return []


def llm_answer(messages, chatbot):
    try:
        history = ""
        if len(messages) >= num_context_messages:
            messages = [messages[0]] + messages[-(num_context_messages-3):]
        for message in messages:
            history += f"{message['role']}:\n{message['content']}\n\n"
            print(message['role'])
            print(message['content'])
            print()

        message_result = chatbot.chat(history) # note: message_result is a generator, the method will return immediately.
        return message_result.wait_until_done()
    

    except Exception as e:
        print(f"An error occurred: {e}")
        return None