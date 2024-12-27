import os
from typing import Any, List
from openai import OpenAI

from sentence_transformers import SentenceTransformer


num_context_messages = 10


def load_bert():
    bert = SentenceTransformer('sergeyzh/LaBSE-ru-turbo')
    bert.to('cpu')
    
    return bert

def get_client(token):
    client = OpenAI(
        base_url="https://api-inference.huggingface.co/v1/",
        api_key=token
    )
    return client


def get_embeddings(texts, bert):
    try:
        return bert.encode(texts, device='cpu').tolist()
    except:
        return []


def llm_answer(messages, client, model_name=os.getenv("MODEL_NAME")):
    try:
        history = ""
        if len(messages) >= num_context_messages:
            messages = [messages[0]] + messages[-(num_context_messages-3):]
        for message in messages:
            history += f"{message['role']}:\n{message['content']}\n\n"
            print(message['role'])
            print(message['content'])
            print()

        completion = client.chat.completions.create(
            model=model_name, 
            messages=messages, 
            temperature=0.7,
            max_tokens=2048
        )
        return completion.choices[0].message.content
    

    except Exception as e:
        print(f"An error occurred: {e}")
        return None