from fastapi import FastAPI, HTTPException, Depends, File, UploadFile, Query, BackgroundTasks
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from datetime import datetime, timedelta
from jose import JWTError, jwt
from passlib.context import CryptContext
from sqlalchemy.orm import sessionmaker
from sqlalchemy.exc import IntegrityError
from typing import List, Optional, Dict, Union
from db.db import *
import os
import uvicorn
from ml.models import load_bert, get_embeddings, llm_answer, hugchat_client
import re

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.vectorstores import Chroma
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders.pdf import PyPDFLoader
from typing import List
from langchain.embeddings.base import Embeddings
from chat_config import system_prompt, start_message, chunk_size, top_k, prompt_for_make_pre_answer

import boto3
from botocore.client import Config
import uuid
import asyncio
import traceback



# SQLAlchemy database URL
DATABASE_URL = f"postgresql://{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}@postgres:{os.getenv('DB_PORT')}/{os.getenv('DB_NAME')}"

# SQLAlchemy engine and SessionLocal creation
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base.metadata.create_all(bind=engine)

bert = load_bert()
client = hugchat_client(login=os.getenv("HUGCHAT_LOGIN"), password=os.getenv("HUGCHAT_PASS"))


s3 = boto3.client('s3',
                  endpoint_url=os.getenv("S3_ENDPOINT"),  # Замените на ваш URL MinIO или S3
                  aws_access_key_id=os.getenv("S3_ACCESS_KEY"),         # Ваш access key
                  aws_secret_access_key=os.getenv("S3_SECRET_KEY"),  # Ваш secret key
                  config=Config(signature_version='s3v4'),
                  region_name='us-east-1')


# FastAPI app instance
app = FastAPI()


origins = [
    "http://localhost",
    "http://localhost:8501"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type"],
)


# Dependency to get SQLAlchemy session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()



# Secret key for encoding/decoding JWT
SECRET_KEY = "xologie"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30


# Password hashing context
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


vec_storages = []

# Token creation function
def create_access_token(data: dict, expires_delta: timedelta = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.now() + expires_delta
    else:
        expire = datetime.now() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

# Dependency to get current user from token
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")


def authenticate_user(db: Session, username: str, password: str):
    user = db.query(User).filter(User.username == username).first()
    if not user or not pwd_context.verify(password, user.password_hash):
        return None
    return user

# Registration route
@app.post("/register", response_model=Token)
def register_user(user: UserRegistration, db: Session = Depends(get_db)):
    try:
        hashed_password = pwd_context.hash(user.password)
        db_user = User(username=user.username, email=user.email, password_hash=hashed_password)
        db.add(db_user)
        db.commit()
        db.refresh(db_user)
        access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
        access_token = create_access_token(data={"sub": db_user.username}, expires_delta=access_token_expires)
        return {"access_token": access_token, "token_type": "bearer"}
    except IntegrityError:
        raise HTTPException(status_code=400, detail="Username or email already registered")

# Token route
@app.post("/token", response_model=Token)
def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    user = authenticate_user(db, form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=401,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(data={"sub": user.username}, expires_delta=access_token_expires)
    return {"access_token": access_token, "token_type": "bearer"}

def get_current_user(token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)):
    credentials_exception = HTTPException(
        status_code=401,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
        user = db.query(User).filter(User.username == username).first()
        if user is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception
    return user

# Protected route example
@app.get("/protected")
def protected_route(current_user: User = Depends(get_current_user)):
    if current_user.role != "admin":
        raise HTTPException(
            status_code=403,
            detail="You don't have enough privileges",
        )
    return {"message": f"Hello, {current_user.username}!"}



# CRUD operations for User
@app.post("/users/", response_model=UserInDB)
def create_user(user: UserCreate, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    try:
        db_user = User(username=user.username, account_id=user.account_id)
        db.add(db_user)
        db.commit()
        db.refresh(db_user)
        return db_user
    except IntegrityError:
        raise HTTPException(status_code=400, detail="Username or email already registered")


@app.get("/users/{user_id}", response_model=UserInDB)
def read_user(user_id: int, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    db_user = db.query(User).filter(User.id == user_id).first()
    if db_user is None:
        raise HTTPException(status_code=404, detail="User not found")
    return db_user


@app.put("/users/{user_id}", response_model=UserInDB)
def update_user(user_id: int, user: UserUpdate, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    db_user = db.query(User).filter(User.id == user_id).first()
    if db_user is None:
        raise HTTPException(status_code=404, detail="User not found")
    for key, value in user.dict(exclude_unset=True).items():
        setattr(db_user, key, value)
    db.commit()
    db.refresh(db_user)
    return db_user


@app.delete("/users/{user_id}")
def delete_user(user_id: int, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    db_user = db.query(User).filter(User.id == user_id).first()
    if db_user is None:
        raise HTTPException(status_code=404, detail="User not found")
    db.delete(db_user)
    db.commit()
    return {"detail": "User deleted"}


# CRUD operations for Message
@app.post("/messages/", response_model=MessageInDB)
def create_message(message: MessageCreate, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    db_message = Message(user_id=message.user_id, text=message.text)
    db.add(db_message)
    db.commit()
    db.refresh(db_message)
    return db_message


@app.get("/messages/{message_id}", response_model=MessageInDB)
def read_message(message_id: int, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    db_message = db.query(Message).filter(Message.id == message_id).first()
    if db_message is None:
        raise HTTPException(status_code=404, detail="Message not found")
    return db_message


@app.put("/messages/{message_id}", response_model=MessageInDB)
def update_message(message_id: int, message: MessageUpdate, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    db_message = db.query(Message).filter(Message.id == message_id).first()
    if db_message is None:
        raise HTTPException(status_code=404, detail="Message not found")
    for key, value in message.dict(exclude_unset=True).items():
        setattr(db_message, key, value)
    db.commit()
    db.refresh(db_message)
    return db_message


@app.delete("/messages/{message_id}")
def delete_message(message_id: int, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    db_message = db.query(Message).filter(Message.id == message_id).first()
    if db_message is None:
        raise HTTPException(status_code=404, detail="Message not found")
    db.delete(db_message)
    db.commit()
    return {"detail": "Message deleted"}


def create_message(
    message: MessageCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    # Проверяем, существует ли пользователь, которому предназначено сообщение
    to_user = db.query(User).filter(User.id == message.user_id).first()
    if not to_user:
        raise HTTPException(status_code=404, detail="User not found")

    # Создаем новое сообщение
    db_message = Message(
        text=message.text,
        user_id=message.user_id,  # Устанавливаем ID текущего пользователя как отправителя
        to_user_id=message.to_user_id  # Устанавливаем ID пользователя, которому предназначено сообщение
    )

    # Добавляем сообщение в базу данных
    db.add(db_message)
    db.commit()
    db.refresh(db_message)
    
    return db_message


class TextRequest(BaseModel):
    text: str

# Модель для списка текстов
class TextsRequest(BaseModel):
    texts: List[str]

# Модель для сообщений
class MessagesRequest(BaseModel):
    messages: List[Dict]

@app.post("/embedding")
def make_embedding(request: TextRequest, current_user: User = Depends(get_current_user)):
    text = request.text
    embs = get_embeddings([text], bert)
    return {"embedding": embs[0]}

@app.post("/embeddings")
def make_embeddings(request: TextsRequest, current_user: User = Depends(get_current_user)):
    texts = request.texts
    embs = get_embeddings(texts, bert)
    return {"embeddings": embs}

@app.post("/llm_answer")
def make_llm_answer(request: MessagesRequest, current_user: User = Depends(get_current_user)):
    messages = request.messages
    model_answer = llm_answer(messages, client)
    return {"answer": model_answer}


class AbbreviationFinder:
    def __init__(self):
        # Initialize the rule base with some predefined rules
        self.rule_base = [
            ('c'*i, 'c'*i) for i in range(2, 10)
            # Add more rules as needed
        ]
        self.stopwords = set(["в", "с", "к", "по", "за", "от", "для", "без", "под", "о", "об", "при"])

    def find_abbreviations(self, text):
        text = self._remove_non_letters(text)
        # Step 1: Find candidate abbreviations
        abbreviations = self._extract_abbreviations(text)
        # Step 2: Find candidate definitions
        definitions = self._extract_definitions(text, abbreviations)
        # Step 3: Match abbreviations with definitions
        abbreviation_definitions = self._match_abbreviations_definitions(abbreviations, definitions)
        
        return abbreviation_definitions

    def _remove_non_letters(self, text):
    # Замена всех символов, кроме букв, пробелов и перевода строк на пустую строку
        cleaned_text = re.sub(r'[^А-Яа-яA-Za-zёЁ\s]', ' ', text)
        return cleaned_text

    
    def _extract_abbreviations(self, text):
        # Regex to find potential abbreviations
        abbreviation_pattern = r'\b\w*[А-ЯA-ZЁ]\w*[А-ЯA-ZЁ]\w*\b'
        return re.findall(abbreviation_pattern, text)
    
    def _extract_definitions(self, text, abbreviations):
        definitions = {}
        for abbreviation in abbreviations:
            # Create a search space around each occurrence of the abbreviation
            search_spaces = self._create_search_spaces(text, abbreviation)
            for search_space in search_spaces:
                candidate_definitions = self._find_definitions_in_search_space(abbreviation, search_space)
                if abbreviation in definitions:
                    definitions[abbreviation].extend(candidate_definitions)
                else:
                    definitions[abbreviation] = candidate_definitions
        return definitions
    
    def _create_search_spaces(self, text, abbreviation):
        # Find all occurrences of the abbreviation
        search_spaces = []
        words = text.split()
        for index in range(len(words)):
            if words[index] == abbreviation:
                left_context = words[max(0, index-10):index]
                right_context = words[index+1:min(len(words), index+11)]
                search_spaces.append(left_context + [abbreviation] + right_context)
        return search_spaces
    
    def _find_definitions_in_search_space(self, abbreviation, search_space):
        candidate_definitions = []
        for i in range(len(search_space)):
            for j in range(i+1, len(search_space)+1):
                definition = search_space[i:j]
                if self._is_valid_definition(definition):
                    candidate_definitions.append(" ".join(definition))
        return candidate_definitions
    
    def _is_valid_definition(self, definition):
        if not definition:
            return False
        if definition[0].lower() in self.stopwords or definition[-1].lower() in self.stopwords:
            return False
        return True
    
    def _match_abbreviations_definitions(self, abbreviations, definitions):
        matches = {}
        for abbreviation in abbreviations:
            abbreviation_pattern = self._generate_pattern(abbreviation)
            for definition in definitions[abbreviation]:
                definition_pattern = self._generate_pattern(definition.split())
                if (abbreviation_pattern, definition_pattern) in self.rule_base and self._first_letters_match(abbreviation, definition):
                    matches[abbreviation] = definition
                    break
        return matches

    def _first_letters_match(self, abbreviation, definition):
        abbreviation_letters = abbreviation.replace('ё', 'е').upper()
        definition_words = definition.split()
        definition_letters = "".join(word[0].upper() for word in definition_words)
        return abbreviation_letters == definition_letters

                    
        return matches
    
    def _generate_pattern(self, word_list):
        pattern = []
        for word in word_list:
            if word.isalpha():
                pattern.append('c')
            elif word.isdigit():
                pattern.append('n')
            else:
                pattern.append('c')
        return "".join(pattern)


class AbbreviationReplacer:
    def __init__(self, abbreviation_dict):
        self.abbreviation_dict = abbreviation_dict
        # Compile a regex pattern to match any abbreviation in the dictionary
        self.pattern = re.compile(r'\b(' + '|'.join(map(re.escape, abbreviation_dict.keys())) + r')\b')

    def replace_abbreviations(self, text):
        try:
            return self.pattern.sub(lambda match: f"{self.abbreviation_dict[match.group(0)]} ({match.group(0)})", text)
        except:
            return text


finder = AbbreviationFinder()


def get_abbreviation_definitions(documents):
    ans = {}
    for document in documents:
        text = document.page_content
        abbreviation_definitions = finder.find_abbreviations(text)
        ans = ans | abbreviation_definitions
    return ans


def extract_abbr(documents, abbreviation_definitions):
    replacer = AbbreviationReplacer(abbreviation_definitions)
    for document in documents:
        document.page_content = replacer.replace_abbreviations(document.page_content)
    return documents


class CustomEmbeddings(Embeddings):
    def __init__(self):
        pass
        
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        embeddings = []
        embeddings = get_embeddings(texts, bert)
        return embeddings

    def embed_query(self, text: str) -> List[float]:
        return get_embeddings([text], bert)[0]


class FileRequest(BaseModel):
    files: List[str]
    chunk_size: int


def create_bucket(bucket_name, s3):
    try:
        response = s3.create_bucket(Bucket=bucket_name)
        print(f'Бакет {bucket_name} успешно создан.')
    except s3.exceptions.BucketAlreadyOwnedByYou:
        print(f'Бакет с именем {bucket_name} уже существует и принадлежит вам.')
    except s3.exceptions.BucketAlreadyExists:
        print(f'Бакет с именем {bucket_name} уже существует.')
    except Exception as e:
        print(f'Ошибка при создании бакета: {e}')


def save_file2bucket(bucket_name, file_path, object_name, s3):
    try:
        s3.upload_file(file_path, bucket_name, object_name)
        print(f'Файл {file_path} успешно загружен в бакет {bucket_name} под именем {object_name}.')
    except Exception as e:
        print(f'Ошибка при загрузке файла: {e}')


UPLOAD_DIR = 'data'
os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.post("/upload_pdf/")
async def upload_pdf(file: UploadFile = File(...)):
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    file_location = os.path.join(UPLOAD_DIR, file.filename)
    
    with open(file_location, "wb") as f:
        content = await file.read()
        f.write(content)
    
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    return {"path": file_path, 'name': file.filename}

@app.delete("/delete_pdf/{filename}")
async def delete_pdf(filename: str):
    file_location = os.path.join(UPLOAD_DIR, filename)
    
    if os.path.isfile(file_location):
        os.remove(file_location)
        return {"message": f"File '{filename}' has been deleted."}
    else:
        raise HTTPException(status_code=404, detail="File not found")
    

def load_vec_storage_from_s3(vec_name):
    local_path = f"./chroma/{vec_name}"
    vec_storage = Chroma(persist_directory=local_path, embedding_function=CustomEmbeddings())
    return vec_storage


def create_vec_db(vec_name, source, is_public, db: Session):
    db_vec_db = VecDb(
        name=vec_name,
        source=source,
        is_public=is_public
    )
    db.add(db_vec_db)
    db.commit()
    db.refresh(db_vec_db)
    return db_vec_db


@app.post("/vectorStorageFromFiles")
async def get_vectorstore_from_files(
    vec_name: str = Query(..., description="Name of the vector storage"),
    files: List[UploadFile] = File(...),
    chunk_size: int = Query(2048, description="Chunk size for text splitting"),
    is_public: bool = False,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
    ):
    document_chunks = []
    create_bucket(vec_name, s3)
    for file in files:
        file = await upload_pdf(file)

        save_file2bucket(vec_name, file["path"], file["name"], s3)

        loader = PyPDFLoader(file['path'])
        document = loader.load()
        await delete_pdf(file['name'])


        # собираем аббревиатуры из текста и меняем их на расшифровки
        abbreviation_definitions = get_abbreviation_definitions(document)
        print(abbreviation_definitions)
        document = extract_abbr(document, abbreviation_definitions)
        
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size,
    chunk_overlap=int(chunk_size)/4)
        chunks = text_splitter.split_documents(document)
        
        document_chunks.extend(chunks)
    
    # Вывод всех разделенных кусков
    for i, chunk in enumerate(document_chunks):
        print(f"Chunk {i+1}:")
        print(chunk)
    
    embeddings = CustomEmbeddings()
    vector_storage = Chroma.from_documents(document_chunks, embeddings, persist_directory=f"./chroma/{vec_name}")
    vector_storage.persist()

    vec_storages.append(vec_name)
    print(vec_storages)
    create_vec_db(vec_name, f"./chroma/{vec_name}", is_public, db)
    return {"response": "Ok"}


@app.get('/getVectorStoresList')
def getVectorStoresList(current_user: User = Depends(auth)):
    return {"answer": vec_storages}




def get_fake_answer(query):
    """Создает похожий на реальный ответ на запрос пользователя без учета контекста, этот ответ будем использовать для поиска релевантной информации
    """
    prompt = prompt_for_make_pre_answer + query
    messages = [{"role": "user", "content": prompt}]
    model_answer = llm_answer(messages, client)
    return model_answer
        

def get_relevant_information(user_query, chat_history, vec_storage_ind):
    """Retrieve relevant information from vector store based on user query."""
    vector_store = load_vec_storage_from_s3(vec_storage_ind)
    history2prompt = ""
    for message in chat_history:
        message_type = "assistant" if isinstance(message, AIMessage) else "user"
        text = message.content
        history2prompt += f"{message_type}:\n{text}\n\n"
    
    history2prompt += "user:\n"    
    llm_query = get_fake_answer(history2prompt + user_query)
    print("LLM query")
    print(llm_query)
    query2vecstore = user_query + '\n' + llm_query
    result = vector_store.similarity_search(query2vecstore, k=top_k)
    print(result)
    if result:
        return '\n\n'.join([str(doc) for doc in result])
    else:
        return None


def merge_consecutive_messages(messages):
    if not messages:
        return []

    merged_messages = []
    current_message = messages[0]

    for next_message in messages[1:]:
        if next_message['role'] == current_message['role']:
            current_message['content'] += " " + next_message['content']
        else:
            merged_messages.append(current_message)
            current_message = next_message

    merged_messages.append(current_message)
    return merged_messages


def convert_chatpromptvalue_to_list(chatprompt_data):
    messages = chatprompt_data.to_messages()
    result = []

    for message in messages:
        if isinstance(message, AIMessage):
            role = "assistant"
        else:
            role = "user"
        result.append({
                "role": role,  # Предполагаемый default role "assistant"
                "content": str(message.content)
            })
    result = merge_consecutive_messages(result)
    return result
    

def custom_llm(prompt):
    """Custom LLM function to interact with the custom chat API."""
    messages = convert_chatpromptvalue_to_list(prompt)
    model_answer = llm_answer(messages, client)
    return model_answer



def get_conversational_rag_chain(retriever):
    """Create a conversational RAG chain using the given context retriever chain."""
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt + """
            {context}
            """),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}")
    ])

    stuff_documents_chain = create_stuff_documents_chain(custom_llm, prompt)
    return create_retrieval_chain(retriever, stuff_documents_chain)

def to_langchain_templates(messages):
    new_messages = []
    for mes in messages:
        if mes.role == "assistant":
            new_messages.append(AIMessage(content=mes.content))
        else:
            new_messages.append(HumanMessage(content=mes.content))
    return new_messages



class ChatMessage(BaseModel):
    role: str
    content: str

class GetAnswerRequest(BaseModel):
    user_input: str
    username: str
    chat_id: int


def get_messages_by_chat_id(db: Session, chat_id: int):
    """
    Получение всех сообщений по chat_id.

    :param db: Сессия базы данных.
    :param chat_id: Идентификатор чата.
    :return: Список сообщений в чате.
    """
    return (
        db.query(Message)
        .join(MessageChat, Message.id == MessageChat.message_id)
        .filter(MessageChat.chat_id == chat_id)
        .order_by(Message.created_at)  # Сортировка по времени создания
        .all()
    )

def get_messages(chat_id: int, db: Session):
    messages = get_messages_by_chat_id(db, chat_id)

    messages_list = []

    for message in messages:
        if message.user_id == 0: 
            mes = {"role": "assistant", "content": message.text}
        else:
            mes = {"role": "user", "content": message.text}
        messages_list.append(mes)

    return messages_list

def create_message_chat(db: Session, chat_id: int, message_id: int):
    message_chat = MessageChat(chat_id=chat_id, message_id=message_id)
    db_message_chat = MessageChat(
        chat_id=message_chat.chat_id,
        message_id=message_chat.message_id
    )
    db.add(db_message_chat)
    db.commit()
    db.refresh(db_message_chat)
    return db_message_chat


def add_new_message(username: str, text: str, to_assistant: bool, chat_id: int, db: Session):

    # Получаем идентификатор пользователя по username
    user_id = db.query(User).filter(User.username == username).first().id
    
    if to_assistant:
        new_message = Message(text=text, user_id=user_id, to_user_id = 0)
    else:
        new_message = Message(text=text, user_id=0, to_user_id = user_id)
    mes = create_message(new_message, db=db)
    create_message_chat(db, chat_id, mes.id)



def create_task(db: Session, task_id: str, result: str, value: str):
    new_task = Task(task_id=task_id, result=result, value=value)
    try:
        db.add(new_task)
        db.commit()
        db.refresh(new_task)
    except IntegrityError:
        db.rollback()
        raise ValueError("Task with this task_id already exists")
    return new_task

# Функция для получения записи по task_id
def get_result_and_value_by_task_id(db: Session, task_id: str):
    task = db.query(Task).filter(Task.task_id == task_id).first()
    if task:
        return task.result, task.value
    return None, None


def update_task_result_and_value(db: Session, task_id: str, new_result: str, new_value: str):
    task = db.query(Task).filter(Task.task_id == task_id).first()
    if task:
        task.result = new_result
        task.value = new_value
        db.commit()
        db.refresh(task)
        return task
    return None


def get_username_and_vecname_by_chat_id(db: Session, chat_id: int):
    """
    Получение username и vec_name по chat_id.

    :param db: Сессия базы данных.
    :param chat_id: Идентификатор чата.
    :return: Словарь с username и vec_name.
    """
    result = (
        db.query(User.username, VecDb.name)
        .join(Chat, Chat.user_id == User.id)
        .join(VecDb, Chat.vec_db_id == VecDb.id)
        .filter(Chat.id == chat_id)
        .first()
    )
    
    if result:
        return result.username, result.name
    return None

async def get_llm_ans(task_id: str, user_input: str, chat_id: int, db: Session):
    try:
        chat_history = await asyncio.to_thread(get_messages, chat_id, db)
        chat_history = to_langchain_templates(chat_history)

        username, vec_storage_ind = await asyncio.to_thread(get_username_and_vecname_by_chat_id, db, chat_id)

        await asyncio.to_thread(add_new_message, username, user_input, True, chat_id, db)
        
        # Используем asyncio.to_thread для вызова блокирующих операций
        context = await asyncio.to_thread(get_relevant_information, user_input, chat_history, vec_storage_ind)
        vector_store = load_vec_storage_from_s3(vec_storage_ind)
        retriever = await asyncio.to_thread(lambda: vector_store.as_retriever())
        conversation_rag_chain = await asyncio.to_thread(lambda: get_conversational_rag_chain(retriever))

        inv_response = await asyncio.to_thread(lambda: conversation_rag_chain.invoke({
            "chat_history": chat_history,
            "context": context,
            "input": user_input
        }))
        response = inv_response['answer']

        await asyncio.to_thread(add_new_message, username, response, False, chat_id, db)

        update_task_result_and_value(db, task_id, "completed", response)


    except Exception as e:
        # Если возникла ошибка, сохраняем ее в статус задачи
        tb_str = traceback.format_exception(e)
        error_message = ''.join(tb_str)
        update_task_result_and_value(db, task_id, "failed", error_message)


@app.post("/get_response")
async def get_response(request: GetAnswerRequest, background_tasks: BackgroundTasks, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    task_id = str(uuid.uuid4())

    # Сохраняем начальный статус задачи
    create_task(db, task_id, "in_progress", None)

    # Запускаем обработку задачи в фоне
    background_tasks.add_task(get_llm_ans, task_id, request.user_input, request.chat_id, db)
    
    return {"task_id": task_id}


@app.get("/task_status/{task_id}")
async def get_task_status(task_id: str, db: Session = Depends(get_db)):
    status, value = get_result_and_value_by_task_id(db, task_id)
    if status == "failed":
        raise HTTPException(status_code=500, detail=f"Task failed: {value}")
    if status is None:
        raise HTTPException(status_code=404, detail="Task not found")
    return {"status": status, "response": value}


@app.post("/create_chat")
def create_chat(username: str, vec_name: str, db: Session = Depends(get_db)):
    vec_db_id = db.query(VecDb).filter(VecDb.name == vec_name).first().id
    user_id = db.query(User).filter(User.id == user_id).first().id
    db_chat = Chat(
        vec_db_id=vec_db_id,
        user_id=user_id
    )
    db.add(db_chat)
    db.commit()
    db.refresh(db_chat)
    return {"chat_id": db_chat.id}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=os.getenv('BACK_PORT'))