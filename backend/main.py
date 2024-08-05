from fastapi import FastAPI, HTTPException, Depends
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

# SQLAlchemy database URL
DATABASE_URL = f"postgresql://{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}@postgres:{os.getenv('DB_PORT')}/{os.getenv('DB_NAME')}"

# SQLAlchemy engine and SessionLocal creation
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base.metadata.create_all(bind=engine)

bert = load_bert()
client = hugchat_client(login=os.getenv("HUGCHAT_LOGIN"), password=os.getenv("HUGCHAT_PASS"))
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


@app.post("/vectorStorageFromFiles")
def get_vectorstore_from_files(request: FileRequest, current_user: User = Depends(get_current_user)):
    files = request.files
    chunk_size = request.chunk_size
    document_chunks = []

    for file in files:
        loader = PyPDFLoader(file)
        document = loader.load()

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
    vector_storage = Chroma.from_documents(document_chunks, embeddings)
    vec_storages.append(vector_storage)
    ind = len(vec_storages) - 1
    return {"index": ind}


def get_fake_answer(query):
    """Создает похожий на реальный ответ на запрос пользователя без учета контекста, этот ответ будем использовать для поиска релевантной информации
    """
    prompt = prompt_for_make_pre_answer + query
    messages = [{"role": "user", "content": prompt}]
    model_answer = llm_answer(messages, client)
    return model_answer
        

def get_relevant_information(user_query, chat_history, vec_storage_ind):
    """Retrieve relevant information from vector store based on user query."""
    vector_store =  vec_storages[vec_storage_ind]

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
    chat_history: List[ChatMessage]
    vec_storage_ind: int

@app.get("/get_response")
def get_response(request: GetAnswerRequest, current_user: User = Depends(get_current_user)):
    user_input = request.user_input
    chat_history = request.chat_history
    vec_storage_ind = request.vec_storage_ind

    chat_history = to_langchain_templates(chat_history)
    
    context = get_relevant_information(user_input, chat_history, vec_storage_ind)
    retriever = vec_storages[vec_storage_ind].as_retriever()
    conversation_rag_chain = get_conversational_rag_chain(retriever)
    inv_response = conversation_rag_chain.invoke({
        "chat_history": chat_history,
        "context": context,
        "input": user_input
    })
    response = inv_response['answer']

    return {"response": response}



if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=os.getenv('BACK_PORT'))