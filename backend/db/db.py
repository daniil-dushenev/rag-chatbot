from sqlalchemy import create_engine, Column, Integer, String, ForeignKey, Text, TIMESTAMP, func
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship, Session
from pydantic import BaseModel


Base = declarative_base()

class User(Base):
    __tablename__ = 'users'
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(50), nullable=False, unique=True, index=True)
    password_hash = Column(String(255), nullable=False)
    email = Column(String(100), nullable=False, unique=True, index=True)
    
    messages = relationship("Message", back_populates="owner")
    db_user = relationship("DbUser", back_populates="user")
    chats = relationship("Chat", back_populates="user")

class Message(Base):
    __tablename__ = 'messages'
    id = Column(Integer, primary_key=True, index=True)
    text = Column(Text, nullable=False)
    user_id = Column(Integer, ForeignKey('users.id'), nullable=False)
    to_user_id = Column(Integer, ForeignKey('users.id'), nullable=True)
    created_at = Column(TIMESTAMP, server_default=func.now())
    
    owner = relationship("User", back_populates="messages")

class Task(Base):
    __tablename__ = 'tasks'
    id = Column(Integer, primary_key=True, index=True)
    task_id = Column(String(50), nullable=False, unique=True, index=True)
    result = Column(Text, nullable=True)
    value = Column(Text, nullable=True)
    created_at = Column(TIMESTAMP, server_default=func.now())

class VecDb(Base):
    __tablename__ = 'vec_db'
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(100), nullable=False)
    source = Column(Text, nullable=True)
    is_public = Column(Boolean, default=False)
    
    db_users = relationship("DbUser", back_populates="vec_db")
    chats = relationship("Chat", back_populates="vec_db")

class DbUser(Base):
    __tablename__ = 'db_user'
    id = Column(Integer, primary_key=True, index=True)
    vec_db_id = Column(Integer, ForeignKey('vec_db.id'), nullable=False)
    user_id = Column(Integer, ForeignKey('users.id'), nullable=False)
    is_owner = Column(Boolean, default=True)

    vec_db = relationship("VecDb", back_populates="db_users")
    user = relationship("User", back_populates="db_user")

class Chat(Base):
    __tablename__ = 'chats'
    id = Column(Integer, primary_key=True, index=True)
    vec_db_id = Column(Integer, ForeignKey('vec_db.id'), nullable=False)
    user_id = Column(Integer, ForeignKey('users.id'), nullable=False)

    vec_db = relationship("VecDb", back_populates="chats")
    user = relationship("User", back_populates="chats")
    messages = relationship("MessageChat", back_populates="chat")

class MessageChat(Base):
    __tablename__ = 'message_chat'
    id = Column(Integer, primary_key=True, index=True)
    chat_id = Column(Integer, ForeignKey('chats.id'), nullable=False)
    message_id = Column(Integer, ForeignKey('messages.id'), nullable=False)

    chat = relationship("Chat", back_populates="messages")
    message = relationship("Message")

class UserCreate(BaseModel):
    username: str
    password_hash: str
    email: str

class MessageCreate(BaseModel):
    text: str
    user_id: int

# Token model
class Token(BaseModel):
    access_token: str
    token_type: str

# User registration model
class UserRegistration(BaseModel):
    username: str
    email: str
    password: str

# Pydantic models
class UserBase(BaseModel):
    username: str
    email: str

class UserCreate(UserBase):
    password_hash: str

class UserUpdate(UserBase):
    pass

class UserInDB(UserBase):
    id: int

    class Config:
        orm_mode = True

class MessageBase(BaseModel):
    text: str

class MessageCreate(MessageBase):
    user_id: int

class MessageUpdate(MessageBase):
    pass

class MessageInDB(MessageBase):
    id: int
    created_at: str

    class Config:
        orm_mode = True

class TaskBase(BaseModel):
    task_id: str
    result: str | None = None
    value: str | None = None

class TaskCreate(TaskBase):
    pass

class TaskInDB(TaskBase):
    id: int
    created_at: str

    class Config:
        orm_mode = True

class VecDbBase(BaseModel):
    name: str
    source: str | None = None
    is_public: bool

class VecDbCreate(VecDbBase):
    pass

class VecDbInDB(VecDbBase):
    id: int

    class Config:
        orm_mode = True

class DbUserBase(BaseModel):
    vec_db_id: int
    user_id: int
    is_owner: bool = True

class DbUserCreate(DbUserBase):
    pass

class DbUserInDB(DbUserBase):
    id: int

    class Config:
        orm_mode = True

class ChatBase(BaseModel):
    vec_db_id: int
    user_id: int

class ChatCreate(ChatBase):
    pass

class ChatInDB(ChatBase):
    id: int

    class Config:
        orm_mode = True

class MessageChatBase(BaseModel):
    chat_id: int
    message_id: int

class MessageChatCreate(MessageChatBase):
    pass

class MessageChatInDB(MessageChatBase):
    id: int

    class Config:
        orm_mode = True