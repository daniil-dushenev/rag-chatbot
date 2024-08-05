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

class Message(Base):
    __tablename__ = 'messages'
    id = Column(Integer, primary_key=True, index=True)
    text = Column(Text, nullable=False)
    user_id = Column(Integer, ForeignKey('users.id'), nullable=False)
    
    owner = relationship("User", back_populates="messages")

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