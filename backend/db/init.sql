-- Создаем таблицу users, если она не существует
CREATE TABLE IF NOT EXISTS users (
    id SERIAL PRIMARY KEY,
    username VARCHAR(50) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    email VARCHAR(100) UNIQUE NOT NULL
);


-- Создаем таблицу messages, если она не существует
CREATE TABLE IF NOT EXISTS messages (
    id SERIAL PRIMARY KEY,
    text TEXT NOT NULL,
    user_id INTEGER REFERENCES users(id),
    to_user_id INTEGER REFERENCES users(id),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_messages_user_id ON messages (user_id);

-- Удаление автоинкрементного свойства
ALTER TABLE users ALTER COLUMN id DROP DEFAULT;

-- Вставка записи с id = 0
INSERT INTO users (id, username, password_hash, email) VALUES (0, 'assistant', 'pass', 'email')
ON CONFLICT (id) DO NOTHING;


CREATE TABLE IF NOT EXISTS tasks (
    id SERIAL PRIMARY KEY,
    task_id VARCHAR(50) UNIQUE NOT NULL,
    result TEXT,
    value TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Восстановление автоинкрементного свойства
ALTER TABLE users ALTER COLUMN id SET DEFAULT nextval('users_id_seq');


-- Создаем таблицу vec_db, если она не существует
CREATE TABLE IF NOT EXISTS vec_db (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    source TEXT,
    is_public BOOLEAN DEFAULT FALSE
);

-- Создаем таблицу db_user, если она не существует
CREATE TABLE IF NOT EXISTS db_user (
    id SERIAL PRIMARY KEY,
    vec_db_id INTEGER REFERENCES vec_db(id),
    user_id INTEGER REFERENCES users(id),
    is_owner BOOLEAN DEFAULT TRUE
);

-- Создаем таблицу chats, если она не существует
CREATE TABLE IF NOT EXISTS chats (
    id SERIAL PRIMARY KEY,
    vec_db_id INTEGER REFERENCES vec_db(id),
    user_id INTEGER REFERENCES users(id),
    name VARCHAR(100)
);

-- Создаем таблицу message_chat, если она не существует
CREATE TABLE IF NOT EXISTS message_chat (
    id SERIAL PRIMARY KEY,
    chat_id INTEGER REFERENCES chats(id),
    message_id INTEGER REFERENCES messages(id)
);

-- Создаем индексы для оптимизации запросов
CREATE INDEX idx_db_user_user_id ON db_user (user_id);
CREATE INDEX idx_chats_user_id ON chats (user_id);
CREATE INDEX idx_message_chat_chat_id ON message_chat (chat_id);
CREATE INDEX idx_message_chat_message_id ON message_chat (message_id);