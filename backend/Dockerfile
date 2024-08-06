# Use the official Python image from the Docker Hub
FROM python:3.10-slim

# Установка необходимых зависимостей
RUN apt-get update && apt-get install -y \
    locales && \
    locale-gen en_US.UTF-8

# Установка переменных окружения
ENV LANG=en_US.UTF-8 \
    LANGUAGE=en_US:en \
    LC_ALL=en_US.UTF-8

# Set the working directory in the container
WORKDIR /app

# Копируем файл requirements.txt в контейнер по пути /app
COPY requirements.txt .

RUN apt-get install -y \
    build-essential \
    gcc \
    g++ 

# Обновляем pip и устанавливаем зависимости
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Копируем все файлы в рабочую директорию
COPY . .

# Порт приложения из аргументов
ARG BACK_PORT
ENV BACK_PORT=${BACK_PORT}

# Expose the port that the app runs on
EXPOSE ${BACK_PORT}

# Command to run the FastAPI application
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port $BACK_PORT"]