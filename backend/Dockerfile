# Use the official Python image from the Docker Hub
FROM python:3.10-slim

# Установка необходимых зависимостей
RUN apt-get update && apt-get install -y \
    locales \
    && echo "en_US.UTF-8 UTF-8" > /etc/locale.gen \
    && locale-gen en_US.UTF-8 \
    && update-locale LANG=en_US.UTF-8


ENV LANG en_US.UTF-8
ENV LANGUAGE en_US:en
ENV LC_ALL en_US.UTF-8


# Set the working directory in the container
WORKDIR /app

# Копируем файл requirements.txt в контейнер по пути /app
COPY requirements.txt .

RUN apt-get install -y \
    build-essential \
    gcc \
    g++ 


# Устанавливаем зависимости, указанные в requirements.txt
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY . .

ARG BACK_PORT
ENV BACK_PORT=${BACK_PORT}

# Expose the port that the app runs on
EXPOSE ${BACK_PORT}

# Command to run the FastAPI application
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port $BACK_PORT"]