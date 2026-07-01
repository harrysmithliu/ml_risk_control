FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY pyproject.toml LICENSE README.md ./
COPY src ./src
COPY scripts ./scripts
COPY configs ./configs
COPY sql ./sql
COPY docs ./docs
COPY streamlit_app.py ./
COPY .env.example ./

RUN python -m pip install --upgrade pip && \
    python -m pip install -e ".[dev]"

EXPOSE 8501

CMD ["python", "-m", "streamlit", "run", "streamlit_app.py", "--server.address=0.0.0.0", "--server.port=8501"]
