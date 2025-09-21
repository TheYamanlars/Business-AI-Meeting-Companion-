FROM python:3.11-slim


# System deps (ffmpeg for Whisper)
RUN apt-get update && apt-get install -y --no-install-recommends \
ffmpeg \
&& rm -rf /var/lib/apt/lists/*


WORKDIR /app
COPY requirements.txt ./
RUN pip install -U pip && pip install -r requirements.txt


COPY app ./app
COPY .env.example ./


ENV PYTHONUNBUFFERED=1 \
WHISPER_MODEL=openai/whisper-small \
LLM_PROVIDER=watsonx


EXPOSE 7860
CMD ["python", "app/speech_analyzer.py"]
