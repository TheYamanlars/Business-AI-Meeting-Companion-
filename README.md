# Business-AI-Meeting-Companion-
Business AI Meeting Companion (STT + LLM + Gradio) 
# Business AI Meeting Companion (STT + LLM + Gradio)

A production-ready template for a **meeting companion** that:

* Transcribes audio (MP3/WAV/…) with **OpenAI Whisper** (via Hugging Face Transformers)
* Summarizes and extracts **key points** with an **LLM** (IBM watsonx, Hugging Face, or OpenAI — pluggable)
* Exposes a simple **Gradio** web UI for upload → transcript → summarized notes

---

## Repository Structure

```
business-ai-meeting-companion/
├─ app/
│  ├─ speech_analyzer.py           # Main app (Gradio UI + Whisper STT + LLM chain)
│  ├─ worker_llm.py                # (Optional) Helpers to select/config LLM providers
│  ├─ prompts/
│  │  └─ meeting_keypoints.txt     # Prompt template used by the LLM
│  └─ samples/
│     └─ demo_meeting.mp3          # (Optional) Example audio file (not tracked by default)
│
├─ docker/
│  └─ Dockerfile                   # Container image for the app
│
├─ .env.example                    # Example env vars for watsonx / HF / OpenAI
├─ requirements.txt                # Python dependencies (pip)
├─ README.md                       # This file (setup, run, usage)
├─ .gitignore                      # Standard Python & build ignores
└─ LICENSE                         # Your chosen license (e.g., MIT)
```

> **Tip:** You can split providers cleanly by keeping LLM-specific glue code in `worker_llm.py`. The main app imports only one provider at runtime via environment variables.

---

## What This App Does

1. **Speech-to-Text**: Splits audio into chunks and transcribes with **Whisper** (`openai/whisper-*` models).
2. **LLM Summarization**: Feeds the raw transcript into a prompt template ("list the key points with details") and calls your chosen LLM to produce a **clean, coherent summary** (often correcting minor STT errors).
3. **Web UI**: A minimal **Gradio** interface to upload audio and get structured output quickly.

Example output (simplified):

```
• Project deadlines and owners confirmed
• Budget reallocation agreed (Phase 2 → Phase 3)
• Risks: vendor delay; mitigation: parallel sourcing
```

---

## Prerequisites

* **Python** ≥ 3.9
* **ffmpeg** installed (Whisper needs it)

  * Linux: `sudo apt update && sudo apt install -y ffmpeg`
  * macOS: `brew install ffmpeg`
  * Windows (choco): `choco install ffmpeg`
* (Optional) **CUDA GPU** for faster transcription (install PyTorch with CUDA per [https://pytorch.org/get-started/](https://pytorch.org/get-started/))

---

## Quick Start (Local)

```bash
# 1) Clone
git clone https://github.com/<your-username>/business-ai-meeting-companion.git
cd business-ai-meeting-companion

# 2) Create a virtual environment
python3 -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 3) Install dependencies
pip install -U pip
pip install -r requirements.txt

# 4) (Optional) Configure env vars for your LLM provider
cp .env.example .env
# edit .env with your watsonx / Hugging Face / OpenAI keys

# 5) Run the app
python app/speech_analyzer.py
# Gradio will start at http://0.0.0.0:7860
```

Upload an audio file and get summarized bullet points.

---

## Docker Run

```bash
# From repo root
docker build -t meeting-companion:latest -f docker/Dockerfile .

# pass .env values into the container
docker run --rm -p 7860:7860 --env-file .env meeting-companion:latest
```

Open **[http://localhost:7860](http://localhost:7860)**.

---

## Configuration (Environment Variables)

Copy `.env.example` → `.env` and fill whichever provider you want to use:

```
# Choose one provider by setting LLM_PROVIDER to: watsonx | hf | openai
LLM_PROVIDER=watsonx

# --- IBM watsonx ---
WATSONX_URL=https://us-south.ml.cloud.ibm.com
WATSONX_PROJECT_ID=skills-network
# (If using API key / bearer flows, add them here per your IBM account setup)

# --- Hugging Face Inference or Hub (if using text-generation-inference or API) ---
HF_TOKEN=hf_xxx
HF_MODEL_ID=meta-llama/Meta-Llama-3-8B-Instruct

# --- OpenAI (if you choose to use it) ---
OPENAI_API_KEY=sk-xxx
OPENAI_MODEL=gpt-4o-mini

# Whisper model size (tiny, base, small, medium, large)
WHISPER_MODEL=openai/whisper-small
```

> **Notes**
>
> * Whisper models with `.en` suffix (e.g., `whisper-tiny.en`) are **English-only**; remove the suffix for multilingual.
> * Larger Whisper models are more accurate but slower.

---

## App Code (speech\_analyzer.py)

```python
# app/speech_analyzer.py
import os
import gradio as gr
from transformers import pipeline
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

# --- LLM selection helpers ---
from worker_llm import build_llm  # you can comment this out and inline a single provider if you prefer

# Prompt template (LLama-style tags work fine; remove if your model dislikes them)
PROMPT_TEMPLATE = """
<s><<SYS>>
List the key points with details from the context:
[INST] The context : {context} [/INST]
<</SYS>>
"""

# Build LLM instance from env (watsonx / hf / openai)
llm = build_llm()

pt = PromptTemplate(input_variables=["context"], template=PROMPT_TEMPLATE)
chain = LLMChain(llm=llm, prompt=pt)

# Whisper STT pipeline
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "openai/whisper-small")
stt = pipeline("automatic-speech-recognition", model=WHISPER_MODEL, chunk_length_s=30)


def transcript_and_summarize(audio_file: str) -> str:
    """Transcribe uploaded audio, then summarize key points with the selected LLM."""
    if not audio_file:
        return "No audio file received."

    # Transcribe
    result = stt(audio_file, batch_size=8)
    transcript_text = result.get("text", "").strip()
    if not transcript_text:
        return "Transcription failed or empty transcript."

    # Summarize with LLM
    summary = chain.run(transcript_text)
    return summary


# Gradio UI
audio_input = gr.Audio(sources="upload", type="filepath", label="Upload meeting audio (.mp3/.wav)")
output_text = gr.Textbox(lines=18, label="Key Points (LLM)")

iface = gr.Interface(
    fn=transcript_and_summarize,
    inputs=audio_input,
    outputs=output_text,
    title="Business AI Meeting Companion",
    description="Upload an audio file to transcribe with Whisper, then summarize with your chosen LLM.",
)

if __name__ == "__main__":
    # 0.0.0.0 allows Docker/remote access
    iface.launch(server_name="0.0.0.0", server_port=7860)
```

---

## LLM Provider Glue (worker\_llm.py)

```python
# app/worker_llm.py
import os
from langchain.llms import HuggingFaceHub
from langchain_openai import OpenAI as OpenAILangChain  # pip install langchain-openai

# IBM watsonx imports (install ibm_watson_machine_learning)
from ibm_watson_machine_learning.foundation_models import Model
from ibm_watson_machine_learning.foundation_models.extensions.langchain import WatsonxLLM
from ibm_watson_machine_learning.metanames import GenTextParamsMetaNames as GenParams


def build_llm():
    provider = os.getenv("LLM_PROVIDER", "watsonx").lower()

    if provider == "watsonx":
        # Minimal watsonx example
        url = os.getenv("WATSONX_URL", "https://us-south.ml.cloud.ibm.com")
        project_id = os.getenv("WATSONX_PROJECT_ID", "skills-network")
        model_id = os.getenv("WATSONX_MODEL_ID", "meta-llama/llama-3-2-11b-vision-instruct")

        params = {
            GenParams.MAX_NEW_TOKENS: int(os.getenv("WATSONX_MAX_NEW_TOKENS", 800)),
            GenParams.TEMPERATURE: float(os.getenv("WATSONX_TEMPERATURE", 0.1)),
        }
        credentials = {"url": url}
        wx_model = Model(model_id=model_id, credentials=credentials, params=params, project_id=project_id)
        return WatsonxLLM(wx_model)

    if provider == "hf":
        # Uses Hugging Face Hub text-generation inference or hosted endpoints
        hf_token = os.getenv("HF_TOKEN")
        hf_model_id = os.getenv("HF_MODEL_ID", "meta-llama/Meta-Llama-3-8B-Instruct")
        if not hf_token:
            raise RuntimeError("HF_TOKEN is required for Hugging Face LLM provider")
        return HuggingFaceHub(repo_id=hf_model_id, huggingfacehub_api_token=hf_token)

    if provider == "openai":
        # OpenAI via LangChain wrapper
        # Requires: export OPENAI_API_KEY=...
        model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        return OpenAILangChain(model=model, temperature=float(os.getenv("OPENAI_TEMPERATURE", 0.2)))

    raise ValueError(f"Unsupported LLM_PROVIDER: {provider}")
```

---

## Prompt Template (prompts/meeting\_keypoints.txt)

```text
<s><<SYS>>
List the key points with details from the context:
[INST] The context : {context} [/INST]
<</SYS>>
```

*(Optional) You can simplify to a plain prompt if your model doesn’t like system/inst tags.*

---

## Requirements (requirements.txt)

```txt
gradio==5.23.2
transformers==4.36.0
torch==2.1.1
langchain==0.0.343
huggingface-hub==0.28.1
ibm_watson_machine_learning==1.0.335
langchain-openai==0.1.7
python-dotenv==1.0.1
```

> Install **ffmpeg** separately (system package), e.g. `sudo apt install -y ffmpeg`.

---

## Dockerfile (docker/Dockerfile)

```dockerfile
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
```

---

## .env.example

```env
# Select one: watsonx | hf | openai
LLM_PROVIDER=watsonx

# IBM watsonx
WATSONX_URL=https://us-south.ml.cloud.ibm.com
WATSONX_PROJECT_ID=skills-network
WATSONX_MODEL_ID=meta-llama/llama-3-2-11b-vision-instruct
WATSONX_TEMPERATURE=0.1
WATSONX_MAX_NEW_TOKENS=800

# Hugging Face
HF_TOKEN=hf_xxx
HF_MODEL_ID=meta-llama/Meta-Llama-3-8B-Instruct

# OpenAI
OPENAI_API_KEY=sk-xxx
OPENAI_MODEL=gpt-4o-mini
OPENAI_TEMPERATURE=0.2

# Whisper
WHISPER_MODEL=openai/whisper-small
```

---

## .gitignore

```gitignore
# Python
__pycache__/
*.pyc
.venv/
.env

# OS
.DS_Store
Thumbs.db

# Data
app/samples/*.mp3
app/samples/*.wav
```

---

## Usage Notes & Pitfalls

* **Whisper model choice**: `tiny/base/small/medium/large` — bigger is more accurate but slower.
* **Language**: English-only models end with `.en`. For multilingual audio, choose models without `.en`.
* **Long audio**: Chunking is enabled via `chunk_length_s=30`. You can increase for fewer boundaries (at cost of memory).
* **GPU**: For speed, install PyTorch with CUDA and run on a GPU host.
* **watsonx auth**: This template assumes basic URL + project usage. Configure API-key/bearer per your IBM account if needed.
* **Rate limits**: External LLMs may be rate limited; add simple retry/backoff for production.

---

## Roadmap (Ideas)

* Add diarization (speaker labels) and timestamps
* Save full transcripts + summaries to Markdown/PDF
* Add VAD (voice activity detection) to trim silence
* Multi-user auth & persistent storage (e.g., SQLite/Postgres)
* Streaming UI (real-time partial transcripts)

---

## License

MIT (replace with your org’s preferred license).

---

## Credits

* Whisper ASR via Hugging Face Transformers
* Summarization via IBM watsonx / Hugging Face / OpenAI (pluggable)
* UI powered by Gradio
