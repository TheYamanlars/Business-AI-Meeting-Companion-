import os
import gradio as gr
from transformers import pipeline
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain


# --- LLM selection helpers ---
from worker_llm import build_llm # you can comment this out and inline a single provider if you prefer


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
