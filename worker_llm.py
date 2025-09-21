# app/worker_llm.py
import os
from langchain.llms import HuggingFaceHub
from langchain_openai import OpenAI as OpenAILangChain # pip install langchain-openai


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
