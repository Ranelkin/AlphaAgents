import os
from langchain.chat_models import init_chat_model
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')

llm = init_chat_model("openai:gpt-5-nano-2025-08-07")

