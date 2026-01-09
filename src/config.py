
import os

API_KEY = os.environ.get('OPENAI_API_KEY')
# LLM config
llm_config = {
    "config_list": [{
        "model": "gpt-5-nano",
        "api_key": API_KEY
    }]
}