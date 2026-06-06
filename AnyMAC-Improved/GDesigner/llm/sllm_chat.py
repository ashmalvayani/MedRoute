"""
OpenAI-compatible LLM client.
Works with any backend that serves /v1/chat/completions:
  - vLLM   (recommended)
  - OpenAI / Azure OpenAI
  - ollama (via its /v1/ compatibility layer)

Configure via environment variables:
  BASE_URL  — e.g. http://localhost:8000  (vLLM default)
  API_KEY   — required for OpenAI; ignored by vLLM/ollama
"""

import aiohttp
from typing import List, Union, Optional, Dict
from tenacity import retry, wait_random_exponential, stop_after_attempt
from dotenv import load_dotenv
import os

from GDesigner.llm.format import Message
from GDesigner.llm.price import cost_count
from GDesigner.llm.llm import LLM
from GDesigner.llm.llm_registry import LLMRegistry

load_dotenv()

BASE_URL = os.getenv('BASE_URL', 'http://localhost:8000')
API_KEY = os.getenv('API_KEY', 'EMPTY')  # vLLM accepts any key; set real key for OpenAI


JUDGE_BASE_URL = os.getenv('JUDGE_BASE_URL', BASE_URL)

@retry(wait=wait_random_exponential(max=100), stop=stop_after_attempt(3))
async def achat(model: str, msg: List[Dict], max_tokens: int = 2048, temperature: float = 0.0, base_url: str = None):
    url = f"{base_url or BASE_URL}/v1/chat/completions"
    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {API_KEY}',
    }
    data = {
        "model": model,
        "messages": msg,
        "temperature": temperature,
        "top_p": 1.0,
        "max_tokens": max_tokens,
        "chat_template_kwargs": {"enable_thinking": False},
    }

    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=300)) as session:
        async with session.post(url, headers=headers, json=data) as response:
            if response.status != 200:
                error_text = await response.text()
                raise Exception(f"API request failed with status {response.status}: {error_text}")

            response_data = await response.json()
            completion = response_data['choices'][0]['message']['content']
            prompt_str = "\n".join([f"{m['role']}: {m['content']}" for m in msg])
            cost_count(prompt_str, completion, model)
            return completion


@LLMRegistry.register('SLLMChat')
class SLLMChat(LLM):

    def __init__(self, model_name: str, base_url: str = None):
        self.model_name = model_name
        self.base_url = base_url

    async def agen(
        self,
        messages: List[Message],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        num_comps: Optional[int] = None,
    ) -> Union[List[str], str]:

        if max_tokens is None:
            max_tokens = self.DEFAULT_MAX_TOKENS
        if temperature is None:
            temperature = self.DEFAULT_TEMPERATURE
        if num_comps is None:
            num_comps = self.DEFUALT_NUM_COMPLETIONS

        if isinstance(messages, str):
            messages = [Message(role="user", content=messages)]
        from tenacity import RetryError

        try:
            completion = await achat(self.model_name, messages, max_tokens=max_tokens, temperature=temperature, base_url=self.base_url)
        except RetryError as e:
            print("RetryError happened!")
            print("Last attempt exception:", e.last_attempt.exception())
            completion = "Error: Failed to get response after multiple attempts."

        return completion

    def gen(
        self,
        messages: List[Message],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        num_comps: Optional[int] = None,
    ) -> Union[List[str], str]:
        pass
