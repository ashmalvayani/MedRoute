# import aiohttp
# from typing import List, Union, Optional
# import traceback
# from tenacity import retry, wait_random_exponential, stop_after_attempt, retry_if_exception_type
# from typing import Dict, Any
# from dotenv import load_dotenv
# import os
# import json
# import asyncio

# from openai import AzureOpenAI  # NEW

# from GDesigner.llm.format import Message
# from GDesigner.llm.price import cost_count
# from GDesigner.llm.llm import LLM
# from GDesigner.llm.llm_registry import LLMRegistry


# # IMPORTANT: load .env before reading env vars
# # load_dotenv()

# # Azure OpenAI env
# AZURE_ENDPOINT = os.getenv("ENDPOINT_URL", "").rstrip("/")  # e.g. https://gaea-testing.openai.azure.com
# AZURE_DEPLOYMENT = os.getenv("DEPLOYMENT_NAME", "")         # e.g. gpt-4.1-mini (deployment name)
# AZURE_API_KEY = os.getenv("AZURE_OPENAI_API_KEY", "")
# AZURE_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2024-08-01-preview")

# # Create Azure client once
# _azure_client: Optional[AzureOpenAI] = None
# if AZURE_ENDPOINT and AZURE_API_KEY:
#     _azure_client = AzureOpenAI(
#         azure_endpoint=AZURE_ENDPOINT,
#         api_key=AZURE_API_KEY,
#         api_version=AZURE_API_VERSION,
#     )


# class APIError(Exception):
#     """Exception raised for errors in the API request."""
#     def __init__(self, status, message, response_text=None):
#         self.status = status
#         self.message = message
#         self.response_text = response_text
#         super().__init__(f"API Error: {status} - {message}")


# def _azure_chat_sync(
#     deployment: str,
#     formatted_messages: List[Dict[str, str]],
#     temperature: float,
#     max_tokens: Optional[int],
# ) -> str:
#     if _azure_client is None:
#         raise ValueError(
#             "Azure OpenAI client not configured. Set ENDPOINT_URL and AZURE_OPENAI_API_KEY (and DEPLOYMENT_NAME)."
#         )

#     # Azure uses deployment name in `model=...`
#     resp = _azure_client.chat.completions.create(
#         model=deployment,
#         messages=formatted_messages,
#         temperature=temperature,
#         max_tokens=max_tokens,
#     )
#     return resp.choices[0].message.content

# def _flatten_prompt_for_cost(messages):
#     parts = []
#     for mm in messages:
#         c = mm.get("content")
#         if isinstance(c, str):
#             parts.append(c)
#         elif isinstance(c, list):
#             # keep only text parts for cost/debug strings
#             for chunk in c:
#                 if chunk.get("type") == "text":
#                     parts.append(chunk.get("text", ""))
#     return "".join(parts)

# def _content_to_text(content: Any) -> str:
#     if isinstance(content, str):
#         return content
#     if isinstance(content, list):
#         # multimodal content: only count text parts
#         out = []
#         for part in content:
#             if isinstance(part, dict) and part.get("type") == "text":
#                 out.append(part.get("text", ""))
#         return "".join(out)
#     return ""
    
# @retry(
#     wait=wait_random_exponential(min=1, max=60),
#     stop=stop_after_attempt(5),
#     retry=retry_if_exception_type((aiohttp.ClientError, APIError, json.JSONDecodeError))
# )
# async def achat(
#     model: str,
#     msg: List[Dict],
#     temperature: float = 0.0,
#     max_tokens: Optional[int] = None,
# ):
#     # Keep backward compatibility: if caller passes "gpt-4o" etc,
#     # but you set DEPLOYMENT_NAME, we use the deployment.
#     deployment = AZURE_DEPLOYMENT or model

#     # Format messages in the way the client expects
#     formatted_messages: List[Dict[str, Any]] = []
#     for m in msg:
#         if isinstance(m, dict) and "role" in m and "content" in m:
#             formatted_messages.append({"role": m["role"], "content": m["content"]})
#         elif isinstance(m, Message):
#             formatted_messages.append({"role": m.role, "content": m.content})

#     # formatted_messages: List[Dict[str, str]] = []
#     # for m in msg:
#     #     if isinstance(m, dict) and "role" in m and "content" in m:
#     #         formatted_messages.append({"role": m["role"], "content": m["content"]})
#     #     elif isinstance(m, Message):
#     #         formatted_messages.append({"role": m.role, "content": m.content})

#     try:
#         # Run sync Azure call in a thread so this stays async-friendly
#         response_text = await asyncio.to_thread(
#             _azure_chat_sync,
#             deployment,
#             formatted_messages,
#             temperature,
#             max_tokens,
#         )

#         # prompt = "".join([m["content"] if isinstance(m, dict) else m.content for m in msg])
#         prompt = "".join(_content_to_text(m["content"]) if isinstance(m, dict) else _content_to_text(m.content) for m in msg)
#         cost_count(prompt, response_text, deployment)
#         return response_text

#     except Exception as e:
#         print(f"Unexpected error during Azure API request: {str(e)}")
#         traceback.print_exc()
#         raise


# @LLMRegistry.register("GPTChat")
# class GPTChat(LLM):

#     def __init__(self, model_name: str):
#         # You can keep passing --llm_name gpt-4o
#         # If DEPLOYMENT_NAME is set, achat() will use it anyway.
#         self.model_name = model_name

#     async def agen(
#         self,
#         messages: List[Message],
#         max_tokens: Optional[int] = None,
#         temperature: Optional[float] = None,
#         num_comps: Optional[int] = None,
#     ) -> Union[List[str], str]:

#         if max_tokens is None:
#             max_tokens = self.DEFAULT_MAX_TOKENS
#         if temperature is None:
#             temperature = self.DEFAULT_TEMPERATURE
#         if num_comps is None:
#             num_comps = self.DEFUALT_NUM_COMPLETIONS

#         if isinstance(messages, str):
#             messages = [Message(role="user", content=messages)]

#         try:
#             return await achat(self.model_name, messages, temperature=temperature, max_tokens=max_tokens)
#         except Exception as e:
#             print(f"Error in agen method: {str(e)}")
#             traceback.print_exc()
#             raise

#     def gen(
#         self,
#         messages: List[Message],
#         max_tokens: Optional[int] = None,
#         temperature: Optional[float] = None,
#         num_comps: Optional[int] = None,
#     ) -> Union[List[str], str]:
#         pass



import aiohttp
from typing import List, Union, Optional, Dict, Any
import traceback
from tenacity import retry, wait_random_exponential, stop_after_attempt, retry_if_exception_type
from dotenv import load_dotenv
import os
import json
import asyncio

from openai import AzureOpenAI

from GDesigner.llm.format import Message
from GDesigner.llm.price import cost_count
from GDesigner.llm.llm import LLM
from GDesigner.llm.llm_registry import LLMRegistry


# ------------------------------------------------------------
# Env loading
# ------------------------------------------------------------
# Auto-find .env in the AnyMAC-Improved root (3 levels up from this file: llm/ -> GDesigner/ -> AnyMAC-Improved/)
_dotenv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..', '.env')
load_dotenv(dotenv_path=_dotenv_path)


# ------------------------------------------------------------
# Azure OpenAI config
# ------------------------------------------------------------
AZURE_ENDPOINT = os.getenv("ENDPOINT_URL", "").rstrip("/")  # e.g. https://gaea-testing.openai.azure.com
AZURE_DEPLOYMENT = os.getenv("DEPLOYMENT_NAME", "")         # e.g. your deployment name
AZURE_API_KEY = os.getenv("AZURE_OPENAI_API_KEY", "")
AZURE_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2024-08-01-preview")

_azure_client: Optional[AzureOpenAI] = None
if AZURE_ENDPOINT and AZURE_API_KEY:
    _azure_client = AzureOpenAI(
        azure_endpoint=AZURE_ENDPOINT,
        api_key=AZURE_API_KEY,
        api_version=AZURE_API_VERSION,
    )


class APIError(Exception):
    """Exception raised for errors in the API request."""
    def __init__(self, status, message, response_text=None):
        self.status = status
        self.message = message
        self.response_text = response_text
        super().__init__(f"API Error: {status} - {message}")


def _content_to_text(content: Any) -> str:
    """
    Convert message content to plain text for cost/debug strings.
    Supports:
      - str
      - multimodal list: [{"type":"text","text":...}, {"type":"image_url",...}]
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        out: List[str] = []
        for part in content:
            if isinstance(part, dict) and part.get("type") == "text":
                out.append(part.get("text", ""))
        return "".join(out)
    return ""


def _azure_chat_sync(
    deployment: str,
    formatted_messages: List[Dict[str, Any]],
    temperature: float,
    max_tokens: Optional[int],
) -> str:
    if _azure_client is None:
        raise ValueError(
            "Azure OpenAI client not configured. Set ENDPOINT_URL, AZURE_OPENAI_API_KEY, and DEPLOYMENT_NAME."
        )

    resp = _azure_client.chat.completions.create(
        model=deployment,              # Azure uses deployment name here
        messages=formatted_messages,   # content may be str OR multimodal list
        temperature=temperature,
        max_tokens=max_tokens,
    )

    content = resp.choices[0].message.content
    if content is None:
        return ""
    return content


@retry(
    wait=wait_random_exponential(min=1, max=60),
    stop=stop_after_attempt(5),
    retry=retry_if_exception_type((aiohttp.ClientError, APIError, json.JSONDecodeError, TimeoutError)),
)
async def achat(
    model: str,
    msg: List[Dict],
    temperature: float = 0.0,
    max_tokens: Optional[int] = None,
) -> str:
    # If DEPLOYMENT_NAME is set, use it; otherwise fall back to `model`
    deployment = AZURE_DEPLOYMENT or model

    # Normalize messages to OpenAI format
    formatted_messages: List[Dict[str, Any]] = []
    for m in msg:
        if isinstance(m, dict) and "role" in m and "content" in m:
            formatted_messages.append({"role": m["role"], "content": m["content"]})
        elif isinstance(m, Message):
            formatted_messages.append({"role": m.role, "content": m.content})

    try:
        # Run sync Azure call in a worker thread to keep this async-friendly
        response_text = await asyncio.to_thread(
            _azure_chat_sync,
            deployment,
            formatted_messages,
            temperature,
            max_tokens,
        )

        # Cost accounting (never crash training if model not in price table)
        try:
            prompt_text = "".join(_content_to_text(m["content"]) for m in formatted_messages)
            cost_count(prompt_text, response_text, deployment)
        except Exception:
            pass

        return response_text

    except Exception as e:
        print(f"Unexpected error during Azure API request: {str(e)}")
        traceback.print_exc()
        raise


@LLMRegistry.register("GPTChat")
class GPTChat(LLM):
    def __init__(self, model_name: str):
        # You can keep passing --llm_name gpt-4o (or anything).
        # If DEPLOYMENT_NAME is set, achat() will use it anyway.
        self.model_name = model_name

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

        try:
            return await achat(
                self.model_name,
                messages,                   # can be List[Message] OR List[dict], achat handles both
                temperature=temperature,
                max_tokens=max_tokens,
            )
        except Exception as e:
            print(f"Error in agen method: {str(e)}")
            traceback.print_exc()
            raise

    def gen(
        self,
        messages: List[Message],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        num_comps: Optional[int] = None,
    ) -> Union[List[str], str]:
        # Not used in your async pipeline currently
        raise NotImplementedError("Use agen() for async generation.")
