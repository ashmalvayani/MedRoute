"""Text & Vision LLM backend for MAM.

Supports two modes, selected at runtime:

1. **Local vLLM engine** (default): the first call to `MedicalAssistant()`
   loads a vLLM `LLM` engine in this process. The instance is a singleton,
   so every pipeline file (role generation, type classification, meeting
   moderator, diagnosis, review, web-search summarization) shares the same
   engine — the model is loaded exactly once.

2. **Remote vLLM server** (API mode): if the env var `MAM_LLM_API_URL` is set
   (e.g. `http://localhost:8000/v1`), no local model is loaded. Every call is
   routed to the server's OpenAI-compatible `/chat/completions` endpoint. This
   is useful when you run `vllm serve <model>` in a separate process / GPU
   (e.g. so a vision model can share the same box, or the LLM can be served
   on a different machine).

Vision support:
    When `generate_response()` is called with `image_path=...`, the image is
    base64-encoded and sent as a multimodal content block in the user message.
    By default vision calls go to the same server as text calls.  Set
    `MAM_VLM_API_URL` to route them to a separate VLM server instead (e.g. a
    Qwen2.5-VL instance), keeping the text LLM for text-only steps.

Common env vars:
    MAM_LLM_MODEL             HF model id (default: sethuiyer/Medichat-Llama3-8B)

Local-mode env vars:
    MAM_LLM_GPU_MEM_UTIL      vLLM gpu_memory_utilization (default: 0.9)
    MAM_LLM_MAX_MODEL_LEN     vLLM max_model_len           (default: model native)
    MAM_LLM_DTYPE             vLLM dtype                   (default: auto)
    MAM_LLM_TENSOR_PARALLEL   tensor parallel size         (default: 1)
    CUDA_VISIBLE_DEVICES      which GPU(s) vLLM can see    (standard)

API-mode env vars:
    MAM_LLM_API_URL           Base URL including /v1, e.g. http://host:8000/v1
    MAM_LLM_API_KEY           Auth token (default: "EMPTY"; vllm serve accepts anything)
    MAM_LLM_API_TIMEOUT       Per-request timeout seconds (default: 600)

Vision-mode env vars:
    MAM_VLM_API_URL           Separate VLM server URL (default: same as MAM_LLM_API_URL)
    MAM_VLM_API_KEY           Auth token for VLM server (default: MAM_LLM_API_KEY)
    MAM_VLM_MODEL             VLM model id (default: auto-detect from server)
    MAM_VLM_EXTRA_BODY        JSON object merged into VLM requests (default: MAM_LLM_EXTRA_BODY)

Cross-mode:
    MAM_LLM_MAX_TOKENS        Lower bound on max_tokens for every call (floor).
                              Useful for thinking models (Qwen3 etc.) whose
                              default 512 would be spent inside <think>...</think>.
    MAM_LLM_EXTRA_BODY        JSON object merged into every chat-completions
                              request body (API mode only), e.g.
                              {"chat_template_kwargs": {"enable_thinking": false}}
"""

import base64
import json as _json
import mimetypes
import os
import threading
from contextlib import contextmanager

import requests

_instance = None
# Thread-local holder for per-example usage accumulation. When eval runs in
# a ThreadPoolExecutor, each worker opens its own scope so counters from
# different samples don't mix.
_tls = threading.local()


class MedicalAssistant:
    def __new__(cls, model_name=None, api_url=None):
        global _instance
        if _instance is not None:
            return _instance
        instance = super().__new__(cls)
        _instance = instance
        return instance

    def __init__(self, model_name=None, api_url=None):
        if getattr(self, "_initialized", False):
            return

        # model_name MAY be None in API mode: we'll probe /v1/models.
        self.model_name = model_name or os.environ.get("MAM_LLM_MODEL")
        self.api_url = api_url or os.environ.get("MAM_LLM_API_URL")
        self.sys_message = (
            "You are an AI Medical Assistant trained on a vast dataset of health "
            "information. Please be thorough and provide an informative answer. "
            "If you don't know the answer to a specific medical inquiry, advise "
            "seeking professional help."
        )

        # Optional floor on max_tokens for every call (for thinking models).
        raw_floor = os.environ.get("MAM_LLM_MAX_TOKENS")
        self.max_tokens_floor = int(raw_floor) if raw_floor else 0

        # Usage counters (process-lifetime; use snapshot() + diff for per-example).
        # Protected by a lock so concurrent workers can both bump safely.
        self._counter_lock = threading.Lock()
        self.calls = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0

        # Optional extra body merged into every chat-completions request.
        extra_raw = os.environ.get("MAM_LLM_EXTRA_BODY", "")
        try:
            self.extra_body = _json.loads(extra_raw) if extra_raw else {}
        except Exception as e:
            print(f"[MedicalAssistant] Warning: MAM_LLM_EXTRA_BODY is not valid JSON ({e}); ignoring.")
            self.extra_body = {}

        if self.api_url:
            self._init_api_mode()
        else:
            if not self.model_name:
                self.model_name = "sethuiyer/Medichat-Llama3-8B"
            self._init_local_mode()

        # ---- VLM endpoint (optional, for vision calls) ----
        self.vlm_api_url = os.environ.get("MAM_VLM_API_URL", "").rstrip("/") or None
        self.vlm_model_name = os.environ.get("MAM_VLM_MODEL")
        vlm_extra_raw = os.environ.get("MAM_VLM_EXTRA_BODY", "")
        try:
            self.vlm_extra_body = _json.loads(vlm_extra_raw) if vlm_extra_raw else None
        except Exception:
            self.vlm_extra_body = None

        if self.vlm_api_url:
            self._init_vlm_endpoint()
        else:
            # Single-model mode: vision calls go to the same server as text.
            self.vlm_api_url = self.api_url if self.mode == "api" else None
            self.vlm_model_name = self.vlm_model_name or self.model_name
            self.vlm_session = self.session if self.mode == "api" else None
            if self.vlm_extra_body is None:
                self.vlm_extra_body = self.extra_body

        # ---- Role generation endpoint (optional, use a larger model for specialist generation) ----
        self.role_gen_api_url = os.environ.get("MAM_ROLE_GEN_API_URL", "").rstrip("/") or None
        self.role_gen_model_name = os.environ.get("MAM_ROLE_GEN_MODEL")
        if self.role_gen_api_url:
            self._init_role_gen_endpoint()
        else:
            self.role_gen_session = None

        self._initialized = True

    # ------------------------------------------------------------------
    # Mode setup
    # ------------------------------------------------------------------
    def _init_api_mode(self):
        self.mode = "api"
        self.api_url = self.api_url.rstrip("/")
        self.api_key = os.environ.get("MAM_LLM_API_KEY", "EMPTY")
        self.api_timeout = float(os.environ.get("MAM_LLM_API_TIMEOUT", "600"))
        self.session = requests.Session()
        self.session.headers.update(
            {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            }
        )
        # Default HTTPAdapter pool_maxsize=10 throttles high-concurrency eval.
        # Bump it so --concurrency can scale well past 10 workers.
        pool = int(os.environ.get("MAM_LLM_API_POOL", "512"))
        from requests.adapters import HTTPAdapter
        adapter = HTTPAdapter(
            pool_connections=pool,
            pool_maxsize=pool,
            max_retries=0,
        )
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)

        # Probe the endpoint. If model_name isn't set, adopt whatever the
        # server actually serves. This guarantees we never send a request
        # with a model id that doesn't match (which yields a 404 in vLLM).
        served = []
        try:
            r = self.session.get(f"{self.api_url}/models", timeout=15)
            r.raise_for_status()
            served = [m.get("id") for m in r.json().get("data", []) if m.get("id")]
        except Exception as e:
            print(f"[MedicalAssistant] Warning: /models probe failed: {e}")

        if not self.model_name:
            if not served:
                raise RuntimeError(
                    f"MAM_LLM_MODEL is not set and the server at {self.api_url} "
                    f"did not return any model ids from /models."
                )
            self.model_name = served[0]
            print(f"[MedicalAssistant] Auto-detected served model: {self.model_name}")
        elif served and self.model_name not in served:
            print(
                f"[MedicalAssistant] Warning: MAM_LLM_MODEL={self.model_name!r} is not "
                f"in the server's advertised models {served}. Requests will likely 404."
            )

        print(
            f"[MedicalAssistant] API mode -> {self.api_url} (model={self.model_name})"
        )
        if self.extra_body:
            print(f"[MedicalAssistant] extra_body = {self.extra_body}")

    def _init_local_mode(self):
        self.mode = "local"
        # Lazy-import heavy deps so API-mode users don't need GPU / vllm.
        from transformers import AutoTokenizer
        from vllm import LLM

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        llm_kwargs = dict(
            model=self.model_name,
            dtype=os.environ.get("MAM_LLM_DTYPE", "auto"),
            trust_remote_code=True,
            gpu_memory_utilization=float(
                os.environ.get("MAM_LLM_GPU_MEM_UTIL", "0.9")
            ),
            tensor_parallel_size=int(
                os.environ.get("MAM_LLM_TENSOR_PARALLEL", "1")
            ),
        )
        max_model_len = os.environ.get("MAM_LLM_MAX_MODEL_LEN")
        if max_model_len:
            llm_kwargs["max_model_len"] = int(max_model_len)

        print(f"[MedicalAssistant] Loading local vLLM engine for {self.model_name} ...")
        self.llm = LLM(**llm_kwargs)
        print("[MedicalAssistant] vLLM engine ready.")

    def _init_vlm_endpoint(self):
        """Set up a separate session + model id for the VLM server."""
        vlm_key = os.environ.get("MAM_VLM_API_KEY",
                                 os.environ.get("MAM_LLM_API_KEY", "EMPTY"))
        self.vlm_session = requests.Session()
        self.vlm_session.headers.update({
            "Authorization": f"Bearer {vlm_key}",
            "Content-Type": "application/json",
        })
        pool = int(os.environ.get("MAM_LLM_API_POOL", "512"))
        from requests.adapters import HTTPAdapter
        adapter = HTTPAdapter(pool_connections=pool, pool_maxsize=pool, max_retries=0)
        self.vlm_session.mount("http://", adapter)
        self.vlm_session.mount("https://", adapter)

        # Probe to auto-detect model id.
        served = []
        try:
            r = self.vlm_session.get(f"{self.vlm_api_url}/models", timeout=15)
            r.raise_for_status()
            served = [m.get("id") for m in r.json().get("data", []) if m.get("id")]
        except Exception as e:
            print(f"[MedicalAssistant] Warning: VLM /models probe failed: {e}")

        if not self.vlm_model_name:
            if served:
                self.vlm_model_name = served[0]
            else:
                self.vlm_model_name = self.model_name  # fallback
        print(f"[MedicalAssistant] VLM endpoint -> {self.vlm_api_url} "
              f"(model={self.vlm_model_name})")

        if self.vlm_extra_body is None:
            self.vlm_extra_body = self.extra_body

    def _init_role_gen_endpoint(self):
        """Set up a separate session + model id for the role generation model."""
        rg_key = os.environ.get("MAM_ROLE_GEN_API_KEY",
                                os.environ.get("MAM_LLM_API_KEY", "EMPTY"))
        self.role_gen_session = requests.Session()
        self.role_gen_session.headers.update({
            "Authorization": f"Bearer {rg_key}",
            "Content-Type": "application/json",
        })
        pool = int(os.environ.get("MAM_LLM_API_POOL", "512"))
        from requests.adapters import HTTPAdapter
        adapter = HTTPAdapter(pool_connections=pool, pool_maxsize=pool, max_retries=0)
        self.role_gen_session.mount("http://", adapter)
        self.role_gen_session.mount("https://", adapter)

        # Probe to auto-detect model id.
        served = []
        try:
            r = self.role_gen_session.get(f"{self.role_gen_api_url}/models", timeout=15)
            r.raise_for_status()
            served = [m.get("id") for m in r.json().get("data", []) if m.get("id")]
        except Exception as e:
            print(f"[MedicalAssistant] Warning: Role-gen /models probe failed: {e}")

        if not self.role_gen_model_name:
            if served:
                self.role_gen_model_name = served[0]
            else:
                self.role_gen_model_name = self.model_name
        print(f"[MedicalAssistant] Role-gen endpoint -> {self.role_gen_api_url} "
              f"(model={self.role_gen_model_name})")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def generate_response(self, question, sys_message=None, max_new_tokens=512,
                          image_path=None):
        """Generate a response. If image_path is provided, the image is sent
        as a base64 content block and the request is routed to the VLM endpoint."""
        if image_path:
            return self._chat_vlm(question, image_path, sys_message, max_new_tokens)
        messages = self._build_messages(question, sys_message)
        if self.mode == "api":
            return self._chat_api(messages, max_new_tokens)
        return self._chat_local([messages], max_new_tokens)[0]

    def generate_response_role_gen(self, question, sys_message=None, max_new_tokens=512):
        """Generate a response using the role-generation model (larger model).
        Falls back to the default model if MAM_ROLE_GEN_API_URL is not set."""
        if not self.role_gen_session or not self.role_gen_api_url:
            return self.generate_response(question, sys_message, max_new_tokens)
        messages = self._build_messages(question, sys_message)
        payload = {
            "model": self.role_gen_model_name,
            "messages": messages,
            "temperature": 0.0,
            "max_tokens": self._effective_max_tokens(max_new_tokens),
        }
        if self.extra_body:
            payload.update(self.extra_body)
        r = self.role_gen_session.post(
            f"{self.role_gen_api_url}/chat/completions",
            json=payload,
            timeout=self.api_timeout,
        )
        r.raise_for_status()
        data = r.json()
        usage = data.get("usage") or {}
        self._bump(
            calls=1,
            ptoks=int(usage.get("prompt_tokens") or 0),
            ctoks=int(usage.get("completion_tokens") or 0),
        )
        import re as _re
        content = data["choices"][0]["message"]["content"].strip()
        # Normalize trailing whitespace before newlines so downstream parsers
        # see ":\n" instead of ":  \n" (Qwen3-32B adds trailing spaces).
        content = _re.sub(r"[ \t]+\n", "\n", content)
        return content

    def generate_batch(self, questions, sys_message=None, max_new_tokens=512):
        """Batched generation — preferred when you have many prompts.

        In API mode we still issue requests one at a time (the server batches
        internally); in local mode we pass the whole list to vLLM at once.
        """
        batch_messages = [self._build_messages(q, sys_message) for q in questions]
        if self.mode == "api":
            return [self._chat_api(m, max_new_tokens) for m in batch_messages]
        return self._chat_local(batch_messages, max_new_tokens)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    @staticmethod
    def _encode_image(path):
        """Read an image file and return a data-URI string."""
        mime, _ = mimetypes.guess_type(path)
        if not mime:
            mime = "image/png" if path.lower().endswith(".png") else "image/jpeg"
        with open(path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("ascii")
        return f"data:{mime};base64,{b64}"

    def _build_messages(self, question, sys_message=None):
        return [
            {"role": "system", "content": sys_message or self.sys_message},
            {"role": "user", "content": question},
        ]

    def _build_vision_messages(self, question, image_path, sys_message=None):
        """Build messages with an image content block for multimodal input."""
        image_url = self._encode_image(image_path)
        return [
            {"role": "system", "content": sys_message or self.sys_message},
            {"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": image_url}},
                {"type": "text", "text": question},
            ]},
        ]

    def _effective_max_tokens(self, requested):
        if self.max_tokens_floor:
            return max(requested, self.max_tokens_floor)
        return requested

    def _chat_api(self, messages, max_new_tokens):
        payload = {
            "model": self.model_name,
            "messages": messages,
            "temperature": 0.0,
            "max_tokens": self._effective_max_tokens(max_new_tokens),
        }
        if self.extra_body:
            payload.update(self.extra_body)
        r = self.session.post(
            f"{self.api_url}/chat/completions",
            json=payload,
            timeout=self.api_timeout,
        )
        r.raise_for_status()
        data = r.json()
        usage = data.get("usage") or {}
        self._bump(
            calls=1,
            ptoks=int(usage.get("prompt_tokens") or 0),
            ctoks=int(usage.get("completion_tokens") or 0),
        )
        return data["choices"][0]["message"]["content"].strip()

    def _chat_vlm(self, question, image_path, sys_message, max_new_tokens):
        """Send a multimodal request to the VLM endpoint."""
        if not self.vlm_api_url or not self.vlm_session:
            raise RuntimeError(
                "Vision call requested but no VLM endpoint is configured. "
                "Set MAM_LLM_API_URL (single-model) or MAM_VLM_API_URL (separate VLM)."
            )
        messages = self._build_vision_messages(question, image_path, sys_message)
        payload = {
            "model": self.vlm_model_name,
            "messages": messages,
            "temperature": 0.0,
            "max_tokens": self._effective_max_tokens(max_new_tokens),
        }
        if self.vlm_extra_body:
            payload.update(self.vlm_extra_body)
        r = self.vlm_session.post(
            f"{self.vlm_api_url}/chat/completions",
            json=payload,
            timeout=self.api_timeout,
        )
        r.raise_for_status()
        data = r.json()
        usage = data.get("usage") or {}
        self._bump(
            calls=1,
            ptoks=int(usage.get("prompt_tokens") or 0),
            ctoks=int(usage.get("completion_tokens") or 0),
        )
        return data["choices"][0]["message"]["content"].strip()

    def _chat_local(self, batch_messages, max_new_tokens):
        from vllm import SamplingParams

        prompts = [
            self.tokenizer.apply_chat_template(
                m, tokenize=False, add_generation_prompt=True
            )
            for m in batch_messages
        ]
        params = SamplingParams(
            temperature=0.0,
            max_tokens=self._effective_max_tokens(max_new_tokens),
        )
        outputs = self.llm.generate(prompts, params, use_tqdm=False)
        for o in outputs:
            self._bump(
                calls=1,
                ptoks=len(getattr(o, "prompt_token_ids", []) or []),
                ctoks=len(o.outputs[0].token_ids or []),
            )
        return [o.outputs[0].text.strip() for o in outputs]

    # ------------------------------------------------------------------
    # Usage accounting
    # ------------------------------------------------------------------
    def _bump(self, calls=1, ptoks=0, ctoks=0):
        with self._counter_lock:
            self.calls += calls
            self.prompt_tokens += ptoks
            self.completion_tokens += ctoks
        scope = getattr(_tls, "scope", None)
        if scope is not None:
            # Thread-local: no lock needed because each thread has its own scope dict.
            scope["calls"] += calls
            scope["prompt_tokens"] += ptoks
            scope["completion_tokens"] += ctoks

    def snapshot(self):
        """Return a snapshot of cumulative counters. Diff two snapshots to get
        per-example usage (only reliable single-threaded; prefer `scope()`
        when running with concurrency > 1)."""
        with self._counter_lock:
            return {
                "calls": self.calls,
                "prompt_tokens": self.prompt_tokens,
                "completion_tokens": self.completion_tokens,
            }

    @staticmethod
    def diff_snapshots(before, after):
        return {k: after[k] - before[k] for k in before}

    @contextmanager
    def scope(self):
        """Per-thread usage accounting. Use under ThreadPoolExecutor:

            with assistant.scope() as usage:
                run_pipeline(...)
            # usage == {'calls': N, 'prompt_tokens': P, 'completion_tokens': C}
        """
        s = {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0}
        prev = getattr(_tls, "scope", None)
        _tls.scope = s
        try:
            yield s
        finally:
            _tls.scope = prev


if __name__ == "__main__":
    assistant = MedicalAssistant()
    q = "Symptoms: Dizziness, headache, and nausea. What is the differential diagnosis?"
    print(assistant.generate_response(q))
