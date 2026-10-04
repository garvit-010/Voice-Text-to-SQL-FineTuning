"""Choosing which model the API serves.

Phase 12.

Three backends, selected by the ``MODEL_BACKEND`` environment variable:

| backend | what it serves | where it runs |
| --- | --- | --- |
| ``hf_space`` | fine-tuned QLoRA adapter via HF Space (ZeroGPU) | free ZeroGPU on HF |
| ``hf`` | base Qwen3-8B via HF Inference Providers | anywhere, needs `HF_TOKEN` |
| ``local`` | base + the Phase 9 LoRA adapter, 4-bit | a CUDA host |
| ``stub`` | canned SQL, no model at all | tests and CI |

**The default is ``hf``, which serves the base model, not the fine-tuned one.**
That is a hardware constraint, not a design preference: this project's
development machine has 8 GB of RAM and no GPU, and the adapter needs one.
Serving the adapter in production means running ``local`` on a GPU host, or
pushing the adapter to a vLLM server with LoRA enabled.

``PROMPT_VERSION`` selects which prompt is rendered, independently of the
backend. **The default is ``v2``**: it adds a business glossary and a
``DATA_AS_OF`` reference date, costs nothing, needs no GPU, and is worth
**+9.47 pp to the base model** (34.22 % -> 43.71 % strict when both are scored
on benchmark v2). ``PROMPT_VERSION=v1`` restores the frozen baseline prompt,
which is what the published v1 numbers were measured against.

A prompt and a set of weights belong together: the v2 adapter was trained
against prompt v2, and rendering v1 for it would be measuring something else.
``/health`` therefore reports the fingerprint of the prompt actually in use
rather than a hard-coded constant.

``stub`` exists so the pipeline, the validation layer, the repair loop and the
HTTP contract can all be tested without a GPU, a network call, or an API
credit. Every test in ``tests/test_api.py`` uses it.
"""

from __future__ import annotations

import os
import time
from typing import Any, Callable

from src.model.base_model import (
    DEFAULT_MODEL_ID,
    GenerationResult,
    InferenceParams,
    TextToSQLModel,
    extract_sql,
)

DEFAULT_BACKEND = "hf"
DEFAULT_PROMPT_VERSION = "v2"


def resolve_prompt(version: str | None = None):
    """The prompt module this service renders with.

    Selected by ``PROMPT_VERSION`` so a deployment can be moved between prompts
    without a code change, and so ``/health`` can report what is really served.
    """
    name = (version or os.getenv("PROMPT_VERSION", DEFAULT_PROMPT_VERSION)
            ).strip().lower()
    if name == "v1":
        from src.model import prompt as module
    elif name == "v2":
        from src.model import prompt_v2 as module
    else:
        raise RuntimeError(
            f"unknown PROMPT_VERSION {name!r}; expected 'v1' or 'v2'")
    return module


class StubModel(TextToSQLModel):
    """Returns canned SQL. No network, no weights, no GPU.

    ``responses`` maps a question to the SQL to return; ``default`` covers
    anything unmapped. ``repair_responses`` lets a test drive the repair path
    deterministically: the first call fails, the repair succeeds.
    """

    model_id = "stub/canned-sql"

    def __init__(
        self,
        responses: dict[str, str] | None = None,
        default: str = "SELECT 1",
        repair_responses: dict[str, str] | None = None,
    ) -> None:
        self._responses = responses or {}
        self._default = default
        self._repair = repair_responses or {}
        self.calls: list[str] = []
        self.repair_calls: list[list[dict[str, str]]] = []
        self.prompt = resolve_prompt()

    def generate(self, question: str, schema: str) -> GenerationResult:
        self.calls.append(question)
        sql = self._responses.get(question, self._default)
        return GenerationResult(sql=sql, raw_output=sql, latency_ms=1.0, ok=True,
                                finish_reason="stop", provider="stub")

    def generate_messages(self, messages: list[dict[str, str]]) -> GenerationResult:
        self.repair_calls.append(messages)
        content = messages[-1]["content"]
        for needle, sql in self._repair.items():
            if needle in content:
                return GenerationResult(sql=sql, raw_output=sql, latency_ms=1.0,
                                        ok=True, provider="stub")
        return GenerationResult(sql=self._default, raw_output=self._default,
                                latency_ms=1.0, ok=True, provider="stub")

    def describe(self) -> dict[str, Any]:
        return {"kind": "stub", "model_id": self.model_id, "fine_tuned": False,
                "adapters": []}


class LocalAdapterModel(TextToSQLModel):
    """Base Qwen3-8B in 4-bit with the Phase 9 LoRA adapter applied.

    Requires CUDA plus torch, transformers, peft and bitsandbytes. Imported
    lazily so that merely importing this module does not drag ~2 GB of
    dependencies into a laptop that cannot use them.

    The chat template is loaded explicitly from the adapter directory and
    rendered with ``enable_thinking=False``. Both matter: the adapter ships no
    inline template, and omitting the empty think block feeds the model a
    format it was never trained on — which cost this project one wasted GPU run
    during Phase 10.
    """

    def __init__(
        self,
        adapter_dir: str,
        model_id: str = DEFAULT_MODEL_ID,
        revision: str | None = "b968826d9c46",
        params: InferenceParams | None = None,
        prompt=None,
    ) -> None:
        import torch
        from peft import PeftModel
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            BitsAndBytesConfig,
        )

        if not torch.cuda.is_available():
            raise RuntimeError(
                "MODEL_BACKEND=local needs a CUDA GPU. This machine has none.\n"
                "Use MODEL_BACKEND=hf (serves the base model) or run the API on "
                "a GPU host."
            )

        self.model_id = model_id
        self.revision = revision
        self.adapter_dir = adapter_dir
        self.params = params or InferenceParams()
        self.prompt = prompt or resolve_prompt()
        self._torch = torch

        self._tok = AutoTokenizer.from_pretrained(adapter_dir)
        if not getattr(self._tok, "chat_template", None):
            with open(os.path.join(adapter_dir, "chat_template.jinja"),
                      encoding="utf-8") as fh:
                self._tok.chat_template = fh.read()
        if self._tok.pad_token is None:
            self._tok.pad_token = self._tok.eos_token
        self._tok.padding_side = "left"

        quant = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16,
        )
        base = AutoModelForCausalLM.from_pretrained(
            model_id, revision=revision, quantization_config=quant,
            device_map={"": 0}, torch_dtype=torch.float16,
            attn_implementation="sdpa",
        )
        self._model = PeftModel.from_pretrained(base, adapter_dir)
        self._model.eval()

    def generate(self, question: str, schema: str) -> GenerationResult:
        return self.generate_messages(
            self.prompt.build_messages(question, schema))

    def generate_messages(self, messages: list[dict[str, str]]) -> GenerationResult:
        torch = self._torch
        text = self._tok.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=False)
        enc = self._tok(text, return_tensors="pt",
                        add_special_tokens=False).to(self._model.device)
        n_in = enc["input_ids"].shape[1]
        started = time.perf_counter()
        with torch.no_grad():
            out = self._model.generate(
                **enc, max_new_tokens=self.params.max_tokens, do_sample=False,
                pad_token_id=self._tok.pad_token_id)
        elapsed = (time.perf_counter() - started) * 1000
        gen = out[0][n_in:]
        kept = [t for t in gen.tolist() if t != self._tok.pad_token_id]
        raw = self._tok.decode(gen, skip_special_tokens=True)
        return GenerationResult(
            sql=extract_sql(raw), raw_output=raw, latency_ms=elapsed, ok=True,
            finish_reason=("length" if len(kept) >= self.params.max_tokens
                           else "stop"),
            prompt_tokens=int(n_in), completion_tokens=len(kept),
            provider="local-qlora",
        )

    def describe(self) -> dict[str, Any]:
        return {
            "kind": "local_qlora_adapter",
            "model_id": self.model_id,
            "model_revision": self.revision,
            "fine_tuned": True,
            "adapters": [{"path": self.adapter_dir, "method": "qlora"}],
        }


class HFSpaceModel(TextToSQLModel):
    """Hits the fine-tuned QLoRA adapter running on Hugging Face Spaces (ZeroGPU).

    ZeroGPU dynamically allocates an A100 GPU for each query, running the full
    70.86 % fine-tuned adapter pipeline without requiring a paid dedicated GPU on Render.
    Communicates via Gradio's standard SSE API over HTTP (using httpx).
    """

    def __init__(
        self,
        space_id: str = "garvit-010/enterprise-text-to-sql",
        hf_token: str | None = None,
        timeout_s: float = 60.0,
        prompt=None,
    ) -> None:
        self.space_id = space_id
        # Convert repository format "garvit-010/enterprise-text-to-sql" to subdomain format:
        subdomain = space_id.replace("/", "-").lower()
        self.base_url = f"https://{subdomain}.hf.space"
        self.hf_token = hf_token or os.getenv("HF_TOKEN")
        self.timeout_s = timeout_s
        self.model_id = f"{space_id} (ZeroGPU)"
        self.prompt = prompt or resolve_prompt()

    def generate(self, question: str, schema: str = "") -> GenerationResult:
        import json
        import httpx

        headers = {}
        if self.hf_token:
            headers["Authorization"] = f"Bearer {self.hf_token}"

        started = time.perf_counter()
        try:
            with httpx.Client(timeout=self.timeout_s) as client:
                call_res = client.post(
                    f"{self.base_url}/gradio_api/call/ask",
                    json={"data": [question]},
                    headers=headers,
                )
                call_res.raise_for_status()
                event_id = call_res.json().get("event_id")
                if not event_id:
                    raise RuntimeError(f"HF Space did not return an event_id: {call_res.text}")

                sql = ""
                raw = ""
                with client.stream("GET", f"{self.base_url}/gradio_api/call/ask/{event_id}", headers=headers) as stream:
                    for line in stream.iter_lines():
                        if line.startswith("data:"):
                            payload = json.loads(line[5:])
                            if isinstance(payload, list) and len(payload) > 0:
                                sql = str(payload[0]) if payload[0] is not None else ""
                                raw = str(payload[2]) if len(payload) > 2 and payload[2] is not None else sql
                            break

                elapsed = (time.perf_counter() - started) * 1000
                cleaned_sql = sql or extract_sql(raw)
                return GenerationResult(
                    sql=cleaned_sql,
                    raw_output=raw,
                    latency_ms=elapsed,
                    ok=bool(cleaned_sql),
                    provider="hf-space-zerogpu",
                )
        except Exception as exc:
            elapsed = (time.perf_counter() - started) * 1000
            return GenerationResult(
                sql="",
                raw_output=str(exc),
                latency_ms=elapsed,
                ok=False,
                error=f"HF Space error: {exc}",
                provider="hf-space-zerogpu",
            )

    def generate_messages(self, messages: list[dict[str, str]]) -> GenerationResult:
        question = ""
        for m in reversed(messages):
            if m.get("role") == "user":
                question = m.get("content", "")
                break
        if not question and messages:
            question = messages[-1].get("content", "")
        return self.generate(question)

    def describe(self) -> dict[str, Any]:
        return {
            "kind": "hf_space_zerogpu",
            "space_id": self.space_id,
            "url": self.base_url,
            "fine_tuned": True,
            "adapter_repo": "garvit-010/qwen3-8b-text2sql-qlora-v3",
        }


def build_model(backend: str | None = None) -> TextToSQLModel:
    """Construct the configured backend.

    Failures are loud and specific. A service that silently degrades to a
    different model than the operator intended would report accuracy figures
    that belong to something else.
    """
    name = (backend or os.getenv("MODEL_BACKEND", DEFAULT_BACKEND)).strip().lower()

    builders: dict[str, Callable[[], TextToSQLModel]] = {
        "stub": lambda: StubModel(),
        "hf": lambda: _build_hf(),
        "hf_space": lambda: HFSpaceModel(
            space_id=os.getenv("HF_SPACE_ID", "garvit-010/enterprise-text-to-sql"),
            hf_token=os.getenv("HF_TOKEN"),
            timeout_s=float(os.getenv("HF_TIMEOUT_S", "60")),
        ),
        "space": lambda: HFSpaceModel(
            space_id=os.getenv("HF_SPACE_ID", "garvit-010/enterprise-text-to-sql"),
            hf_token=os.getenv("HF_TOKEN"),
            timeout_s=float(os.getenv("HF_TIMEOUT_S", "60")),
        ),
        "local": lambda: LocalAdapterModel(
            adapter_dir=os.getenv(
                "ADAPTER_DIR", "models/finetuned/final_adapter")),
    }
    if name not in builders:
        raise RuntimeError(
            f"unknown MODEL_BACKEND {name!r}; expected one of "
            f"{sorted(builders)}")
    return builders[name]()


def _build_hf() -> TextToSQLModel:
    from src.model.base_model import HFInferenceModel

    # A failover chain. Free providers flip between fine, busy, timing out and
    # out-of-credits, so the service tries each in turn rather than depending on
    # one. Order matters: the first provider is tried first, and a dead one
    # costs a full timeout before the next is reached.
    #
    # The remaining settings are the difference between a benchmark run and a
    # web request. A batch run can afford a 120 s timeout and minutes of
    # throttle backoff; here a person is watching a spinner, so a dead provider
    # must be abandoned in seconds. On 2026-09-23 nscale was returning 504 after
    # 121 s, which made every request appear to hang for ~166 s before failing.
    return HFInferenceModel(
        model_id=os.getenv("MODEL_ID", DEFAULT_MODEL_ID),
        provider=os.getenv("HF_PROVIDER", "featherless-ai,nscale"),
        timeout_s=float(os.getenv("HF_TIMEOUT_S", "30")),
        max_retries=int(os.getenv("HF_MAX_RETRIES", "4")),
        backoff_s=(3, 8),
        params=InferenceParams(temperature=0.0, max_tokens=512),
        prompt=resolve_prompt(),
    )
