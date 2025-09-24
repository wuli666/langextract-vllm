"""Provider implementation for vLLM"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, Generator, List, Optional, Union

import langextract as lx
from langextract.core import base_model, exceptions, types

logger = logging.getLogger(__name__)

VLLM_PATTERNS = (r"^vllm:",)

# Lazy import to avoid hard dependency at import time
try:
    from vllm import LLM, SamplingParams  # type: ignore
except Exception:
    LLM = None  # type: ignore
    SamplingParams = None  # type: ignore

# For API server support
try:
    import openai
except Exception:
    openai = None  # type: ignore


@lx.providers.registry.register(*VLLM_PATTERNS, priority=20)
class VLLMLanguageModel(base_model.BaseLanguageModel):
    """vLLM integration with local engine and API server support.

    Supported model_id patterns:
      - "vllm:model_name" -> Local vLLM engine (default)
      - "vllm:http://host:port/v1" or "vllm:https://host:port/v1" -> vLLM API server

    Features:
      - Plain string or messages input
      - True batching (single vLLM generate call for local, batch API calls for server)
      - SamplingParams passthrough/override
      - Auto/custom stop tokens
      - Minimal JSON cleaning (opt-in via return_json=True)
    """

    def __init__(
        self,
        model_id: str,
        *,
        system_prompt: Optional[str] = None,
        temperature: float = 0.0,
        top_p: float = 1.0,
        max_tokens: int = 1024,
        gpu_memory_utilization: float = 0.7,
        max_model_len: int = 1024,
        **kwargs: Any,
    ):
        super().__init__()

        self.model_id = model_id
        self.system_prompt = system_prompt

        # Normalize model id ("vllm:foo/bar" -> "foo/bar")
        actual_model_id = model_id.split(":", 1)[1] if model_id.startswith("vllm:") else model_id

        # Check if this is an API server URL
        self.is_api_server = actual_model_id.startswith(("http://", "https://"))

        if self.is_api_server:
            self._initialize_api_client(actual_model_id, temperature, top_p, max_tokens, **kwargs)
        else:
            self._initialize_local_engine(actual_model_id, temperature, top_p, max_tokens, gpu_memory_utilization, max_model_len, **kwargs)

    def _initialize_api_client(self, base_url: str, temperature: float, top_p: float, max_tokens: int, **kwargs: Any):
        """Initialize OpenAI-compatible API client for vLLM server."""
        if openai is None:
            raise exceptions.InferenceConfigError(
                "OpenAI package is required for API server mode. Please run: pip install openai"
            )

        self._client = openai.OpenAI(
            base_url=base_url,
            api_key="dummy",  # vLLM doesn't require real API key
        )

        # Get available models from the server
        try:
            models_response = self._client.models.list()
            available_models = [model.id for model in models_response.data]
            if available_models:
                self._model_name = available_models[0]  # Use the first available model
                logger.info(f"Using model: {self._model_name} (available: {available_models})")
            else:
                raise exceptions.InferenceConfigError("No models available on the VLLM server")
        except Exception as e:
            logger.warning(f"Failed to get models from server, using 'dummy': {e}")
            self._model_name = "dummy"  # Fallback

        # Store API parameters
        self._api_params = {
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens,
        }

        # Filter out vLLM-specific params for API mode
        api_compatible_params = {"temperature", "top_p", "max_tokens", "stop", "seed"}
        for k, v in kwargs.items():
            if k in api_compatible_params:
                self._api_params[k] = v

    def _initialize_local_engine(self, actual_model_id: str, temperature: float, top_p: float, max_tokens: int, gpu_memory_utilization: float, max_model_len: int, **kwargs: Any):
        """Initialize local vLLM engine."""
        if LLM is None or SamplingParams is None:
            raise exceptions.InferenceConfigError(
                "vLLM is not installed. Please run: pip install vllm"
            )

        # Split kwargs into engine kwargs vs SamplingParams overrides
        sampling_keys = self._sampling_param_names()
        engine_block = {"temperature", "top_p", "max_tokens", "gpu_memory_utilization", "max_model_len", "stop", "timeout"}
        sampling_overrides = {k: v for k, v in kwargs.items() if k in sampling_keys}
        engine_kwargs = {k: v for k, v in kwargs.items() if k not in sampling_keys and k not in engine_block}

        # Safe engine defaults (can be overridden)
        engine_kwargs.setdefault("enforce_eager", True)
        engine_kwargs.setdefault("disable_custom_all_reduce", True)

        # Initialize engine
        self._engine = LLM(
            model=actual_model_id,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
            **engine_kwargs,
        )

        # Tokenizer (for templates/stop detection)
        self.tokenizer = self._engine.get_tokenizer()
        if self.tokenizer is None:
            raise exceptions.InferenceConfigError("Failed to obtain tokenizer from vLLM engine.")
        self.use_chat_template = bool(getattr(self.tokenizer, "chat_template", None))

        # Stop detection (custom > auto > None)
        stop_tokens = self._auto_stop(kwargs.get("stop"))

        # Build initial SamplingParams (defaults + overrides)
        base_sampling = {
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens,
            "stop": stop_tokens,          # None => let engine/model handle EOS
            "skip_special_tokens": True,
        }
        base_sampling.update(sampling_overrides)
        self._sampling = self._build_sampling(base_sampling)

    # -------------------------
    # Internal helpers
    # -------------------------

    def _sampling_param_names(self) -> set[str]:
        """Discover valid SamplingParams fields (fallback to common set)."""
        try:
            code = SamplingParams.__init__.__code__  # type: ignore[attr-defined]
            names = set(code.co_varnames[: code.co_argcount])
            names.discard("self")
            return names
        except Exception:
            return {
                "temperature", "top_p", "top_k", "min_p", "top_a",
                "max_tokens",
                "stop", "stop_token_ids",
                "presence_penalty", "frequency_penalty", "repetition_penalty", "length_penalty",
                "n", "best_of",
                "use_beam_search", "beam_width",
                "seed", "logprobs",
                "ignore_eos", "skip_special_tokens", "include_stop_str_in_output",
            }

    def _build_sampling(self, params: Dict[str, Any]) -> SamplingParams:
        """Filter unknown keys and build SamplingParams safely."""
        valid = self._sampling_param_names()
        safe = {k: v for k, v in params.items() if k in valid}
        try:
            return SamplingParams(**safe)
        except Exception as e:
            raise exceptions.InferenceConfigError(f"Invalid SamplingParams: {type(e).__name__}: {e}") from e

    def _auto_stop(self, custom: Optional[Union[str, List[str]]]) -> Optional[List[str]]:
        """Return custom stops if provided; otherwise only high-confidence stops."""
        if custom:
            if isinstance(custom, str):
                custom = [custom]
            custom = [str(s) for s in custom if s]
            # dedup keep order
            seen, out = set(), []
            for s in custom:
                if s not in seen:
                    seen.add(s)
                    out.append(s)
            return out or None

        stops: List[str] = []
        eos = getattr(self.tokenizer, "eos_token", None)
        if eos:
            stops.append(str(eos))
        if self.use_chat_template:
            tmpl = getattr(self.tokenizer, "chat_template", "") or ""
            for mark in ("<|eot_id|>", "<|im_end|>", "</s>", "<|endoftext|>"):
                if mark in tmpl:
                    stops.append(mark)
        pad = getattr(self.tokenizer, "pad_token", None)
        stops = [s for s in stops if s and s != pad]
        # dedup keep order
        seen, out = set(), []
        for s in stops:
            if s not in seen:
                seen.add(s)
                out.append(s)
        return out or None

    def _apply_template(self, messages: List[Dict[str, str]]) -> str:
        """Apply tokenizer chat template when available; fallback to stitched transcript."""
        if self.use_chat_template:
            try:
                return self.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
            except Exception as e:
                logger.warning("Chat template failed: %r", e)
        # Fallback (plain stitched dialogue)
        buf = [f"{m.get('role', 'user')}: {m.get('content', '')}" for m in messages]
        buf.append("assistant:")
        return "\n".join(buf)

    def _format_one(self, item: Union[str, List[Dict[str, str]]]) -> str:
        """Format one input (string or messages) into a ready prompt."""
        if isinstance(item, str):
            if not self.use_chat_template:
                return item
            msgs = []
            if self.system_prompt:
                msgs.append({"role": "system", "content": self.system_prompt})
            msgs.append({"role": "user", "content": item})
            return self._apply_template(msgs)

        # messages list
        msgs = list(item)
        if self.system_prompt and not any(m.get("role") == "system" for m in msgs):
            msgs.insert(0, {"role": "system", "content": self.system_prompt})
        return self._apply_template(msgs)

    def _clean_json_min(self, text: str) -> str:
        """Extract and fix JSON from model output, add markers if needed."""
        t = text.strip()

        # 1) Try the text as-is first
        try:
            parsed = json.loads(t)
            # If it's valid JSON, wrap it with markers for langextract
            if isinstance(parsed, dict) and 'extractions' in parsed:
                return f"```json\n{t}\n```"
            return t
        except Exception:
            pass

        # 2) Find JSON content - look for { or [ and extract from there
        json_start = -1
        for i, char in enumerate(t):
            if char in '{[':
                json_start = i
                break

        if json_start >= 0:
            # Find the matching closing brace/bracket
            extracted = t[json_start:]

            # Try to extract a complete JSON object/array
            for end_pos in range(len(extracted), 0, -1):
                candidate = extracted[:end_pos].rstrip()
                try:
                    parsed = json.loads(candidate)
                    # If valid and contains extractions, wrap with markers
                    if isinstance(parsed, dict) and 'extractions' in parsed:
                        return f"```json\n{candidate}\n```"
                    return candidate
                except Exception:
                    continue

            # If no valid JSON found, try some basic fixes
            # Close unmatched braces
            if extracted.startswith("{"):
                open_n, close_n = extracted.count("{"), extracted.count("}")
                if open_n > close_n:
                    fixed = extracted + ("}" * (open_n - close_n))
                    try:
                        parsed = json.loads(fixed)
                        if isinstance(parsed, dict) and 'extractions' in parsed:
                            return f"```json\n{fixed}\n```"
                        return fixed
                    except Exception:
                        pass

        # 3) Last resort - return original text
        return t

    def _generate_batch(self, prompts: List[str], sampling: SamplingParams) -> List[str]:
        """Single batched call to vLLM; returns first candidate per input."""
        outputs = self._engine.generate(prompts, sampling, use_tqdm=False)
        results: List[str] = []
        for out in outputs:
            if not out.outputs:
                results.append("")
            else:
                results.append(out.outputs[0].text)
        return results

    def _generate_batch_api(self, prompts: List[str], **kwargs: Any) -> List[str]:
        """Generate using API server (individual API calls)."""
        results: List[str] = []
        params = dict(self._api_params)
        params.update(kwargs)

        for prompt in prompts:
            try:
                response = self._client.chat.completions.create(
                    model=self._model_name,
                    messages=[{"role": "user", "content": prompt}],
                    **params
                )
                content = response.choices[0].message.content or ""
                results.append(content)
            except Exception as e:
                raise exceptions.InferenceRuntimeError(f"API server error: {type(e).__name__}: {e}", original=e) from e

        return results

    # -------------------------
    # Public API
    # -------------------------

    def infer(
        self,
        batch_prompts: List[Union[str, List[Dict[str, str]]]],
        **kwargs: Any,
    ) -> Generator[List[types.ScoredOutput], None, None]:
        """Run batched inference with vLLM.

        Args:
            batch_prompts: Each item is either a plain string or a list of messages
                           like {'role': 'user|system|assistant', 'content': '...'}.
            **kwargs: SamplingParams overrides (e.g., temperature/top_p/max_tokens/seed/stop/...).
                      Extra flag:
                        - return_json: bool = False -> apply minimal JSON cleaning if True.

        Yields:
            For each input, yields a list with a single ScoredOutput.
        """
        do_json = bool(kwargs.get("return_json", False))

        # Format inputs
        try:
            if self.is_api_server:
                # For API mode, we need simpler formatting since we don't have tokenizer
                formatted = []
                for p in batch_prompts:
                    if isinstance(p, str):
                        if self.system_prompt:
                            # Simple concatenation for API mode
                            formatted.append(f"{self.system_prompt}\n\n{p}")
                        else:
                            formatted.append(p)
                    else:
                        # Handle messages format - simple concatenation
                        parts = []
                        if self.system_prompt:
                            parts.append(f"system: {self.system_prompt}")
                        for msg in p:
                            role = msg.get("role", "user")
                            content = msg.get("content", "")
                            parts.append(f"{role}: {content}")
                        formatted.append("\n".join(parts))
            else:
                formatted = [self._format_one(p) for p in batch_prompts]
        except Exception as e:
            raise exceptions.InferenceConfigError(f"Failed to format prompts: {type(e).__name__}: {e}") from e

        # Generate
        try:
            if self.is_api_server:
                # Prepare API parameters
                api_kwargs = {}
                api_compatible_params = {"temperature", "top_p", "max_tokens", "stop", "seed"}
                for k, v in kwargs.items():
                    if k in api_compatible_params:
                        api_kwargs[k] = v
                texts = self._generate_batch_api(formatted, **api_kwargs)
            else:
                # Merge per-call SamplingParams overrides for local engine
                if kwargs:
                    valid = self._sampling_param_names()
                    ov = {k: v for k, v in kwargs.items() if k in valid}
                    if ov:
                        cur = {k: getattr(self._sampling, k, None) for k in valid}
                        cur.update(ov)
                        cur.setdefault("stop", getattr(self._sampling, "stop", None))
                        cur.setdefault("skip_special_tokens", getattr(self._sampling, "skip_special_tokens", True))
                        self._sampling = self._build_sampling(cur)
                texts = self._generate_batch(formatted, self._sampling)
        except Exception as e:
            error_type = "API server" if self.is_api_server else "vLLM engine"
            raise exceptions.InferenceRuntimeError(f"{error_type} error: {type(e).__name__}: {e}", original=e) from e

        # Yield results
        for t in texts:
            # Always try to clean/extract JSON if it looks like we need structured output
            if do_json or ('{' in t or '[' in t):
                out = self._clean_json_min(t)
            else:
                out = t
            yield [types.ScoredOutput(score=1.0, output=out)]
