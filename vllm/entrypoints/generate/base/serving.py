# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import json
import time
from collections.abc import Awaitable, Mapping
from dataclasses import MISSING, dataclass, field
from dataclasses import fields as dataclass_fields
from http import HTTPStatus
from typing import ClassVar, Generic, TypeVar

from fastapi import Request
from pydantic import ConfigDict
from starlette.datastructures import Headers

from vllm.engine.protocol import EngineClient
from vllm.entrypoints.generate.beam_search.online import BeamSearchOnlineMixin
from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.entrypoints.openai.completion.protocol import CompletionRequest
from vllm.entrypoints.openai.engine.protocol import (
    ErrorResponse,
    GenerationError,
    PerRequestTimingMetrics,
)
from vllm.entrypoints.openai.models.serving import OpenAIServingModels
from vllm.entrypoints.openai.responses.protocol import ResponsesRequest
from vllm.entrypoints.serve.engine.serving import BaseServing
from vllm.entrypoints.serve.engine.typing import AnyRequest
from vllm.entrypoints.serve.utils.request_logger import RequestLogger
from vllm.inputs import EngineInput
from vllm.logger import init_logger
from vllm.logprobs import Logprob, PromptLogprobs
from vllm.lora.request import LoRARequest
from vllm.sampling_params import (
    BeamSearchParams,
    SamplingParams,
    StructuredOutputsParams,
)
from vllm.tokenizers import TokenizerLike
from vllm.tracing import (
    contains_trace_headers,
    extract_trace_headers,
    log_tracing_disabled_warning,
)
from vllm.v1.metrics.stats import RequestStateStats

logger = init_logger(__name__)

RequestT = TypeVar("RequestT", bound=AnyRequest)
_T = TypeVar("_T")
SESSION_ID_HEADER = "X-Session-ID"
PRIORITY_HEADER = "X-Vllm-Priority"

# ---------------------------------------------------------------------------
# Fail-closed guard: structured output x speculative decoding.
#
# Incident 2026-08-28: a structured-output request served while speculative
# decoding was active killed the engine core (AttributeError in
# `Scheduler.update_draft_token_ids_in_output` truncating an immutable
# draft-token sequence -> EngineDeadError, ~4.5-minute outage; deterministic
# at concurrency 1). The scheduler root cause is fixed in
# vllm/v1/core/sched/scheduler.py, which makes the standard constraint kinds
# below safe again. This guard remains as defense in depth so that any OTHER
# latent spec-decode x structured-output interaction fails the single request
# with HTTP 400 instead of the whole engine: while speculation is active, a
# structured-output request is rejected unless every constraint kind it uses
# is in the validated allowlist below.
#
# Removal: this flag is the single switch — set it to False to disable the
# guard entirely; nothing else depends on it.
GUARD_SPEC_DECODE_STRUCTURED_OUTPUT: bool = True

# Structured-output constraint kinds validated to take the fixed scheduler
# path (all are trimmed in `Scheduler.update_draft_token_ids_in_output` and
# routed through `grammar.validate_tokens`). A kind not listed here — e.g.
# one introduced by a future merge — fails closed with HTTP 400 while
# speculative decoding is active.
SPEC_DECODE_SAFE_STRUCTURED_OUTPUT_KINDS: frozenset[str] = frozenset(
    {"json", "json_object", "regex", "choice", "grammar", "structural_tag"}
)

# StructuredOutputsParams fields that are tuning options rather than
# constraint kinds; they only affect grammar compilation, not which scheduler
# path the request takes, so they are exempt from the allowlist check.
_SPEC_DECODE_STRUCTURED_OPTION_FIELDS: frozenset[str] = frozenset(
    {
        "disable_any_whitespace",
        "disable_additional_properties",
        "whitespace_pattern",
    }
)


def build_per_request_timing_metrics(
    metrics: RequestStateStats | None,
    num_generation_tokens: int,
) -> PerRequestTimingMetrics:
    """Build per-request timing metrics from ``RequestStateStats``.

    ``generation_time_ms`` is the decode interval only (first output token to
    last output token); it excludes both queue wait and prefill/TTFT.
    ``tokens_per_second`` is overall output throughput: all generated tokens
    over the inference interval (scheduling to last output token), so it counts
    the prefill/TTFT phase and is not simply the reciprocal of ``mean_itl_ms``.
    Each field is left ``None`` when the timestamps it depends on are
    unavailable.
    """
    if metrics is None:
        return PerRequestTimingMetrics()

    queued_ts = metrics.queued_ts
    scheduled_ts = metrics.scheduled_ts
    first_token_ts = metrics.first_token_ts
    last_token_ts = metrics.last_token_ts

    time_to_first_token_ms: float | None = None
    generation_time_ms: float | None = None
    queue_time_ms: float | None = None
    mean_itl_ms: float | None = None
    tokens_per_second: float | None = None

    if scheduled_ts > 0 and first_token_ts > 0:
        time_to_first_token_ms = (first_token_ts - scheduled_ts) * 1000

    if first_token_ts > 0 and last_token_ts > 0:
        generation_time_ms = (last_token_ts - first_token_ts) * 1000

    if queued_ts > 0 and scheduled_ts > 0:
        queue_time_ms = (scheduled_ts - queued_ts) * 1000

    if first_token_ts > 0 and last_token_ts > 0 and num_generation_tokens > 1:
        decode_time = last_token_ts - first_token_ts
        mean_itl_ms = decode_time / (num_generation_tokens - 1) * 1000

    if scheduled_ts > 0 and last_token_ts > 0:
        inference_time_ms = (last_token_ts - scheduled_ts) * 1000
        if inference_time_ms > 0:
            tokens_per_second = num_generation_tokens / inference_time_ms * 1000

    return PerRequestTimingMetrics(
        time_to_first_token_ms=time_to_first_token_ms,
        generation_time_ms=generation_time_ms,
        queue_time_ms=queue_time_ms,
        mean_itl_ms=mean_itl_ms,
        tokens_per_second=tokens_per_second,
    )


@dataclass(kw_only=True)
class ServeContext(Generic[RequestT]):
    request: RequestT
    raw_request: Request | None = None
    model_name: str
    request_id: str
    created_time: int = field(default_factory=lambda: int(time.time()))
    lora_request: LoRARequest | None = None
    engine_inputs: list[EngineInput] | None = None
    model_config = ConfigDict(arbitrary_types_allowed=True)


class GenerateBaseServing(BaseServing, BeamSearchOnlineMixin):
    request_id_prefix: ClassVar[str] = """
    A short string prepended to every request’s ID.
    """

    def __init__(
        self,
        engine_client: EngineClient,
        models: OpenAIServingModels,
        *,
        request_logger: RequestLogger | None,
        return_tokens_as_token_ids: bool = False,
    ):
        super().__init__(
            models=models,
            model_config=engine_client.model_config,
            request_logger=request_logger,
        )

        self.engine_client = engine_client
        self.return_tokens_as_token_ids = return_tokens_as_token_ids
        self.renderer = engine_client.renderer
        self.input_processor = engine_client.input_processor
        vllm_config = getattr(engine_client, "vllm_config", None)
        kv_transfer_config = getattr(vllm_config, "kv_transfer_config", None)
        self.has_kv_connector = kv_transfer_config is not None
        # Needed by the fail-closed structured-output x spec-decode guard
        # (see GUARD_SPEC_DECODE_STRUCTURED_OUTPUT above).
        self.speculative_config = getattr(vllm_config, "speculative_config", None)

        # Computed once at startup (cached by ``vllm_config`` identity) and
        # stamped on non-streaming responses. Streaming chunks deliberately
        # omit it to avoid per-chunk overhead.
        from vllm.entrypoints.serve.utils.fingerprint import get_system_fingerprint

        try:
            self.system_fingerprint: str | None = get_system_fingerprint(
                engine_client.vllm_config
            )
        except Exception:
            # Never fail server startup over the fingerprint.
            self.system_fingerprint = None

    def create_streaming_error_response(
        self,
        message: str | Exception,
        err_type: str = "BadRequestError",
        status_code: HTTPStatus = HTTPStatus.BAD_REQUEST,
        param: str | None = None,
    ) -> str:
        json_str = json.dumps(
            self.create_error_response(
                message=message,
                err_type=err_type,
                status_code=status_code,
                param=param,
            ).model_dump()
        )
        return json_str

    def _check_spec_decode_structured_output(
        self, params: SamplingParams | BeamSearchParams
    ) -> ErrorResponse | None:
        """Fail-closed guard for structured output x speculative decoding.

        See GUARD_SPEC_DECODE_STRUCTURED_OUTPUT at module level. Returns an
        HTTP 400 ErrorResponse when speculative decoding is active and the
        request carries a structured-output constraint that is not validated
        to take the fixed scheduler path; returns None when the request is
        safe to admit.
        """
        if not GUARD_SPEC_DECODE_STRUCTURED_OUTPUT:
            return None
        if self.speculative_config is None:
            # No speculation configured on this engine: nothing to guard.
            return None
        struct_out = getattr(params, "structured_outputs", None)
        if struct_out is None:
            # Not a structured-output request (includes beam search params).
            return None
        if isinstance(struct_out, StructuredOutputsParams):
            # Enumerate constraint state dynamically from the dataclass so
            # that a constraint kind added by a future merge is caught here
            # (fails closed) instead of silently reaching the engine.
            active_kinds: list[str] = []
            for f in dataclass_fields(struct_out):
                name = f.name
                if name.startswith("_"):
                    # Internal bookkeeping (_backend, ...).
                    continue
                if name in _SPEC_DECODE_STRUCTURED_OPTION_FIELDS:
                    # Known tuning options; do not select a scheduler path.
                    continue
                default = f.default
                if default is MISSING and f.default_factory is not MISSING:  # type: ignore[misc]
                    default = f.default_factory()  # type: ignore[misc]
                value = getattr(struct_out, name)
                is_set = (
                    value is not None if default is MISSING else value != default
                )
                if is_set:
                    active_kinds.append(name)
            unsafe_kinds = sorted(
                set(active_kinds) - SPEC_DECODE_SAFE_STRUCTURED_OUTPUT_KINDS
            )
        else:
            # Unknown carrier of structured-output state: we cannot prove it
            # takes the validated path, so fail closed.
            unsafe_kinds = [f"<{type(struct_out).__name__}>"]
        if not unsafe_kinds:
            return None
        spec_method = getattr(self.speculative_config, "method", None)
        return self.create_error_response(
            message=(
                f"Structured output constraint kind(s) {unsafe_kinds} are not "
                "validated for use with speculative decoding "
                f"(method={spec_method!r}), which is active on this server. "
                "Rejecting this request instead of risking the engine: on "
                "2026-08-28 a structured-output request combined with "
                "speculative decoding crashed the engine core "
                "(EngineDeadError), causing a multi-minute outage for all "
                "requests. Validated kinds: "
                f"{sorted(SPEC_DECODE_SAFE_STRUCTURED_OUTPUT_KINDS)}. Either "
                "remove the structured output constraint (response_format / "
                "structured_outputs) from the request, or have the server "
                "operator disable speculative decoding "
                "(--speculative-config)."
            ),
            err_type="BadRequestError",
            status_code=HTTPStatus.BAD_REQUEST,
            param="response_format",
        )

    def _raise_if_error(self, finish_reason: str | None, request_id: str) -> None:
        """Raise GenerationError if finish_reason indicates an error."""
        if finish_reason == "error":
            logger.error(
                "Request %s failed with an internal error during generation",
                request_id,
            )
            raise GenerationError("Internal server error")

    def _convert_generation_error_to_streaming_response(
        self, e: GenerationError
    ) -> str:
        """Convert GenerationError to streaming error response."""
        return self.create_streaming_error_response(
            str(e),
            err_type="InternalServerError",
            status_code=e.status_code,
        )

    async def _get_trace_headers(
        self,
        headers: Headers,
    ) -> Mapping[str, str] | None:
        is_tracing_enabled = await self.engine_client.is_tracing_enabled()

        if is_tracing_enabled:
            return extract_trace_headers(headers)

        if contains_trace_headers(headers):
            log_tracing_disabled_warning()

        return None

    @staticmethod
    def _get_data_parallel_rank(raw_request: Request | None) -> int | None:
        """Pulls the data parallel rank from a header, if provided"""
        if raw_request is None:
            return None

        rank_str = raw_request.headers.get("X-data-parallel-rank")
        if rank_str is None:
            return None

        try:
            return int(rank_str)
        except ValueError:
            return None

    @staticmethod
    def _get_session_id_from_headers(raw_request: Request | None) -> str | None:
        if raw_request is None:
            return None
        if value := raw_request.headers.get(SESSION_ID_HEADER):
            return value
        return None

    @staticmethod
    def _get_session_id(
        request: ChatCompletionRequest | CompletionRequest | ResponsesRequest,
        raw_request: Request | None,
    ) -> str | None:
        if request.session_id:
            return request.session_id
        if value := GenerateBaseServing._get_session_id_from_headers(raw_request):
            return value
        if request.vllm_xargs:
            session_id = request.vllm_xargs.get("session_id")
            if isinstance(session_id, str) and session_id:
                return session_id
        return None

    @staticmethod
    def _get_priority(
        request: ChatCompletionRequest | CompletionRequest | ResponsesRequest,
        raw_request: Request | None,
    ) -> int:
        if raw_request is not None:
            priority = raw_request.headers.get(PRIORITY_HEADER)
            if priority is not None:
                try:
                    return int(priority)
                except ValueError:
                    pass
        return request.priority

    async def _with_kv_transfer_rejection_cleanup(
        self,
        awaitable: Awaitable[_T],
        request: ChatCompletionRequest | CompletionRequest | ResponsesRequest,
        raw_request: Request | None,
    ) -> _T:
        """Wrap a `create_*` coroutine so that, if it raises or returns an
        ErrorResponse (i.e. the request never reached the engine), the KV
        connector is notified to free any pinned remote-prefill blocks."""
        kv_transfer_params = self.has_kv_connector and request.kv_transfer_params
        if not kv_transfer_params or not kv_transfer_params.get("do_remote_prefill"):
            return await awaitable

        notify = True
        try:
            result = await awaitable
            if not isinstance(result, ErrorResponse):
                notify = False
            return result
        finally:
            if notify:
                try:
                    await self.engine_client.notify_kv_transfer_request_rejected(
                        request.request_id,
                        kv_transfer_params,
                        data_parallel_rank=self._get_data_parallel_rank(raw_request),
                    )
                except Exception:
                    logger.warning(
                        "Failed to notify KV connector about rejected request %s",
                        request.request_id,
                        exc_info=True,
                    )

    @staticmethod
    def _get_decoded_token(
        logprob: Logprob,
        token_id: int,
        tokenizer: TokenizerLike | None,
        return_as_token_id: bool = False,
    ) -> str:
        if return_as_token_id:
            return format_token_id_placeholder(token_id)

        if logprob.decoded_token is not None:
            return logprob.decoded_token

        if tokenizer is None:
            raise ValueError(
                "Unable to get tokenizer because `skip_tokenizer_init=True`"
            )

        return tokenizer.decode([token_id])


def format_token_id_placeholder(token_id: int) -> str:
    return f"token_id:{token_id}"


def resolve_token_id_placeholder(
    token: str, tokenizer: TokenizerLike
) -> tuple[str, list[int] | None]:
    """Decode a 'token_id:N' placeholder back to a token string and UTF-8 bytes.

    Returns (token, None) unchanged if token is not a placeholder.
    This is the inverse of format_token_id_placeholder / _get_decoded_token
    when return_as_token_id=True.
    """
    suffix = token.removeprefix("token_id:")
    if suffix == token:
        return token, None
    try:
        token_id = int(suffix)
    except ValueError:
        return token, None
    token_repr = tokenizer.convert_ids_to_tokens([token_id])[0]
    if token_repr is None:
        logger.warning_once(
            "resolve_token_id_placeholder: token_id %d has no vocab entry; "
            "substituting empty string",
            token_id,
        )
        return "", None
    token_str = tokenizer.convert_tokens_to_string([token_repr])
    return token_str, list(token_str.encode("utf-8", errors="replace"))


def clamp_prompt_logprobs(
    prompt_logprobs: PromptLogprobs | None,
) -> PromptLogprobs | None:
    if prompt_logprobs is None:
        return prompt_logprobs

    for logprob_dict in prompt_logprobs:
        if logprob_dict is None:
            continue
        for logprob_values in logprob_dict.values():
            if logprob_values.logprob == float("-inf"):
                logprob_values.logprob = -9999.0
    return prompt_logprobs
