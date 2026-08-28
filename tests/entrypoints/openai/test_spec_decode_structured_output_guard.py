# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the fail-closed structured-output x speculative-decoding guard.

Incident 2026-08-28: a structured-output request served while speculative
decoding was active crashed the engine core (EngineDeadError). The scheduler
root cause is fixed; this serving-layer guard remains as defense in depth so
that any OTHER latent spec+structured interaction fails the single request
with HTTP 400 instead of the engine.

Behavior under test (see GenerateBaseServing._check_spec_decode_structured_output):
  * speculation active + structured constraint NOT on the validated (fixed)
    path -> HTTP 400 ErrorResponse citing the incident;
  * speculation active + validated structured constraint kinds -> admitted
    (they take the now-fixed scheduler path);
  * non-structured request + speculation -> admitted;
  * structured request + no speculation -> admitted;
  * single documented flag disables the guard entirely (removability).
"""

from dataclasses import dataclass
from types import SimpleNamespace

import pytest

import vllm.entrypoints.generate.base.serving as serving_mod
from vllm.entrypoints.generate.base.serving import GenerateBaseServing
from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.entrypoints.openai.engine.protocol import ErrorResponse
from vllm.sampling_params import (
    BeamSearchParams,
    SamplingParams,
    StructuredOutputsParams,
)

SPEC_CONFIG = SimpleNamespace(method="mtp", num_speculative_tokens=2)


@dataclass
class FutureStructuredOutputsParams(StructuredOutputsParams):
    """Simulates a constraint kind added by a future upstream merge that has
    NOT been validated against speculative decoding."""

    ebnf_v2: str | None = None


def _check(params, speculative_config):
    """Run the guard exactly as the serving paths do, on a minimal self."""
    fake_serving = SimpleNamespace(
        speculative_config=speculative_config,
        create_error_response=GenerateBaseServing.create_error_response,
    )
    return GenerateBaseServing._check_spec_decode_structured_output(
        fake_serving, params
    )


def _assert_rejected_400(result, expect_in_message=()):
    assert isinstance(result, ErrorResponse), (
        "guard must reject with an ErrorResponse, got: %r" % (result,)
    )
    assert result.error.code == 400
    assert result.error.type == "BadRequestError"
    # Actionable, incident-citing message.
    assert "2026-08-28" in result.error.message
    assert "speculative decoding" in result.error.message
    for fragment in expect_in_message:
        assert fragment in result.error.message


def test_unvalidated_structured_kind_with_spec_rejected_400():
    """Structured request off the validated path + speculation -> 400."""
    params = SamplingParams(
        structured_outputs=FutureStructuredOutputsParams(
            json_object=True, ebnf_v2="root ::= 'x'"
        )
    )
    result = _check(params, SPEC_CONFIG)
    _assert_rejected_400(result, expect_in_message=("ebnf_v2", "'mtp'"))


def test_unknown_structured_outputs_carrier_with_spec_rejected_400():
    """A structured_outputs value we cannot introspect fails closed."""
    params = SimpleNamespace(structured_outputs={"json": "{}"})
    result = _check(params, SPEC_CONFIG)
    _assert_rejected_400(result, expect_in_message=("<dict>",))


@pytest.mark.parametrize(
    "struct_kwargs",
    [
        {"json": '{"type": "object"}'},
        {"json_object": True},
        {"regex": r"\d+"},
        {"choice": ["a", "b"]},
        {"grammar": "root ::= 'x'"},
        {"structural_tag": '{"type": "structural_tag", "format": {}}'},
    ],
    ids=["json", "json_object", "regex", "choice", "grammar", "structural_tag"],
)
def test_validated_structured_kinds_with_spec_admitted(struct_kwargs):
    """Validated kinds take the now-fixed scheduler path -> admitted."""
    params = SamplingParams(
        structured_outputs=StructuredOutputsParams(**struct_kwargs)
    )
    assert _check(params, SPEC_CONFIG) is None


def test_validated_kind_with_options_with_spec_admitted():
    """Known tuning options do not trip the allowlist."""
    params = SamplingParams(
        structured_outputs=StructuredOutputsParams(
            json='{"type": "object"}',
            disable_any_whitespace=True,
            whitespace_pattern=r"\s*",
        )
    )
    assert _check(params, SPEC_CONFIG) is None


def test_non_structured_with_spec_admitted():
    assert _check(SamplingParams(), SPEC_CONFIG) is None


def test_beam_search_params_with_spec_admitted():
    params = BeamSearchParams(beam_width=2, max_tokens=8)
    assert _check(params, SPEC_CONFIG) is None


def test_structured_without_spec_admitted():
    """No speculation configured: even unvalidated kinds are not guarded."""
    params = SamplingParams(
        structured_outputs=FutureStructuredOutputsParams(
            json_object=True, ebnf_v2="root ::= 'x'"
        )
    )
    assert _check(params, None) is None


def test_response_format_maps_to_validated_kind_and_is_admitted():
    """End-to-end protocol check: OpenAI response_format=json_object builds
    sampling params that pass the guard while speculation is active."""
    request = ChatCompletionRequest(
        model="test-model",
        messages=[{"role": "user", "content": "hi"}],
        response_format={"type": "json_object"},
    )
    params = request.to_sampling_params(32, {})
    assert params.structured_outputs is not None
    assert _check(params, SPEC_CONFIG) is None


def test_single_flag_disables_guard(monkeypatch):
    """Removability: the documented flag is the entire switch."""
    monkeypatch.setattr(
        serving_mod, "GUARD_SPEC_DECODE_STRUCTURED_OUTPUT", False
    )
    params = SamplingParams(
        structured_outputs=FutureStructuredOutputsParams(
            json_object=True, ebnf_v2="root ::= 'x'"
        )
    )
    assert _check(params, SPEC_CONFIG) is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
