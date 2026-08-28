# SPDX-License-Identifier: Apache-2.0
"""`top_logprobs` defaults to 0 here, so asking for logprobs the ordinary way
asked the engine for zero of them.

`include: ["message.output_text.logprobs"]` with `top_logprobs` left alone is
the documented way to request logprobs on this endpoint. It produced
`SamplingParams.logprobs = 0`, the engine returned nothing, and the response
carried no logprobs at all -- while the identical request shape on
/chat/completions and /completions returns the chosen token's logprob. That is
a contract divergence between our own endpoints, not a missing extra.

These tests pin BOTH halves of the fix: the engine request is floored at
top-1, and the echoed `top_logprobs` still reports the user's own value.
"""

from vllm.entrypoints.openai.responses.protocol import ResponsesRequest

LOGPROBS_INCLUDE = ["message.output_text.logprobs"]


def _params(**kwargs):
    request = ResponsesRequest(input="hi", **kwargs)
    return request, request.to_sampling_params(default_max_tokens=16)


def test_default_top_logprobs_floors_engine_request_to_one():
    """THE DEFECT: include logprobs, leave top_logprobs at its default 0."""
    request, params = _params(include=LOGPROBS_INCLUDE)
    assert request.top_logprobs == 0, "precondition: the field defaults to 0"
    assert params.logprobs == 1, (
        "the engine must be asked for the chosen token's logprob; 0 is what "
        "silently returned nothing"
    )


def test_explicit_zero_is_also_floored():
    _, params = _params(include=LOGPROBS_INCLUDE, top_logprobs=0)
    assert params.logprobs == 1


def test_explicit_value_passes_through_unfloored():
    _, params = _params(include=LOGPROBS_INCLUDE, top_logprobs=5)
    assert params.logprobs == 5, "the floor must not clamp a real request"


def test_logprobs_not_requested_stays_none():
    """No `include` -> the engine is not asked for logprobs at all."""
    _, params = _params()
    assert params.logprobs is None


def test_include_without_logprobs_entry_stays_none():
    _, params = _params(include=["message.input_image.image_url"])
    assert params.logprobs is None


def test_echo_reports_the_users_value_not_the_floored_one():
    """The wire contract is unchanged: a user who sent 0 (or sent nothing)
    still sees 0 echoed back, even though the engine was asked for 1. This is
    the half that makes the floor invisible to callers."""
    request, params = _params(include=LOGPROBS_INCLUDE)
    assert params.logprobs == 1
    echoed = (
        request.top_logprobs if request.is_include_output_logprobs() else None
    )
    assert echoed == 0, (
        "echoing the ENGINE value here would report 1 and change the wire "
        "shape; the echo must come from the request"
    )
