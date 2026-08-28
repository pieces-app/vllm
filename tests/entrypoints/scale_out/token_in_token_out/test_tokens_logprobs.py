# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm.entrypoints.scale_out.token_in_token_out.serving import ServingTokens
from vllm.logprobs import Logprob


def test_top_logprobs_alternatives_have_own_token_ids():
    """Each top_logprobs alternative must carry its own token_id placeholder."""
    result = ServingTokens._create_tokens_logprobs(
        None,
        token_ids=[262],
        top_logprobs=[{262: Logprob(-0.1), 257: Logprob(-1.2), 428: Logprob(-2.3)}],
        num_output_top_logprobs=2,
    )
    tokens = {e.token for e in result.content[0].top_logprobs}
    assert tokens == {"token_id:262", "token_id:257"}, f"got {tokens}"


def test_logprobs_zero_emits_sampled_token():
    """logprobs=0 must still emit 1 entry (the sampled token)."""
    result = ServingTokens._create_tokens_logprobs(
        None,
        token_ids=[7],
        top_logprobs=[{7: Logprob(-0.9), 8: Logprob(-1.1)}],
        num_output_top_logprobs=0,
    )
    assert len(result.content[0].top_logprobs) == 1


def test_logprobs_short_engine_data_no_500():
    """Regression for the crash site: a backend may return *fewer* logprob
    positions than sampled tokens (seen with `logprobs: 0` on backends that
    gate logprob gathering on `num_logprobs > 0`). This endpoint takes a raw
    SamplingParams from the caller, so it cannot floor the count the way the
    validated OpenAI protocols do -- it must degrade to token-only entries
    instead of raising IndexError -> HTTP 500. Sibling of the chat fix in PR #2.
    """
    result = ServingTokens._create_tokens_logprobs(
        None,
        token_ids=[7, 8, 9],
        top_logprobs=[],
        num_output_top_logprobs=0,
    )
    assert [e.token for e in result.content] == [
        "token_id:7",
        "token_id:8",
        "token_id:9",
    ]
    # Token-only entries: no engine logprob, so the field keeps its default
    # sentinel and the top list stays empty (no IndexError, no 500).
    assert all(e.logprob == -9999.0 for e in result.content)
    assert all(e.top_logprobs == [] for e in result.content)


def test_logprobs_partial_engine_data_keeps_real_entries():
    """Control for the degrade path: positions the runner *did* return keep
    their real logprob; only the missing tail degrades to a token-only entry."""
    result = ServingTokens._create_tokens_logprobs(
        None,
        token_ids=[7, 8],
        top_logprobs=[{7: Logprob(-0.9)}],
        num_output_top_logprobs=1,
    )
    assert result.content[0].logprob == -0.9
    assert result.content[0].token == "token_id:7"
    assert result.content[1].logprob == -9999.0
    assert result.content[1].token == "token_id:8"
