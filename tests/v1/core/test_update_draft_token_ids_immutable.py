# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Regression tests for the 2026-08-28 spec-decode x structured-output
engine crash.

`Scheduler.update_draft_token_ids_in_output` truncated the per-request draft
token ids with `del spec_token_ids[n:]` and later mutated them with
`.extend(...)`. Some executor backends deliver those ids as an immutable
sequence (observed live on the TPU runner with structured output +
speculative decoding), so the in-place mutation raised AttributeError inside
the engine core -> EngineDeadError -> multi-minute outage.

These tests drive the method directly (CPU only, no engine build) with a
lightweight fake scheduler `self`, and prove:
  * an immutable draft sequence no longer crashes, and the TRUNCATED result
    is written back into `scheduler_output.scheduled_spec_decode_tokens`
    (the store that `get_grammar_bitmask` and the worker read) - a local
    `list(...)` nobody stores would fail these assertions;
  * the plain-list fast path still truncates in place (same object, no
    reallocation);
  * the grammar validate/pad path works from immutable inputs as well.

Negative control: `test_immutable_draft_ids_truncation_and_write_back` MUST
fail with AttributeError when run against the unpatched
vllm/v1/core/sched/scheduler.py (e.g. after
`git stash push -- vllm/v1/core/sched/scheduler.py`).
"""

from collections.abc import Sequence
from types import SimpleNamespace

import pytest

from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.outputs import DraftTokenIds

REQ_ID = "req-0"


class ImmutableTokenSeq(Sequence):
    """Immutable integer sequence mimicking runner-side draft-id containers.

    Reads (len/index/slice) work; any mutation attempt raises AttributeError,
    the signature measured in the 2026-08-28 incident traceback.
    """

    def __init__(self, items):
        self._items = tuple(items)

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            return ImmutableTokenSeq(self._items[idx])
        return self._items[idx]

    def __len__(self):
        return len(self._items)

    def __delitem__(self, idx):
        raise AttributeError(
            "'ImmutableTokenSeq' object is immutable (no item deletion)"
        )

    def __eq__(self, other):
        if isinstance(other, ImmutableTokenSeq):
            return self._items == other._items
        return NotImplemented

    def __hash__(self):
        return hash(self._items)


class _FakeGrammar:
    """Grammar stub recording the type it received from the scheduler."""

    def __init__(self, result):
        self.result = result
        self.received = None

    def validate_tokens(self, tokens):
        self.received = tokens
        return self.result


def _make_fake_scheduler(should_advance: bool = False, grammar=None):
    request = SimpleNamespace(
        is_finished=lambda: False,
        structured_output_request=SimpleNamespace(grammar=grammar),
    )
    fake_scheduler = SimpleNamespace(
        requests={REQ_ID: request},
        structured_output_manager=SimpleNamespace(
            should_advance=lambda req: should_advance
        ),
    )
    return fake_scheduler


def _make_scheduler_output(num_placeholder_spec_tokens: int):
    # Placeholder ids as produced by the async-scheduling path; their length
    # is the scheduled number of spec tokens.
    return SimpleNamespace(
        scheduled_spec_decode_tokens={
            REQ_ID: [-1] * num_placeholder_spec_tokens,
        },
        num_invalid_spec_tokens={},
    )


def _run(fake_scheduler, scheduler_output, draft_ids):
    Scheduler.update_draft_token_ids_in_output(
        fake_scheduler,
        DraftTokenIds(req_ids=[REQ_ID], draft_token_ids=[draft_ids]),
        scheduler_output,
    )


def test_immutable_draft_ids_truncation_and_write_back():
    """NEGATIVE CONTROL vs unpatched source (fails there w/ AttributeError).

    Immutable draft sequence, truncation required (4 drafts, 2 scheduled):
    must not raise, and the truncated MUTABLE list must be written back to
    scheduler_output.scheduled_spec_decode_tokens where downstream reads it.
    """
    fake_scheduler = _make_fake_scheduler(should_advance=False)
    scheduler_output = _make_scheduler_output(num_placeholder_spec_tokens=2)

    _run(fake_scheduler, scheduler_output, ImmutableTokenSeq([5, 6, 7, 8]))

    stored = scheduler_output.scheduled_spec_decode_tokens[REQ_ID]
    assert stored == [5, 6]
    assert isinstance(stored, list), (
        "truncated draft ids must be materialized as a mutable list in the "
        "scheduler output"
    )
    assert scheduler_output.num_invalid_spec_tokens == {}


def test_tuple_draft_ids_truncation_and_write_back():
    """A plain tuple (the other immutable shape) must also survive."""
    fake_scheduler = _make_fake_scheduler(should_advance=False)
    scheduler_output = _make_scheduler_output(num_placeholder_spec_tokens=3)

    _run(fake_scheduler, scheduler_output, (11, 12, 13, 14, 15))

    stored = scheduler_output.scheduled_spec_decode_tokens[REQ_ID]
    assert stored == [11, 12, 13]
    assert isinstance(stored, list)


def test_plain_list_path_unchanged_in_place():
    """Control: mutable list drafts still truncate in place (no copy)."""
    fake_scheduler = _make_fake_scheduler(should_advance=False)
    scheduler_output = _make_scheduler_output(num_placeholder_spec_tokens=2)

    drafts = [5, 6, 7, 8]
    _run(fake_scheduler, scheduler_output, drafts)

    stored = scheduler_output.scheduled_spec_decode_tokens[REQ_ID]
    assert stored == [5, 6]
    # Same object: the hot path stays allocation-free for list inputs.
    assert stored is drafts
    assert drafts == [5, 6]


def test_plain_list_no_truncation_needed_is_noop():
    """Control: list already at the scheduled length is untouched."""
    fake_scheduler = _make_fake_scheduler(should_advance=False)
    scheduler_output = _make_scheduler_output(num_placeholder_spec_tokens=2)

    drafts = [5, 6]
    _run(fake_scheduler, scheduler_output, drafts)

    stored = scheduler_output.scheduled_spec_decode_tokens[REQ_ID]
    assert stored is drafts
    assert stored == [5, 6]
    assert scheduler_output.num_invalid_spec_tokens == {}


def test_short_immutable_draft_ids_are_padded():
    """Immutable drafts shorter than scheduled get -1 padding, mutably."""
    fake_scheduler = _make_fake_scheduler(should_advance=False)
    scheduler_output = _make_scheduler_output(num_placeholder_spec_tokens=4)

    _run(fake_scheduler, scheduler_output, ImmutableTokenSeq([5, 6]))

    stored = scheduler_output.scheduled_spec_decode_tokens[REQ_ID]
    assert stored == [5, 6, -1, -1]
    assert isinstance(stored, list)
    assert scheduler_output.num_invalid_spec_tokens == {REQ_ID: 2}


def test_immutable_draft_ids_grammar_validation_and_padding():
    """Structured-output path: immutable drafts -> grammar filter -> pad.

    Also proves the grammar backend receives a materialized list (never the
    immutable runner sequence).
    """
    grammar = _FakeGrammar(result=[5])  # accepts only the first draft token
    fake_scheduler = _make_fake_scheduler(should_advance=True, grammar=grammar)
    scheduler_output = _make_scheduler_output(num_placeholder_spec_tokens=2)

    _run(fake_scheduler, scheduler_output, ImmutableTokenSeq([5, 9, 7, 8]))

    assert isinstance(grammar.received, list), (
        "grammar.validate_tokens must be handed a mutable list"
    )
    assert grammar.received == [5, 9]
    stored = scheduler_output.scheduled_spec_decode_tokens[REQ_ID]
    assert stored == [5, -1]
    assert scheduler_output.num_invalid_spec_tokens == {REQ_ID: 1}


def test_grammar_backend_returning_immutable_sequence_is_padded():
    """Even if a grammar backend hands back an immutable prefix, the pad
    (`.extend`) path must not mutate it in place."""
    grammar = _FakeGrammar(result=(5,))  # tuple prefix from the backend
    fake_scheduler = _make_fake_scheduler(should_advance=True, grammar=grammar)
    scheduler_output = _make_scheduler_output(num_placeholder_spec_tokens=2)

    _run(fake_scheduler, scheduler_output, ImmutableTokenSeq([5, 9]))

    stored = scheduler_output.scheduled_spec_decode_tokens[REQ_ID]
    assert stored == [5, -1]
    assert isinstance(stored, list)
    assert scheduler_output.num_invalid_spec_tokens == {REQ_ID: 1}


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
