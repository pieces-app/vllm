"""Grammar bitmask must constrain speculative positions after a -1 draft (issue #152).

Root cause (grammar_bitmask, StructuredOutputManager): a -1 draft is an
invalid/pad speculative token. The loop correctly skips the FSM advance for it
(advance_grammar=False), but it USED TO also latch apply_bitmask=False, which
filled every SUBSEQUENT speculative row with _full_mask (all-tokens-allowed).
Those rows are the target model's recovery-token positions; unconstrained, the
target could sample an out-of-grammar token that then tripped the xgrammar FSM
(measured: 1 of 16 constrained MTP requests died, BENCH_RESULTS.md A3).

Two tests:

* test_minus_one_block_does_not_disable_masking -- dependency-free AST guard
  (runs on any CPython; the arm fork_gate reverts): the `if token == -1:` block
  sets advance_grammar (skip the FSM advance) but must NOT assign apply_bitmask
  (which would re-arm the all-allow regression). Negative control: re-add
  `apply_bitmask = False` -> red.

* test_spec_positions_after_minus_one_stay_constrained -- behavioral, needs the
  full vllm stack (guidance backend + tokenizer); runs in the in-image CPU
  gate. Builds a real StructuredOutputManager with a spec-decode config, feeds
  spec tokens containing a -1, and asserts the bitmask rows after the -1 are
  NOT all-allow (_full_mask).
"""
import ast
from pathlib import Path

import pytest

# vllm/v1/structured_output/__init__.py, four parents up from this test file.
SO_INIT = (Path(__file__).resolve().parents[3] / "vllm" / "v1" /
           "structured_output" / "__init__.py")


def _is_neg_one(node: ast.expr) -> bool:
    return (isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub)
            and isinstance(node.operand, ast.Constant)
            and node.operand.value == 1) or (
                isinstance(node, ast.Constant) and node.value == -1)


def _minus_one_block() -> ast.If:
    tree = ast.parse(SO_INIT.read_text())
    fn = next(n for n in ast.walk(tree)
              if isinstance(n, ast.FunctionDef) and n.name == "grammar_bitmask")
    for node in ast.walk(fn):
        if (isinstance(node, ast.If) and isinstance(node.test, ast.Compare)
                and isinstance(node.test.left, ast.Name)
                and node.test.left.id == "token"
                and any(_is_neg_one(c) for c in node.test.comparators)):
            return node
    raise AssertionError("no `if token == -1:` block in grammar_bitmask")


def _assigns(block: ast.If, name: str) -> bool:
    return any(isinstance(n, ast.Assign)
               and any(isinstance(t, ast.Name) and t.id == name
                       for t in n.targets)
               for n in block.body)


def test_minus_one_block_does_not_disable_masking():
    block = _minus_one_block()
    assert _assigns(block, "advance_grammar"), (
        "the -1 block must still skip the FSM advance (advance_grammar=False)")
    assert not _assigns(block, "apply_bitmask"), (
        "the -1 block must NOT assign apply_bitmask -- latching it False fills "
        "post--1 speculative rows with the all-allow mask and lets the target "
        "sample out-of-grammar recovery tokens (issue #152).")


@pytest.mark.cpu_test
def test_spec_positions_after_minus_one_stay_constrained():
    pytest.importorskip("torch")
    AutoTokenizer = pytest.importorskip(
        "transformers").AutoTokenizer
    from vllm.config import (ModelConfig, SpeculativeConfig,
                             StructuredOutputsConfig, VllmConfig)
    from vllm.sampling_params import (SamplingParams,
                                      StructuredOutputsParams)
    from vllm.v1.request import Request
    from vllm.v1.structured_output import StructuredOutputManager

    TOKENIZER = "Qwen/Qwen2.5-1.5B-Instruct"
    tok = AutoTokenizer.from_pretrained(TOKENIZER)
    prompt = tok.encode('{"a": "b"}')
    vllm_config = VllmConfig(
        model_config=ModelConfig(tokenizer=TOKENIZER),
        structured_outputs_config=StructuredOutputsConfig(backend="guidance"),
        speculative_config=SpeculativeConfig(model="[ngram]",
                                             num_speculative_tokens=3),
    )
    mgr = StructuredOutputManager(vllm_config)
    sp = SamplingParams(structured_outputs=StructuredOutputsParams(
        json='{"type": "object"}'))
    sp.structured_outputs._backend = "guidance"
    sp.update_from_generation_config({}, tok.eos_token_id)
    req = Request("r0", prompt_token_ids=prompt[:1], sampling_params=sp,
                  pooling_params=None)
    mgr.grammar_init(req)
    while not req.structured_output_request._check_grammar_completion():
        continue
    assert req.structured_output_request.grammar.accept_tokens("r0", prompt[:1])

    # spec tokens: one valid draft then a -1 pad; rows for index >= the -1 are
    # the recovery positions that regressed to all-allow.
    spec = [prompt[1] if len(prompt) > 1 else tok.eos_token_id, -1, -1]
    mgr.grammar_bitmask(
        requests={"r0": req},
        structured_output_request_ids={"r0": 0},
        scheduled_spec_decode_tokens={"r0": spec},
    )
    bm = mgr._grammar_bitmask  # (1+K) rows, int32; _full_mask == -1 (all bits)
    # Rows 2 and 3 (0-based indices 2,3) follow the -1 at index 1. A fully
    # constrained JSON grammar never allows the ENTIRE vocabulary, so an
    # all -1 row means the mask was disabled there.
    for row in (2, 3):
        assert not bool((bm[row] == -1).all()), (
            f"bitmask row {row} (a post--1 recovery position) is all-allow; "
            f"the grammar constraint was dropped (issue #152 regression)")
