# SPDX-License-Identifier: Apache-2.0

"""The added-token vocabulary of a GGUF-derived tokenizer.

Two defects, opposite in sign. Tokens the converters **drop**: a GGUF's
``tokenizer.ggml.token_type`` array marks entries CONTROL (3) or
USER_DEFINED (4) — what HF stores in ``added_tokens_decoder`` — and
``GGUFQwen2Converter`` ignores it, registering three hardcoded ChatML strings
instead. The rest stay in the BPE vocab, so decoding works and only
``encode()`` is wrong. For Qwen3.5/3.6 that silently breaks chat: the
template prefills ``<think>``, the model receives ``<th|ink|>`` where one
token belongs, and answers with an immediate end-of-turn — an empty response
with no error.

Tokens the converters **invent**: bos/eos come from the GGUF as ids only, so
the backend supplies ``<s>``/``</s>`` from its class defaults and appends
them past ``vocab_size``. A prompt containing ``<s>`` then indexes off the
embedding matrix and kills the engine.
"""

import pytest
from gguf import TokenType
from tokenizers import AddedToken

from vllm_gguf_plugin.tokenizer import (
    added_tokens_from_gguf_vocab,
    gguf_special_token_kwargs,
    restore_gguf_added_tokens,
    special_token_kwargs_from_vocab,
)


class _FakeTokenizer:
    """Records add_tokens calls and resolves ids from a fixed vocab."""

    def __init__(self, vocab: dict[str, int]):
        self._vocab = vocab
        self.added: list[AddedToken] = []

    def convert_tokens_to_ids(self, token: str) -> int | None:
        return self._vocab.get(token)

    def add_tokens(self, tokens: list[AddedToken]) -> int:
        self.added.extend(tokens)
        return len(tokens)


def _vocab(*names: str) -> dict[str, int]:
    return {name: i for i, name in enumerate(names)}


# --- selecting which tokens to re-register --------------------------------


def test_control_tokens_are_registered_as_special():
    picked = added_tokens_from_gguf_vocab(
        ["a", "<|im_end|>"], [TokenType.NORMAL, TokenType.CONTROL]
    )
    assert [i for i, _ in picked] == [1]
    assert picked[0][1].content == "<|im_end|>"
    assert picked[0][1].special is True


def test_user_defined_tokens_are_registered_as_non_special():
    """HF marks <think>/</think> special=False, so they survive decoding with
    skip_special_tokens=True. CONTROL vs USER_DEFINED must map onto that."""
    picked = added_tokens_from_gguf_vocab(
        ["a", "<think>"], [TokenType.NORMAL, TokenType.USER_DEFINED]
    )
    assert [i for i, _ in picked] == [1]
    assert picked[0][1].content == "<think>"
    assert picked[0][1].special is False


@pytest.mark.parametrize(
    "token_type",
    [TokenType.NORMAL, TokenType.UNKNOWN, TokenType.UNUSED, TokenType.BYTE],
)
def test_other_token_types_are_left_out(token_type):
    """Only 3 and 4 are added tokens. UNUSED in particular is padding - this
    vocab carries 243 of them, and registering those would bloat the
    added-token trie that every encode() consults."""
    assert added_tokens_from_gguf_vocab(["x"], [token_type]) == []


def test_added_tokens_are_never_normalized():
    """Normalisation would let a lowercased or stripped variant match."""
    picked = added_tokens_from_gguf_vocab(["<think>"], [TokenType.USER_DEFINED])
    assert picked[0][1].normalized is False


def test_ragged_type_array_is_rejected():
    with pytest.raises(ValueError, match="248 tokens.*2 token types"):
        added_tokens_from_gguf_vocab(["a"] * 248, [TokenType.NORMAL] * 2)


# --- applying them to a tokenizer -----------------------------------------


def test_restore_registers_dropped_tokens(monkeypatch):
    tok = _FakeTokenizer(_vocab("a", "<|im_end|>", "<think>"))
    monkeypatch.setattr(
        "vllm_gguf_plugin.tokenizer._read_gguf_vocab",
        lambda _: (
            ["a", "<|im_end|>", "<think>"],
            [TokenType.NORMAL, TokenType.CONTROL, TokenType.USER_DEFINED],
        ),
    )

    assert restore_gguf_added_tokens(tok, "model.gguf") == 2
    assert [t.content for t in tok.added] == ["<|im_end|>", "<think>"]


def test_token_at_a_different_id_is_skipped(monkeypatch, caplog):
    """The added token must resolve to the id the GGUF gave it. If the
    converter placed it elsewhere, re-registering would either allocate a
    fresh id past vocab_size — indexing off the end of the embedding matrix —
    or point the string at another row entirely."""
    tok = _FakeTokenizer({"a": 0, "<think>": 7})
    monkeypatch.setattr(
        "vllm_gguf_plugin.tokenizer._read_gguf_vocab",
        lambda _: (["a", "<think>"], [TokenType.NORMAL, TokenType.USER_DEFINED]),
    )

    with caplog.at_level("WARNING"):
        assert restore_gguf_added_tokens(tok, "model.gguf") == 0
    assert tok.added == []
    assert "<think>" in caplog.text


def test_token_missing_from_the_vocab_is_skipped(monkeypatch, caplog):
    tok = _FakeTokenizer({"a": 0})
    monkeypatch.setattr(
        "vllm_gguf_plugin.tokenizer._read_gguf_vocab",
        lambda _: (["a", "<think>"], [TokenType.NORMAL, TokenType.USER_DEFINED]),
    )

    with caplog.at_level("WARNING"):
        assert restore_gguf_added_tokens(tok, "model.gguf") == 0
    assert tok.added == []


def test_nothing_to_add_leaves_the_tokenizer_untouched(monkeypatch):
    tok = _FakeTokenizer(_vocab("a", "b"))
    monkeypatch.setattr(
        "vllm_gguf_plugin.tokenizer._read_gguf_vocab",
        lambda _: (["a", "b"], [TokenType.NORMAL, TokenType.NORMAL]),
    )

    assert restore_gguf_added_tokens(tok, "model.gguf") == 0
    assert tok.added == []


def test_gguf_without_a_token_type_array_is_a_noop(monkeypatch):
    """Not every GGUF carries token_type; absence must not fail the load."""
    monkeypatch.setattr("vllm_gguf_plugin.tokenizer._read_gguf_vocab", lambda _: None)
    tok = _FakeTokenizer(_vocab("a"))

    assert restore_gguf_added_tokens(tok, "model.gguf") == 0
    assert tok.added == []


def test_unreadable_gguf_does_not_break_tokenizer_construction(monkeypatch, caplog):
    """This runs inside AutoTokenizer.from_pretrained. A tokenizer that
    encodes <think> as three tokens still mostly works; one that fails to
    build does not."""

    def _boom(_):
        raise OSError("truncated file")

    monkeypatch.setattr("vllm_gguf_plugin.tokenizer._read_gguf_vocab", _boom)
    tok = _FakeTokenizer(_vocab("a"))

    with caplog.at_level("WARNING"):
        assert restore_gguf_added_tokens(tok, "model.gguf") == 0
    assert "truncated file" in caplog.text


def test_truncated_gguf_on_disk_is_survivable(tmp_path):
    """GGUFReader has no typed failure contract - a stub file raises
    IndexError out of its header parse, not the ValueError its own magic
    check would give. Exercises the real reader, no monkeypatch."""
    stub = tmp_path / "model.gguf"
    stub.write_bytes(b"GGUF")

    assert restore_gguf_added_tokens(_FakeTokenizer(_vocab("a")), stub) == 0


def test_restore_is_idempotent(monkeypatch):
    """register() may run more than once per process, and re-adding a token
    that is already in the added vocab must not shift any id."""
    tok = _FakeTokenizer(_vocab("a", "<think>"))
    monkeypatch.setattr(
        "vllm_gguf_plugin.tokenizer._read_gguf_vocab",
        lambda _: (["a", "<think>"], [TokenType.NORMAL, TokenType.USER_DEFINED]),
    )

    first = restore_gguf_added_tokens(tok, "model.gguf")
    second = restore_gguf_added_tokens(tok, "model.gguf")
    assert first == second == 1
    assert [t.content for t in tok.added] == ["<think>", "<think>"]


# --- bos/eos strings, so the backend's own defaults never mint a token ----
#
# A GGUF declares bos/eos as *ids*, never strings. TokenizersBackend fills
# the missing strings from its class defaults, "<s>" and "</s>", and setting
# a special token that is not in the vocab appends it — landing at
# vocab_size and vocab_size + 1, past the last embedding row. Those ids are
# reachable from ordinary text: a prompt containing "<s>" encodes to one and
# takes down the engine with a device-side assert. Passing the real strings
# up front means the defaults never apply.


def test_bos_and_eos_ids_resolve_to_their_vocab_strings():
    kwargs = special_token_kwargs_from_vocab(
        ["a", "<|endoftext|>", "<|im_end|>"], bos_id=1, eos_id=2
    )
    assert kwargs == {"bos_token": "<|endoftext|>", "eos_token": "<|im_end|>"}


def test_absent_ids_are_omitted_rather_than_guessed():
    assert special_token_kwargs_from_vocab(["a"], bos_id=None, eos_id=None) == {}


@pytest.mark.parametrize("bad_id", [-1, 5, 99])
def test_out_of_range_ids_are_omitted(bad_id):
    """A declared id past the vocab cannot name a string, and inventing one
    would recreate exactly the bug this prevents."""
    assert special_token_kwargs_from_vocab(["a", "b"], bos_id=bad_id, eos_id=None) == {}


def test_only_the_declared_end_is_returned():
    kwargs = special_token_kwargs_from_vocab(["a", "<|im_end|>"], bos_id=None, eos_id=1)
    assert kwargs == {"eos_token": "<|im_end|>"}


def test_special_token_kwargs_reads_the_gguf(monkeypatch):
    monkeypatch.setattr(
        "vllm_gguf_plugin.tokenizer._read_gguf_special_tokens",
        lambda _: (["a", "<|endoftext|>", "<|im_end|>"], 1, 2),
    )
    assert gguf_special_token_kwargs("model.gguf") == {
        "bos_token": "<|endoftext|>",
        "eos_token": "<|im_end|>",
    }


def test_unreadable_gguf_yields_no_kwargs(monkeypatch, caplog):
    def _boom(_):
        raise OSError("truncated file")

    monkeypatch.setattr("vllm_gguf_plugin.tokenizer._read_gguf_special_tokens", _boom)
    with caplog.at_level("WARNING"):
        assert gguf_special_token_kwargs("model.gguf") == {}


def test_truncated_gguf_on_disk_yields_no_kwargs(tmp_path):
    stub = tmp_path / "model.gguf"
    stub.write_bytes(b"GGUF")

    assert gguf_special_token_kwargs(stub) == {}
