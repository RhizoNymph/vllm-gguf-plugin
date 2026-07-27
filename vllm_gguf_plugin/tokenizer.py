# SPDX-License-Identifier: Apache-2.0

"""Repair the added-token vocabulary of a GGUF-derived tokenizer.

Two defects, opposite in sign, both in what ``encode()`` matches verbatim:
tokens the converters *drop* and tokens they *invent*.

**Dropped.** A GGUF vocab tags every entry in ``tokenizer.ggml.token_type``.
CONTROL (3) and USER_DEFINED (4) are the two kinds HF keeps in
``added_tokens_decoder``: strings matched verbatim, ahead of BPE, so they
always encode to one id.

transformers largely ignores that array. ``GGUFLlamaConverter`` reads it but
only honours CONTROL; the converters this plugin routes through —
``GGUFQwen2Converter`` for qwen35/qwen35moe, ``GGUFGemmaConverter`` for
gemma4 — never consult it at all. Qwen2 hard-codes three ChatML strings and
Gemma adds none, so everything else falls through to BPE.

The dropped tokens are still in the BPE vocab, which is what makes this so
quiet: ``convert_ids_to_tokens`` and ``decode`` behave, and only ``encode``
is wrong. On Qwen3.5/3.6 the chat template prefills ``<think>``; the model
gets ``<th|ink|>`` where one token belongs and replies with an immediate
end-of-turn, so chat returns an empty string with no error anywhere.

The non-GGUF SentencePiece path already does this correctly — see
``transformers.convert_slow_tokenizer``, which selects ``p.type in [3, 4]``
and marks special as ``p.type == 3``. This module applies that same rule to
the GGUF path.

**Invented.** A GGUF declares bos/eos as ids and never as strings, so the
backend tokenizer class fills the strings in from its own defaults, ``<s>``
and ``</s>``. Absent from a Qwen vocab, setting them *appends* them at
``vocab_size`` and ``vocab_size + 1`` — past the last embedding row. Nothing
generates them, but a prompt containing the literal text ``<s>`` encodes to
one and takes the engine down with a device-side assert. Naming bos/eos from
the GGUF before construction stops the defaults from ever applying.

Both repairs share one invariant: **never change the vocabulary.** Added
tokens are only registered at ids the tokenizer already resolves them to, and
bos/eos are only named from ids that exist, so nothing can be minted past
``vocab_size``.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import gguf
from gguf import Keys, TokenType
from tokenizers import AddedToken
from vllm.logger import init_logger

logger = init_logger(__name__)

# The token types HF represents as added tokens. CONTROL is structural
# (<|im_start|>) and hidden by skip_special_tokens; USER_DEFINED is content
# the caller is meant to see (<think>), so it stays in decoded text.
_ADDED_TOKEN_TYPES = (TokenType.CONTROL, TokenType.USER_DEFINED)


def added_tokens_from_gguf_vocab(
    tokens: Sequence[str], token_types: Sequence[int]
) -> list[tuple[int, AddedToken]]:
    """Pick the vocab entries that belong in the added-token vocabulary.

    Returns ``(vocab_id, AddedToken)`` pairs. The id is the token's index in
    the GGUF vocab, which callers must check against the built tokenizer
    before registering anything.
    """
    if len(tokens) != len(token_types):
        raise ValueError(
            f"GGUF vocab is inconsistent: {len(tokens)} tokens but "
            f"{len(token_types)} token types"
        )

    return [
        (
            index,
            AddedToken(
                token,
                normalized=False,
                special=token_type == TokenType.CONTROL,
            ),
        )
        for index, (token, token_type) in enumerate(zip(tokens, token_types))
        if token_type in _ADDED_TOKEN_TYPES
    ]


def _read_gguf_vocab(
    gguf_path: str | Path,
) -> tuple[list[str], list[int]] | None:
    """Read the vocab and its type tags, or None if either is absent."""
    reader = gguf.GGUFReader(str(gguf_path))
    tokens = reader.get_field(Keys.Tokenizer.LIST)
    token_types = reader.get_field(Keys.Tokenizer.TOKEN_TYPE)
    if tokens is None or token_types is None:
        return None
    return tokens.contents(), token_types.contents()


def special_token_kwargs_from_vocab(
    tokens: Sequence[str], bos_id: int | None, eos_id: int | None
) -> dict[str, str]:
    """Name the bos/eos tokens a GGUF declares only by id.

    Ids outside the vocab are dropped rather than guessed — naming a token
    that does not exist is precisely what creates a phantom.
    """

    def named(token_id: int | None) -> str | None:
        if token_id is None or not 0 <= token_id < len(tokens):
            return None
        return tokens[token_id]

    return {
        key: name
        for key, name in (
            ("bos_token", named(bos_id)),
            ("eos_token", named(eos_id)),
        )
        if name is not None
    }


def _read_gguf_special_tokens(
    gguf_path: str | Path,
) -> tuple[list[str], int | None, int | None] | None:
    """Read the vocab plus the declared bos/eos ids."""
    reader = gguf.GGUFReader(str(gguf_path))
    tokens = reader.get_field(Keys.Tokenizer.LIST)
    if tokens is None:
        return None

    def scalar(key: str) -> int | None:
        field = reader.get_field(key)
        if field is None:
            return None
        return int(field.parts[field.data[0]].tolist()[0])

    return (
        tokens.contents(),
        scalar(Keys.Tokenizer.BOS_ID),
        scalar(Keys.Tokenizer.EOS_ID),
    )


def gguf_special_token_kwargs(gguf_path: str | Path) -> dict[str, str]:
    """``from_pretrained`` kwargs pinning bos/eos to the GGUF's own tokens.

    A GGUF stores bos/eos as ids and never as strings, so the backend
    tokenizer class supplies the missing strings from its own defaults —
    ``<s>`` and ``</s>``. Neither is in a Qwen vocab, and setting a special
    token that is absent *appends* it, so they land at ``vocab_size`` and
    ``vocab_size + 1``: one and two rows past the end of the embedding
    matrix.

    Nothing generates them, which is why this hid for so long, but they are
    reachable from input. A prompt containing the literal text ``<s>``
    encodes to that id and the embedding lookup fails a device-side assert,
    killing the engine core. Supplying the real strings up front means the
    class defaults never apply, and fixes ``bos_token_id``/``eos_token_id``
    at the same time — they otherwise point at the phantoms.

    Returns an empty dict rather than raising; see
    :func:`restore_gguf_added_tokens` for why this path must not fail a load.
    """
    try:
        declared = _read_gguf_special_tokens(gguf_path)
    except Exception as exc:
        # Broad for the same reason as the vocab read: GGUFReader has no
        # typed failure contract.
        logger.warning("could not read GGUF special tokens from %s: %s", gguf_path, exc)
        return {}

    if declared is None:
        return {}

    tokens, bos_id, eos_id = declared
    return special_token_kwargs_from_vocab(tokens, bos_id, eos_id)


def restore_gguf_added_tokens(tokenizer, gguf_path: str | Path) -> int:
    """Re-register a GGUF's CONTROL/USER_DEFINED tokens on ``tokenizer``.

    Only tokens the tokenizer already resolves to their GGUF id are added.
    That keeps the operation incapable of changing the vocab: ``add_tokens``
    reuses an existing entry rather than allocating, so no id can land past
    ``vocab_size`` — which would index off the end of the embedding matrix.

    Returns the number of tokens registered. Never raises: this runs inside
    ``AutoTokenizer.from_pretrained``, and a tokenizer that splits ``<think>``
    is still far more useful than a model that refuses to load.
    """
    try:
        vocab = _read_gguf_vocab(gguf_path)
    except Exception as exc:
        # Deliberately broad: GGUFReader has no typed failure contract. A
        # truncated file surfaces as IndexError from its header parse, a bad
        # magic as ValueError, an unreadable one as OSError. Narrowing this
        # would turn "this file isn't a usable GGUF" into a failed model load.
        logger.warning("could not read GGUF vocab from %s: %s", gguf_path, exc)
        return 0

    if vocab is None:
        return 0

    try:
        candidates = added_tokens_from_gguf_vocab(*vocab)
    except ValueError as exc:
        logger.warning("skipping added-token restore for %s: %s", gguf_path, exc)
        return 0

    keep: list[AddedToken] = []
    misplaced: list[str] = []
    for vocab_id, added in candidates:
        if tokenizer.convert_tokens_to_ids(added.content) == vocab_id:
            keep.append(added)
        else:
            misplaced.append(added.content)

    if misplaced:
        # Re-registering these would either mint a new id past the embedding
        # matrix or bind the string to some other row, so they are dropped.
        logger.warning(
            "%d GGUF added token(s) are absent or at a different id in the "
            "converted tokenizer and were skipped: %s",
            len(misplaced),
            ", ".join(sorted(misplaced)[:10]),
        )

    if not keep:
        return 0

    tokenizer.add_tokens(keep)
    logger.debug(
        "restored %d GGUF added token(s) dropped by the transformers converter: %s",
        len(keep),
        ", ".join(t.content for t in keep[:10]),
    )
    return len(keep)
