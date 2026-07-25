# SPDX-License-Identifier: Apache-2.0

"""How the patched AutoTokenizer decides which .gguf backs the tokenizer.

transformers resolves a GGUF tokenizer as (repo_or_dir, gguf_file); it cannot
take a bare .gguf path. The patch restates an exact file in that form, and
only falls back to scanning a directory when no specific file is known. Naming
the wrong file does not fail loudly — it builds a tokenizer from an unrelated
model — so these tests pin the selection down.
"""

import pytest
import transformers

import vllm_gguf_plugin.gemma4 as gemma4


@pytest.fixture
def recorded_from_pretrained(monkeypatch):
    """Install the tokenizer patch over a recorder and return the call log."""
    calls = []

    def recorder(pretrained_model_name_or_path, *args, **kwargs):
        calls.append((str(pretrained_model_name_or_path), kwargs))
        return object()

    monkeypatch.setattr(
        transformers.AutoTokenizer, "from_pretrained", recorder, raising=True
    )
    monkeypatch.setattr(gemma4, "_TOKENIZER_PATCHED", False, raising=True)
    gemma4._patch_gemma4_tokenizer()
    monkeypatch.setattr(gemma4, "_TOKENIZER_PATCHED", False, raising=True)
    return calls


def _write_gguf(path):
    path.write_bytes(b"GGUF")
    return path


def test_exact_gguf_path_is_restated_as_dir_plus_gguf_file(
    tmp_path, recorded_from_pretrained
):
    qwen = _write_gguf(tmp_path / "Qwen3.6-27B-UD-Q4_K_XL.gguf")
    # Sorts ahead of the Qwen file ('L' < 'Q', uppercase before lowercase),
    # so a directory scan would pick it.
    _write_gguf(tmp_path / "Laguna-XS-2.1-Q4_K_M.gguf")

    transformers.AutoTokenizer.from_pretrained(str(qwen))

    path, kwargs = recorded_from_pretrained[0]
    assert path == str(tmp_path)
    assert kwargs["gguf_file"] == "Qwen3.6-27B-UD-Q4_K_XL.gguf"


def test_explicit_gguf_file_kwarg_is_preserved(tmp_path, recorded_from_pretrained):
    _write_gguf(tmp_path / "Qwen3.6-27B-UD-Q4_K_XL.gguf")
    _write_gguf(tmp_path / "Laguna-XS-2.1-Q4_K_M.gguf")

    transformers.AutoTokenizer.from_pretrained(
        str(tmp_path), gguf_file="Qwen3.6-27B-UD-Q4_K_XL.gguf"
    )

    _, kwargs = recorded_from_pretrained[0]
    assert kwargs["gguf_file"] == "Qwen3.6-27B-UD-Q4_K_XL.gguf"


def test_directory_scan_skips_mmproj_and_mtp_sidecars(
    tmp_path, recorded_from_pretrained
):
    _write_gguf(tmp_path / "Qwen3.6-27B-UD-Q4_K_XL.gguf")
    # Both sort ahead of the real weights but are sidecars, not models.
    _write_gguf(tmp_path / "mmproj-Qwen3.6-27B.gguf")
    _write_gguf(tmp_path / "Qwen3.6-27B-MTP-UD-Q4_K_XL.gguf")

    transformers.AutoTokenizer.from_pretrained(str(tmp_path))

    _, kwargs = recorded_from_pretrained[0]
    assert kwargs["gguf_file"] == "Qwen3.6-27B-UD-Q4_K_XL.gguf"


def test_ambiguous_directory_warns(tmp_path, recorded_from_pretrained, caplog):
    _write_gguf(tmp_path / "Qwen3.6-27B-UD-Q4_K_XL.gguf")
    _write_gguf(tmp_path / "Laguna-XS-2.1-Q4_K_M.gguf")

    with caplog.at_level("WARNING"):
        transformers.AutoTokenizer.from_pretrained(str(tmp_path))

    assert "Point at the specific .gguf file" in caplog.text


def test_directory_with_hf_tokenizer_files_is_left_alone(
    tmp_path, recorded_from_pretrained
):
    """A real HF tokenizer in the dir wins; no gguf_file is injected."""
    _write_gguf(tmp_path / "Qwen3.6-27B-UD-Q4_K_XL.gguf")
    (tmp_path / "tokenizer.json").write_text("{}")

    transformers.AutoTokenizer.from_pretrained(str(tmp_path))

    path, kwargs = recorded_from_pretrained[0]
    assert path == str(tmp_path)
    assert "gguf_file" not in kwargs
