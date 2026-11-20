# SPDX-License-Identifier: Apache-2.0
"""The weight-type companion of a quantized tensor renames only its suffix.

Qwen4Exp hyper-connection parameters carry "weight" inside the module name
(``input_mix_weight_down.weight``). Replacing every occurrence yielded
``input_mix_weight_type_down.weight_type``, which matches no parameter.
"""

import numpy as np
import pytest
from gguf import GGMLQuantizationType, GGUFWriter
from gguf.quants import quantize

from vllm_gguf_plugin.weight_utils import gguf_quant_weights_iterator_multi

pytestmark = [pytest.mark.cpu]


def test_only_the_weight_suffix_is_renamed(tmp_path):
    path = tmp_path / "hc.gguf"
    writer = GGUFWriter(str(path), "llama")
    data = np.random.default_rng(0).standard_normal((4, 64)).astype(np.float32)
    qtype = GGMLQuantizationType.Q8_0
    writer.add_tensor(
        "blk.0.hc_attn_down.weight",
        quantize(data, qtype),
        raw_dtype=qtype,
    )
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    hf = "model.layers.0.attn_hyper_connection.input_mix_weight_down.weight"
    names = [
        name
        for name, _ in gguf_quant_weights_iterator_multi(
            [str(path)], {"blk.0.hc_attn_down.weight": hf}
        )
    ]
    assert names == [
        "model.layers.0.attn_hyper_connection.input_mix_weight_down.weight_type",
        hf,
    ]
