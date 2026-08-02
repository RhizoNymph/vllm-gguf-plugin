# SPDX-License-Identifier: Apache-2.0

import os
import pathlib
import sys

import tomllib
from setuptools import setup


def _package_version() -> str:
    project = tomllib.loads(pathlib.Path("pyproject.toml").read_text())
    version = project["tool"]["vllm_gguf_plugin"]["base_version"]
    suffix = os.environ.get("VLLM_GGUF_PLUGIN_LOCAL_VERSION_SUFFIX")
    if not suffix:
        return version
    normalized_suffix = suffix if suffix.startswith("+") else f"+{suffix}"
    return f"{version}{normalized_suffix}"


def _should_build_extension() -> bool:
    packaging_commands = {"sdist", "egg_info", "dist_info"}
    return not any(command in packaging_commands for command in sys.argv[1:])


setup_kwargs: dict = {"version": _package_version()}

if _should_build_extension():
    import torch
    from torch.utils.cpp_extension import BuildExtension, CUDAExtension

    is_rocm = getattr(torch.version, "hip", None) is not None

    nvcc_args = [
        "-O3",
        "-std=c++17",
        # Exposes aoti_torch_get_current_cuda_stream in the AOTI shim.
        "-DUSE_CUDA",
        # torch's CUDAExtension injects -D__CUDA_NO_HALF_CONVERSIONS__ and
        # friends, which remove the implicit float<->__half conversions and the
        # __half2/__nv_bfloat162 constructors. The vendored llama.cpp kernels
        # (llamacpp/) are written against stock CUDA and rely on those, so they
        # fail to compile with torch's defaults. Undefining only re-enables
        # conversions that stock nvcc allows; it cannot invalidate code that
        # already compiled without them.
        "-U__CUDA_NO_HALF_OPERATORS__",
        "-U__CUDA_NO_HALF_CONVERSIONS__",
        "-U__CUDA_NO_HALF2_OPERATORS__",
        "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
        "-U__CUDA_NO_BFLOAT16_OPERATORS__",
        "-U__CUDA_NO_BFLOAT162_OPERATORS__",
    ]
    if not is_rocm:
        # hipcc (ROCm 7.x) rejects nvcc-only flags like --use_fast_math.
        nvcc_args.insert(2, "--use_fast_math")

    setup_kwargs.update(
        ext_modules=[
            CUDAExtension(
                name="vllm_gguf_plugin._C_gguf",
                sources=[
                    "vllm_gguf_plugin/csrc/torch_bindings.cpp",
                    "vllm_gguf_plugin/csrc/gguf/gguf_kernel.cu",
                    # Vendored llama.cpp MMA (tensor-core) MMQ. Kept in separate
                    # translation units: llamacpp/ggml-common.h and the plugin's
                    # own ggml-common.h are different vintages of the same header
                    # and redefine the same block structs, so they must never
                    # meet in one TU.
                    "vllm_gguf_plugin/csrc/gguf/mmq_mma.cu",
                    "vllm_gguf_plugin/csrc/gguf/mmq_mma_quantize.cu",
                    "vllm_gguf_plugin/csrc/gguf/mmq_mma_shim.cu",
                    "vllm_gguf_plugin/csrc/gguf/llamacpp/quantize.cu",
                ]
                + sorted(
                    str(p)
                    for p in pathlib.Path(
                        "vllm_gguf_plugin/csrc/gguf/llamacpp/template-instances"
                    ).glob("mmq-instance-*.cu")
                ),
                include_dirs=[
                    "vllm_gguf_plugin/csrc",
                    "vllm_gguf_plugin/csrc/gguf",
                    "vllm_gguf_plugin/csrc/gguf/llamacpp",
                ],
                py_limited_api=True,
                extra_compile_args={
                    "cxx": ["-O3", "-std=c++17"],
                    "nvcc": nvcc_args,
                },
            )
        ],
        cmdclass={"build_ext": BuildExtension},
        options={"bdist_wheel": {"py_limited_api": "cp310"}},
    )

setup(**setup_kwargs)
