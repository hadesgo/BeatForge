from __future__ import annotations

from typing import Literal


QuantizationMode = Literal["none", "int8", "nf4"]


def quantized_load_kwargs(mode: QuantizationMode, torch, device: str) -> dict:
    """Build Transformers load options without importing bitsandbytes on CPU."""
    if device != "cuda" or mode == "none":
        return {}
    try:
        from transformers import BitsAndBytesConfig
    except ImportError as exc:
        raise RuntimeError("量化模式需要 transformers 与 bitsandbytes>=0.50.2") from exc
    if mode == "int8":
        return {"quantization_config": BitsAndBytesConfig(load_in_8bit=True)}
    return {
        "quantization_config": BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        ),
    }
