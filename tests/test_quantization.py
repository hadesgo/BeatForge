from types import SimpleNamespace
import sys

from beatforge.models.quantization import quantized_load_kwargs


def test_cpu_never_imports_or_enables_gpu_quantization() -> None:
    assert quantized_load_kwargs("nf4", SimpleNamespace(), "cpu") == {}


def test_nf4_uses_bfloat16_and_nested_quantization(monkeypatch) -> None:
    captured = {}

    class Config:
        def __init__(self, **options):
            captured.update(options)

    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(BitsAndBytesConfig=Config))
    torch = SimpleNamespace(bfloat16="bf16")

    result = quantized_load_kwargs("nf4", torch, "cuda")

    assert isinstance(result["quantization_config"], Config)
    assert captured == {
        "load_in_4bit": True,
        "bnb_4bit_quant_type": "nf4",
        "bnb_4bit_compute_dtype": "bf16",
        "bnb_4bit_use_double_quant": True,
    }


def test_int8_mode_is_available(monkeypatch) -> None:
    class Config:
        def __init__(self, **options):
            self.options = options

    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(BitsAndBytesConfig=Config))
    result = quantized_load_kwargs("int8", SimpleNamespace(), "cuda")
    assert result["quantization_config"].options == {"load_in_8bit": True}
