from __future__ import annotations

import inspect
import unittest
from types import SimpleNamespace

from qsol_geo_reason.capture import CaptureContractError, HuggingFacePyTorchBackend


class FakeContentTensor:
    def __init__(self, content: bytes):
        self._version = 0
        self.shape = (len(content),)
        self.dtype = "torch.uint8"
        self.device = "cpu"
        self.content = bytearray(content)

    def detach(self):
        return self

    def to(self, device):
        if device != "cpu":
            raise AssertionError(device)
        return self

    def contiguous(self):
        return self

    def reshape(self, *shape):
        return self

    def view(self, dtype):
        return self

    def numpy(self):
        return self.content


class FakeContentModel:
    def __init__(self):
        self.weight = FakeContentTensor(b"canonical-weight-bytes")
        self.cache = FakeContentTensor(b"canonical-buffer-bytes")

    def named_parameters(self):
        return [("weight", self.weight)]

    def named_buffers(self):
        return [("cache", self.cache)]


class FakeTokenizerBackend:
    def to_str(self):
        return '{"model":"fake"}'


class FakeTokenizer:
    backend_tokenizer = FakeTokenizerBackend()
    added_tokens_encoder = {}
    special_tokens_map = {}
    all_special_tokens = []
    all_special_ids = []
    init_kwargs = {}

    def get_vocab(self):
        return {"a": 0}


class CaptureRound18RegressionTests(unittest.TestCase):
    def _sealed_backend(self):
        backend = object.__new__(HuggingFacePyTorchBackend)
        backend._torch = SimpleNamespace(uint8=object())
        backend._model = FakeContentModel()
        backend._tokenizer = FakeTokenizer()
        backend._canonical_model_live_state = backend._model_live_state_seal()
        backend._canonical_model_content_state = backend._model_content_state_seal()
        backend._canonical_tokenizer_live_state = backend._tokenizer_live_state_seal()
        backend._live_state_seal_initialized = True
        return backend

    def test_content_seal_rejects_version_bypassing_parameter_mutation(self):
        backend = self._sealed_backend()
        original_version = backend._model.weight._version
        backend._assert_live_state_authentication()

        backend._model.weight.content[0] ^= 0x01
        self.assertEqual(backend._model.weight._version, original_version)
        with self.assertRaisesRegex(CaptureContractError, "tensor contents changed"):
            backend._assert_live_state_authentication()

    def test_content_seal_rejects_version_bypassing_buffer_mutation(self):
        backend = self._sealed_backend()
        original_version = backend._model.cache._version
        backend._model.cache.content[-1] ^= 0x01
        self.assertEqual(backend._model.cache._version, original_version)
        with self.assertRaisesRegex(CaptureContractError, "tensor contents changed"):
            backend._assert_live_state_authentication()

    def test_live_state_seal_hashes_tensor_bytes_and_checks_each_capture(self):
        content_source = inspect.getsource(HuggingFacePyTorchBackend._model_content_state_seal)
        tensor_source = inspect.getsource(HuggingFacePyTorchBackend._tensor_content_sha256)
        auth_sources = [
            inspect.getsource(cls.__dict__["_assert_live_state_authentication"])
            for cls in HuggingFacePyTorchBackend.__mro__
            if "_assert_live_state_authentication" in cls.__dict__
        ]
        capture_source = inspect.getsource(HuggingFacePyTorchBackend.hidden_states)

        self.assertIn("self._tensor_content_sha256", content_source)
        self.assertIn("hashlib.sha256", tensor_source)
        # Exactly one layer owns the content check. Requiring a duplicate in the
        # public wrapper would reintroduce the full-model transfer regression.
        self.assertEqual(
            sum(source.count("self._model_content_state_seal()") for source in auth_sources),
            1,
        )
        self.assertGreaterEqual(capture_source.count("self._assert_live_state_authentication()"), 2)


if __name__ == "__main__":
    unittest.main()
