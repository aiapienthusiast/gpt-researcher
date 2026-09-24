"""Unit tests for the Cheaper Inference provider (LLM only).

These tests construct the provider object with a dummy key and assert it
points at the Cheaper Inference endpoint; no network calls are made.
"""
import os
import unittest

from gpt_researcher.llm_provider.generic.base import (
    GenericLLMProvider,
    _SUPPORTED_PROVIDERS as LLM_PROVIDERS,
)
from gpt_researcher.memory.embeddings import (
    _SUPPORTED_PROVIDERS as EMBEDDING_PROVIDERS,
)

CHEAPER_INFERENCE_BASE_URL = "https://api.cheaperinference.com/v1"


class TestCheaperInferenceProvider(unittest.TestCase):
    def setUp(self):
        self._orig = os.environ.get("CHEAPER_INFERENCE_API_KEY")
        os.environ["CHEAPER_INFERENCE_API_KEY"] = "test-key"

    def tearDown(self):
        if self._orig is None:
            os.environ.pop("CHEAPER_INFERENCE_API_KEY", None)
        else:
            os.environ["CHEAPER_INFERENCE_API_KEY"] = self._orig

    def test_registered_for_llm_only(self):
        self.assertIn("cheaperinference", LLM_PROVIDERS)
        # The gateway has no embeddings endpoint.
        self.assertNotIn("cheaperinference", EMBEDDING_PROVIDERS)

    def test_llm_construction_uses_cheaper_inference_base_url(self):
        provider = GenericLLMProvider.from_provider(
            "cheaperinference", model="gpt-5.4-mini", verbose=False
        )
        self.assertEqual(
            str(provider.llm.openai_api_base), CHEAPER_INFERENCE_BASE_URL
        )
        self.assertEqual(provider.llm.model_name, "gpt-5.4-mini")
        self.assertEqual(
            provider.llm.openai_api_key.get_secret_value(), "test-key"
        )

    def test_llm_requires_api_key(self):
        os.environ.pop("CHEAPER_INFERENCE_API_KEY", None)
        with self.assertRaises(KeyError):
            GenericLLMProvider.from_provider(
                "cheaperinference", model="gpt-5.4-mini", verbose=False
            )


if __name__ == "__main__":
    unittest.main()
