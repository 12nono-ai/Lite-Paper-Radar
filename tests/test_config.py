from pathlib import Path
import os
import tempfile
import unittest
from unittest import mock

from arxiv_llm_watch.config import AppConfig


class ConfigTests(unittest.TestCase):
    def test_from_env_supports_openai_compatible_settings(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            env_path = Path(temp_dir) / ".env"
            env_path.write_text(
                "\n".join(
                    [
                        "LLM_PROVIDER=openai",
                        "LLM_API_KEY=test-key",
                        "LLM_BASE_URL=https://api.example.com/v1",
                        "LLM_MODEL=gpt-test",
                        "LLM_API_PATH=/chat/completions",
                        'LLM_HEADERS_JSON={"X-Test":"1"}',
                    ]
                ),
                encoding="utf-8",
            )
            with mock.patch.dict(os.environ, {}, clear=True):
                config = AppConfig.from_env(env_path)

        self.assertEqual(config.llm_provider, "openai_compatible")
        self.assertEqual(config.llm_api_key, "test-key")
        self.assertEqual(config.llm_base_url, "https://api.example.com/v1")
        self.assertEqual(config.llm_model, "gpt-test")
        self.assertEqual(config.llm_headers, {"X-Test": "1"})
        self.assertEqual(config.llm_endpoint, "https://api.example.com/v1/chat/completions")
        self.assertEqual(config.llm_provider_label, "OpenAI Compatible")

    def test_from_env_accepts_legacy_ark_settings(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            env_path = Path(temp_dir) / ".env"
            env_path.write_text(
                "\n".join(
                    [
                        "ARK_API_KEY=ark-key",
                        "ARK_BASE_URL=https://ark.example.com/api/v3",
                        "ARK_MODEL=ark-model",
                    ]
                ),
                encoding="utf-8",
            )
            with mock.patch.dict(os.environ, {}, clear=True):
                config = AppConfig.from_env(env_path)

        self.assertEqual(config.llm_provider, "ark")
        self.assertEqual(config.llm_api_key, "ark-key")
        self.assertEqual(config.llm_base_url, "https://ark.example.com/api/v3")
        self.assertEqual(config.llm_model, "ark-model")
        self.assertEqual(config.llm_provider_label, "Ark")


if __name__ == "__main__":
    unittest.main()
