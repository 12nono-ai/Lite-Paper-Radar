from pathlib import Path
import os
import tempfile
import unittest
from unittest import mock

from arxiv_llm_watch.cli import _doctor_report, _init_env_file
from arxiv_llm_watch.config import AppConfig


class CliTests(unittest.TestCase):
    def test_init_env_file_creates_openai_template(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            env_path = Path(temp_dir) / ".env"
            summary = _init_env_file(env_path, provider="openai", force=False)

            self.assertTrue(env_path.exists())
            content = env_path.read_text(encoding="utf-8")
            self.assertIn("LLM_PROVIDER=openai_compatible", content)
            self.assertIn("LLM_BASE_URL=https://api.openai.com/v1", content)
            self.assertIn("LLM_API_PATH=/chat/completions", content)
            self.assertEqual(summary["provider"], "openai_compatible")

    def test_doctor_report_flags_missing_api_key(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            env_path = Path(temp_dir) / ".env"
            env_path.write_text(
                "\n".join(
                    [
                        "LLM_PROVIDER=openai_compatible",
                        "LLM_API_KEY=",
                        "LLM_BASE_URL=https://api.example.com/v1",
                        "LLM_MODEL=gpt-test",
                    ]
                ),
                encoding="utf-8",
            )
            with mock.patch.dict(os.environ, {}, clear=True):
                config = AppConfig.from_env(env_path)
                report = _doctor_report(config, env_path)

        self.assertFalse(report["ok"])
        api_key_check = next(item for item in report["checks"] if item["name"] == "api_key")
        self.assertFalse(api_key_check["ok"])
        self.assertIn("LLM_API_KEY", api_key_check["fix"])


if __name__ == "__main__":
    unittest.main()
