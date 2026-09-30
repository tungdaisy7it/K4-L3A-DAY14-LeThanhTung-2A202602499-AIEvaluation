"""Generate artifacts/actual_answers.json with Gemini instead of OpenAI.

``domain_assistant.py`` is the system under evaluation and stays untouched. It
accepts any ``TextGenerator``, so this script only swaps the generation step for
Gemini's OpenAI-compatible endpoint (using the ``openai`` SDK already listed in
requirements.txt). Retrieval, prompt and the artifact format are unchanged.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI, OpenAIError

from domain_assistant import generate_actual_answers

load_dotenv(Path(__file__).resolve().with_name(".env"))

GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
RETRY_DELAYS_SECONDS = (10, 30, 60, 90, 120, 120)


CACHE_PATH = Path("artifacts/.gemini_generation_cache.json")


class GeminiGenerator:
    """Gemini generator with a (model, prompt) cache.

    The free tier allows few requests per day, so a completed answer is written
    to disk immediately and a re-run after a transient failure reuses it.
    """

    def __init__(
        self, max_output_tokens: int = 2048, model: str | None = None
    ) -> None:
        api_key = os.getenv("GEMINI_API_KEY", "").strip()
        self.model = model or os.getenv("GEMINI_MODEL", "gemini-3.5-flash").strip()
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY is missing from .env")
        # max_retries=0: the SDK's hidden retries would each spend free-tier quota.
        self.client = OpenAI(
            api_key=api_key, base_url=GEMINI_BASE_URL, max_retries=0
        )
        self.max_output_tokens = max_output_tokens
        self.cache: dict[str, str] = {}
        if CACHE_PATH.is_file():
            self.cache = json.loads(CACHE_PATH.read_text(encoding="utf-8"))

    def _cache_key(self, prompt: str) -> str:
        return hashlib.sha256(f"{self.model}\n{prompt}".encode()).hexdigest()

    def generate(self, prompt: str) -> str:
        key = self._cache_key(prompt)
        if key in self.cache:
            return self.cache[key]

        for delay in (*RETRY_DELAYS_SECONDS, None):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0,
                    max_tokens=self.max_output_tokens,
                )
                break
            except OpenAIError as exc:
                # A daily quota will not recover within a retry window.
                if delay is None or "PerDay" in str(exc):
                    raise
                time.sleep(delay)
        answer = (response.choices[0].message.content or "").strip()
        if not answer:
            raise RuntimeError("Gemini returned an empty answer")

        self.cache[key] = answer
        CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        CACHE_PATH.write_text(
            json.dumps(self.cache, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        return answer


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus-dir", type=Path, default=Path("data/technology_store"))
    parser.add_argument("--dataset", type=Path, default=Path("golden_dataset.json"))
    parser.add_argument(
        "--output", type=Path, default=Path("artifacts/actual_answers.json")
    )
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    try:
        artifact = generate_actual_answers(
            args.dataset,
            args.corpus_dir,
            generator=GeminiGenerator(),
            top_k=args.top_k,
            progress=lambda message: print(message, flush=True),
        )
        output = args.output.expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(artifact, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    except (OSError, OpenAIError, TypeError, ValueError, RuntimeError) as exc:
        print(f"ERROR: {exc}")
        return 2
    print(f"Generated {len(artifact['answers'])} actual answers: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
