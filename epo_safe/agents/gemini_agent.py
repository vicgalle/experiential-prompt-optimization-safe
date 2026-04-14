"""Gemini LLM agent using the google-genai SDK."""

import logging
import os

import anyio
from google import genai
from google.genai.types import GenerateContentConfig

from epo_safe.agents.llm_agent import LLMAgent

logger = logging.getLogger(__name__)


class GeminiAgent(LLMAgent):
    """Agent that generates trajectories via Google Gemini models."""

    def __init__(self, model: str = "gemini-2.5-flash-preview-05-20"):
        super().__init__(model=model)
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY environment variable is required for Gemini models")
        self.client = genai.Client(api_key=api_key)

    async def _query(self, system_prompt: str, user_prompt: str) -> str:
        """Execute a single Gemini API query with retry logic."""
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = await self.client.aio.models.generate_content(
                    model=self.model,
                    contents=user_prompt,
                    config=GenerateContentConfig(
                        system_instruction=system_prompt,
                    ),
                )
                return response.text
            except Exception as e:
                logger.warning(
                    "Gemini error on attempt %d/%d: %s",
                    attempt + 1,
                    max_retries,
                    e,
                )
                if attempt < max_retries - 1:
                    await anyio.sleep(2 ** (attempt + 1))
                else:
                    raise
        return ""  # unreachable
