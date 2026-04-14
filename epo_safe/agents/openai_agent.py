"""OpenAI LLM agent using the openai SDK."""

import logging
import os

import anyio
from openai import AsyncOpenAI

from epo_safe.agents.llm_agent import LLMAgent

logger = logging.getLogger(__name__)


class OpenAIAgent(LLMAgent):
    """Agent that generates trajectories via OpenAI models."""

    def __init__(self, model: str = "gpt-5.2-codex"):
        super().__init__(model=model)
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable is required for OpenAI models")
        self.client = AsyncOpenAI(api_key=api_key)

    async def _query(self, system_prompt: str, user_prompt: str) -> str:
        """Execute a single OpenAI API query with retry logic."""
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = await self.client.responses.create(
                    model=self.model,
                    instructions=system_prompt,
                    input=user_prompt,
                )
                return response.output_text
            except Exception as e:
                logger.warning(
                    "OpenAI error on attempt %d/%d: %s",
                    attempt + 1,
                    max_retries,
                    e,
                )
                if attempt < max_retries - 1:
                    await anyio.sleep(2 ** (attempt + 1))
                else:
                    raise
        return ""  # unreachable
