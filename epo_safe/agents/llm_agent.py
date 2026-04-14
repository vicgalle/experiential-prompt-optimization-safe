"""LLM agent using claude-agent-sdk for trajectory generation and reflection."""

import logging
import re

import anyio

from claude_agent_sdk import (
    query,
    ClaudeAgentOptions,
    AssistantMessage,
    TextBlock,
    ClaudeSDKError,
)

from epo_safe.environments.base_wrapper import ACTION_FROM_NAME

logger = logging.getLogger(__name__)


class LLMAgent:
    """Agent that generates trajectories via claude-agent-sdk."""

    def __init__(self, model: str = "sonnet"):
        self.model = model

    async def generate_trajectory(
        self,
        system_prompt: str,
        user_prompt: str,
    ) -> str:
        """Send a single query to generate a full action sequence."""
        return await self._query(system_prompt, user_prompt)

    async def reflect(
        self,
        system_prompt: str,
        reflection_prompt: str,
    ) -> str:
        """Send a reflection query to analyze trajectories and update spec."""
        return await self._query(system_prompt, reflection_prompt)

    async def _query(self, system_prompt: str, user_prompt: str) -> str:
        """Execute a single claude-agent-sdk query with retry logic."""
        max_retries = 3
        for attempt in range(max_retries):
            try:
                options = ClaudeAgentOptions(
                    system_prompt=system_prompt,
                    model=self.model,
                    max_turns=1,
                    allowed_tools=[],
                    # Unset CLAUDECODE so the SDK subprocess doesn't think
                    # it's nested inside another Claude Code session.
                    env={"CLAUDECODE": ""},
                )
                response_text = ""
                try:
                    async for message in query(prompt=user_prompt, options=options):
                        if isinstance(message, AssistantMessage):
                            for block in message.content:
                                if isinstance(block, TextBlock):
                                    response_text += block.text
                except ClaudeSDKError as e:
                    # The SDK may raise on unknown event types (e.g. rate_limit_event)
                    # after already yielding the actual response. If we have text, use it.
                    if response_text:
                        logger.debug("SDK stream ended with: %s (response captured)", e)
                    else:
                        raise
                return response_text
            except ClaudeSDKError as e:
                logger.warning(
                    "SDK error on attempt %d/%d: %s", attempt + 1, max_retries, e
                )
                if attempt < max_retries - 1:
                    await anyio.sleep(2 ** (attempt + 1))
                else:
                    raise
            except Exception as e:
                logger.warning(
                    "Unexpected error on attempt %d/%d: %s",
                    attempt + 1,
                    max_retries,
                    e,
                )
                if attempt < max_retries - 1:
                    await anyio.sleep(2 ** (attempt + 1))
                else:
                    raise
        return ""  # unreachable

    def parse_actions(
        self,
        response: str,
        action_vocab: dict[str, int] | None = None,
    ) -> list[int]:
        """Parse action sequence from LLM response.

        Args:
            response: Raw LLM response text.
            action_vocab: Optional mapping of action name → int for text scenarios.
                          If None, uses the default up/down/left/right vocab.

        Handles formats:
          ACTIONS: Up, Down, Left, Right
          Up, Down, Left, Right
          1. Up  2. Down  3. Left
          up down left right
        """
        # Try to find ACTIONS: line first
        match = re.search(r"ACTIONS:\s*(.+)", response, re.IGNORECASE)
        if match:
            action_text = match.group(1).strip()
        else:
            # Fall back to finding action words anywhere
            action_text = response

        if action_vocab is not None:
            # Build pattern from custom vocab keys
            escaped = [re.escape(k) for k in action_vocab]
            pattern = re.compile(
                r"\b(" + "|".join(escaped) + r")\b", re.IGNORECASE
            )
            found = pattern.findall(action_text)
            actions = []
            for name in found:
                action = action_vocab.get(name.lower())
                if action is not None:
                    actions.append(action)
        else:
            # Default: up/down/left/right
            action_pattern = re.compile(
                r"\b(up|down|left|right)\b", re.IGNORECASE
            )
            found = action_pattern.findall(action_text)
            actions = []
            for name in found:
                action = ACTION_FROM_NAME.get(name.lower())
                if action is not None:
                    actions.append(action)

        if not actions:
            logger.warning("No valid actions parsed from response: %s", response[:200])

        return actions

    def parse_specification(self, response: str) -> str | None:
        """Extract updated behavioral specification from <specification> tags."""
        match = re.search(
            r"<specification>(.*?)</specification>", response, re.DOTALL
        )
        if match:
            spec = match.group(1).strip()
            # Ensure it starts with the header
            if not spec.startswith("## Behavioral Specification"):
                spec = "## Behavioral Specification\n" + spec
            return spec
        logger.warning("No <specification> tags found in reflection response")
        return None


def create_agent(model: str) -> LLMAgent:
    """Factory: return the right agent subclass based on model name."""
    if model.startswith("gemini"):
        from epo_safe.agents.gemini_agent import GeminiAgent

        return GeminiAgent(model=model)
    if model.startswith("gpt") or model.startswith("o"):
        from epo_safe.agents.openai_agent import OpenAIAgent

        return OpenAIAgent(model=model)
    return LLMAgent(model=model)
