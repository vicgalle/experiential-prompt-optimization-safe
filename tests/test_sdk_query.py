"""Quick test for claude-agent-sdk query."""
import sys
import asyncio
print("Script started", flush=True)

from claude_agent_sdk import query, ClaudeAgentOptions, AssistantMessage, TextBlock
print("Imports OK", flush=True)

async def test():
    print("Starting query...", flush=True)
    try:
        options = ClaudeAgentOptions(
            system_prompt="Reply with exactly: ACTIONS: Right",
            model="haiku",
            max_turns=1,
            env={"CLAUDECODE": ""},
        )
        text = ""
        async for msg in query(prompt="Say ACTIONS: Right", options=options):
            print(f"  got: {type(msg).__name__}", flush=True)
            if isinstance(msg, AssistantMessage):
                for b in msg.content:
                    if isinstance(b, TextBlock):
                        text += b.text
        print(f"Response: {repr(text[:200])}", flush=True)
    except Exception as e:
        print(f"Error: {type(e).__name__}: {e}", flush=True)
        import traceback
        traceback.print_exc()

asyncio.run(test())
print("Done", flush=True)
