import os
import json
import asyncio
from huggingface_hub import InferenceClient
import aiohttp
from fastmcp import Client
from mcp_server import server as mcp_server
import base64
import io
from pathlib import Path
from PIL import Image
import datetime

class AIAgent:
    def __init__(
        self,
        model: str = "openai/gpt-oss-120b"
    ):
        """
        model: which HF GPT-OSS model to use (e.g., openai/gpt-oss-120b:groq)
        """
        self.model = model
        self.api_key = os.getenv("HF_TOKEN")
        if not self.api_key:
            raise RuntimeError("HF_TOKEN not set. Add it as an environment variable.")

        self.mcp = mcp_server.mcp 
        self.client = None
        self.invoke = InferenceClient(api_key=self.api_key)
    
    async def init_client(self):
        """Initialize MCP client."""
        self.client = await Client(self.mcp).__aenter__()
        

    async def close_client(self):
        """Close MCP client."""
        if self.client:
            await self.client.__aexit__(None, None, None)

    async def _decide_tool(self, payload: dict) -> dict:
        """
        Ask the GPT-OSS model which tool to use and with what parameters.
        Model must respond with a JSON object like:
        { "tool": "generate_image", "params": { "prompt": "..." } }
        """
        system_prompt = """
You are a routing assistant that decides which MCP tool to call.
Available tools:
1. generate_image — for creating an image from a text prompt.
2. generate_object — for creating a 3D object from an image (image_base64).
3. generate_object_from_text — for creating a 3D object from a text prompt.

Rules:
- ALWAYS respond with a JSON object, nothing else.
- JSON structure:
    {
      "tool": "<tool_name>",
      "params": { "<parameter>": "<value>" }
    }
- Do not explain or include natural language; just output valid JSON.
- CRITICAL: output must be raw JSON. No markdown, no backticks, no explanations.
"""

        user_prompt = f"User input:\n{json.dumps(payload)}"

        # Use asyncio.to_thread to keep async
        response_str = await asyncio.to_thread(
            lambda: self.invoke.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=256,
                temperature=0.0,
                top_p=0.1
            ).choices[0].message["content"]
        )

        raw_output = response_str.strip()

        # Defensive JSON parsing
        try:
            decision = json.loads(raw_output)
        except json.JSONDecodeError:
            start = raw_output.find("{")
            end = raw_output.rfind("}") + 1
            if start != -1 and end != -1:
                decision = json.loads(raw_output[start:end])
            else:
                raise ValueError(f"Model returned invalid JSON: {raw_output}")

        # Validate keys
        if "tool" not in decision or "params" not in decision:
            raise ValueError(f"Model response missing keys: {decision}")

        return decision

    async def route_request(self, payload: dict) -> dict:
        """
        Ask the AI which tool to use. This version only returns the
        routing decision since HF InferenceClient does not call MCP tools.
        Returns:
        - used_tool: the tool suggested by the AI
        - params: parameters suggested
        """
        decision = await self._decide_tool(payload)
        print(f"[AIAgent] Decided to use tool: {decision}")

        # Here you would call your MCP server if you still have one.
        # For now, we just return the routing decision.
        tool = decision["tool"]
        params = {"payload": decision["params"]}

        if not self.client:
            raise RuntimeError("MCP client not initialized. Call init_client() first.")

        result = await self.client.call_tool(tool, params)
        
        return {
            "used_tool": tool,
            "params": decision["params"],
            "result": result.data,
        }

# Example usage
async def main():
    agent = AIAgent()
    payload = {"text": "Create a 3D model of a futuristic car."}
    result = await agent.route_request(payload)
    print(result)

if __name__ == "__main__":
    asyncio.run(main())
