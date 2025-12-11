import aiohttp
import json
from fastmcp import Client
from mcp_server import server as mcp_server

import base64
import io
from pathlib import Path
from PIL import Image
import datetime

class AIAgent:
    def __init__(self, model: str = "gpt-oss:20b-cloud", base_url: str = "http://localhost:11434"):
        """
        model: which Ollama model to use (e.g., gpt-oss:20b-cloud)
        base_url: local or remote Ollama endpoint
        """
        self.model = model
        self.base_url = base_url
        self.mcp = mcp_server.mcp 
        self.client = None

    async def init_client(self):
        """Initialize MCP client."""
        self.client = await Client(self.mcp).__aenter__()
        

    async def close_client(self):
        """Close MCP client."""
        if self.client:
            await self.client.__aexit__(None, None, None)

    async def _decide_tool(self, payload: dict) -> dict:
        """
        Ask the GPT-OSS model which MCP tool to use and with what parameters.
        The model should respond with a JSON object like:
        { "tool": "generate_image", "params": { "text": "..." } }
        """
        system_prompt = """
        You are a routing assistant that decides which MCP tool to call.
        Available tools:
        1. generate_image — for creating an image from a text prompt.
        2. generate_object — for creating a 3D object from an image (image_base64).
        3. generate_object_from_text — for creating a 3D object from a text description.

        Rules:
        - ALWAYS respond with a JSON object, nothing else.
        - JSON structure:
            {
              "tool": "<tool_name>",
              "params": { "<parameter>": "<value>" }
            }
        - Do not explain or include natural language; just output valid JSON.
        """

        user_prompt = f"User input:\n{json.dumps(payload)}"

        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": system_prompt + "\n\n" + user_prompt,
                    "stream": False,
                },
            ) as resp:
                response = await resp.json()
                raw_output = response.get("response", "").strip()

                # Parse the model output into a dict
                try:
                    decision = json.loads(raw_output)
                except json.JSONDecodeError:
                    raise ValueError(f"Model returned invalid JSON: {raw_output}")

                if "tool" not in decision or "params" not in decision:
                    raise ValueError(f"Model response missing keys: {decision}")

                return decision

    async def route_request(self, payload: dict) -> dict:
        """Ask the AI which tool to use, then call that tool on the MCP server."""
        decision = await self._decide_tool(payload)
        print(f"[AIAgent] Decided to use tool: {decision}")
        tool = decision["tool"]
        params = {"payload": decision["params"]}
        

        result = await self.client.call_tool(tool, params)
        image_b64 = result.data.get("image_base64")
        if image_b64:
            # Ensure the /images folder exists
            images_dir = Path("images")
            images_dir.mkdir(exist_ok=True)

            # Convert base64 to image
            image_bytes = base64.b64decode(image_b64)
            image = Image.open(io.BytesIO(image_bytes))

            # Save with a dynamic filename (e.g., timestamp)
            filename = images_dir / f"{tool}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            image.save(filename)
            print(f"[AIAgent] Saved generated image to {filename}")
        return {
            "used_tool": tool,
            "params": decision["params"],
            "result": result.data,
        }
