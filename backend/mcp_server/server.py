from fastmcp import FastMCP
from pydantic import BaseModel
import base64

from inner_layer.backend import (
    generate_image_from_text,
    generate_object_from_image,
    generate_object_from_text
)

# -----------------------------
# Input Models
# -----------------------------
class TextPrompt(BaseModel):
    prompt: str

class ImageInput(BaseModel):
    image_base64: str

class ObjectFromTextInput(BaseModel):
    prompt: str


# -----------------------------
# MCP Server Setup
# -----------------------------
mcp = FastMCP(name="VR Object Generator MCP")

@mcp.tool(name="generate_image")
def tool_generate_image(payload: TextPrompt):
    b64_image = generate_image_from_text(payload.prompt)
    return {"object_base64": b64_image}

@mcp.tool(name="generate_object")
def tool_generate_object(payload: ImageInput):
    """Image → 3D object"""
    b64_object = generate_object_from_image(payload.image_base64)
    return {"object_base64": b64_object}

@mcp.tool(name="generate_object_from_text")
def tool_generate_object_from_text(payload: ObjectFromTextInput):
    """Text → Image → 3D object"""
    b64_object = generate_object_from_text(payload.prompt)
    return {"object_base64": b64_object}


def start_mcp_server(blocking=True):
    """Start MCP server (blocking or non-blocking)."""
    mcp.run(blocking=blocking)


if __name__ == "__main__":
    start_mcp_server()
