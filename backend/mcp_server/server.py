from fastmcp import FastMCP
from pydantic import BaseModel
import base64

# Input models
class TextPrompt(BaseModel):
    prompt: str

class ImageInput(BaseModel):
    image_base64: str

class ObjectFromTextInput(BaseModel):
    text: str  # keep it as 'text' because AI is sending {'text': 'car'}

# -----------------------------
# MCP Server
# -----------------------------
mcp = FastMCP(name="VR Object Generator MCP")

@mcp.tool(name="generate_image")
def tool_generate_image(payload: TextPrompt):
    return {"image_base64": base64.b64encode(f"Generated image for: {payload.prompt}".encode()).decode()}

@mcp.tool(name="generate_object")
def tool_generate_object(payload: ImageInput):
    glb_content = f"3D object generated from image: {payload.image_base64[:30]}..."
    return {"object_base64": base64.b64encode(glb_content.encode()).decode()}

@mcp.tool(name="generate_object_from_text")
def tool_generate_object_from_text(payload: ObjectFromTextInput):
    # Access the correct field 'text'
    glb_content = f"3D object generated from description: {payload.text}"
    return {"object_base64": base64.b64encode(glb_content.encode()).decode()}

def start_mcp_server(blocking=True):
    mcp.run(blocking=blocking)

if __name__ == "__main__":
    start_mcp_server()
