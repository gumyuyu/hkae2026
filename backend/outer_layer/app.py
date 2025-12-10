from contextlib import asynccontextmanager
from fastapi import FastAPI
from pydantic import BaseModel
from outer_layer.ai_agent import AIAgent


# --- Initialize the AI Agent globally ---
agent = AIAgent(model="gpt-oss:20b-cloud", base_url="http://localhost:11434")


# --- Lifecycle management using lifespan (modern replacement for @on_event) ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("[Startup] Initializing AI Agent and MCP client...")
    await agent.init_client()
    yield
    print("[Shutdown] Closing AI Agent and MCP client...")
    await agent.close_client()


# --- Create the FastAPI app ---
app = FastAPI(
    title="VR Image & 3D Object Generator Backend",
    description="Outer layer API that routes text/image inputs to the appropriate MCP tool via GPT-OSS reasoning.",
    lifespan=lifespan,
)


# --- Request schema ---
class InputData(BaseModel):
    text: str | None = None
    image_base64: str | None = None


# --- API endpoint for frontend ---
@app.post("/generate")
async def generate(payload: InputData):
    """
    Accepts text or image_base64 input from the frontend.
    Sends the payload to the AI Agent, which uses GPT-OSS to decide which MCP tool to call.
    Returns the MCP tool result.
    """
    result = await agent.route_request(payload.dict(exclude_none=True))
    return result


# --- Health check ---
@app.get("/")
async def root():
    return {"status": "ok", "message": "VR Image/3D Object Generator running."}
