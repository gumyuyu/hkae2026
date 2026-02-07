from contextlib import asynccontextmanager
from fastapi import FastAPI
from pydantic import BaseModel
from outer_layer.ai_agent import AIAgent
import os
from fastapi import HTTPException, Request

# --- Initialize the AI Agent globally ---
agent = AIAgent(model="openai/gpt-oss-120b")


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
API_KEY = os.getenv("MY_API_KEY")

@app.post("/generate")
async def generate(payload: InputData, request: Request):
    auth_header = request.headers.get("Authorization")
    if auth_header != f"Bearer {API_KEY}":
        raise HTTPException(status_code=401, detail="Unauthorized")
    result = await agent.route_request(payload.dict(exclude_none=True))
    return result


# --- Health check ---
@app.get("/")
async def root():
    return {"status": "ok", "message": "VR Image/3D Object Generator running."}
