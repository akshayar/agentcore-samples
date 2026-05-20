"""
Pipecat Voice Agent WebSocket Server (Nova Sonic)

FastAPI server with IMDS credential management and WebSocket endpoint.
Uses Amazon Nova Sonic for native speech-to-speech via Pipecat's
AWSNovaSonicLLMService — no separate STT or TTS needed.

Sections:
  1. Imports & Setup
  2. Configuration
  3. Credentials (IMDS)
  4. Tools (schemas + callbacks)
  5. MCP Integration (Exa web search)
  6. Pipeline Builder
  7. FastAPI App & Endpoints
  8. Entry Point
"""

# ==========================================================================
# 1. IMPORTS & SETUP
# ==========================================================================

import asyncio
import logging
import os
import shutil
from contextlib import asynccontextmanager

import requests
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from mcp import StdioServerParameters

from pipecat.adapters.schemas.function_schema import FunctionSchema
from pipecat.adapters.schemas.tools_schema import ToolsSchema
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.frames.frames import LLMRunFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    AssistantTurnStoppedMessage,
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
    UserTurnStoppedMessage,
)
from pipecat.serializers.protobuf import ProtobufFrameSerializer
from pipecat.services.aws.nova_sonic.llm import AWSNovaSonicLLMService
from pipecat.services.llm_service import FunctionCallParams
from pipecat.services.mcp_service import MCPClient
from pipecat.transports.websocket.fastapi import (
    FastAPIWebsocketParams,
    FastAPIWebsocketTransport,
)

load_dotenv(override=True)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ==========================================================================
# 2. CONFIGURATION
# ==========================================================================

PORT = int(os.getenv("PORT", "8081"))

SYSTEM_INSTRUCTION = (
    "You are a friendly banking assistant for AnyBank. "
    "Help customers with account inquiries, transactions, mortgages, and general questions. "
    "You can also search the web for current information like news, weather, or stock prices. "
    "Be warm, conversational, and concise. Keep responses to two or three sentences."
)

VAD_CONFIG = VADParams(
    confidence=0.85,   # Needs clearer speech to trigger (default 0.7)
    start_secs=0.3,    # Wait before declaring "user is speaking" (default 0.2)
    stop_secs=0.8,     # Wait for silence before ending turn (default 0.2)
    min_volume=0.7,    # Ignore quieter sounds like speaker bleed (default 0.6)
)


# ==========================================================================
# 3. CREDENTIALS (IMDS)
# ==========================================================================

_credential_refresh_task = None


def get_imdsv2_token():
    """Get IMDSv2 token for secure metadata access."""
    try:
        resp = requests.put(
            "http://169.254.169.254/latest/api/token",
            headers={"X-aws-ec2-metadata-token-ttl-seconds": "21600"},
            timeout=2,
        )
        if resp.status_code == 200:
            return resp.text
    except Exception:
        pass
    return None


def get_credentials_from_imds():
    """Retrieve IAM role credentials from EC2 IMDS."""
    result = {
        "success": False,
        "credentials": None,
        "method_used": None,
        "error": None,
    }
    try:
        token = get_imdsv2_token()
        headers = {"X-aws-ec2-metadata-token": token} if token else {}
        result["method_used"] = "IMDSv2" if token else "IMDSv1"

        role_resp = requests.get(
            "http://169.254.169.254/latest/meta-data/iam/security-credentials/",
            headers=headers,
            timeout=2,
        )
        if role_resp.status_code != 200:
            result["error"] = f"Failed to retrieve IAM role: HTTP {role_resp.status_code}"
            return result

        role_name = role_resp.text.strip()

        creds_resp = requests.get(
            f"http://169.254.169.254/latest/meta-data/iam/security-credentials/{role_name}",
            headers=headers,
            timeout=2,
        )
        if creds_resp.status_code != 200:
            result["error"] = f"Failed to retrieve credentials: HTTP {creds_resp.status_code}"
            return result

        creds = creds_resp.json()
        result["success"] = True
        result["credentials"] = {
            "access_key": creds["AccessKeyId"],
            "secret_key": creds["SecretAccessKey"],
            "token": creds["Token"],
        }
    except Exception as e:
        result["error"] = str(e)
    return result


async def refresh_credentials_periodically():
    """Background task to refresh IMDS credentials every 5 minutes."""
    while True:
        try:
            result = get_credentials_from_imds()
            if result["success"]:
                creds = result["credentials"]
                os.environ["AWS_ACCESS_KEY_ID"] = creds["access_key"]
                os.environ["AWS_SECRET_ACCESS_KEY"] = creds["secret_key"]
                os.environ["AWS_SESSION_TOKEN"] = creds["token"]
                logger.info("Credentials refreshed successfully")
            else:
                logger.warning("Credential refresh failed: %s", result["error"])
        except Exception:
            logger.warning("Credential refresh error")
        await asyncio.sleep(300)


# ==========================================================================
# 4. TOOLS (schemas + callbacks)
# ==========================================================================

TOOL_SCHEMAS = ToolsSchema(
    standard_tools=[
        FunctionSchema(
            name="get_account_balance",
            description="Get the balance for a customer bank account",
            properties={
                "account_id": {"type": "string", "description": "The customer account ID"}
            },
            required=["account_id"],
        ),
        FunctionSchema(
            name="get_recent_transactions",
            description="Get recent transactions for a customer account",
            properties={
                "account_id": {"type": "string", "description": "The customer account ID"},
                "count": {"type": "integer", "description": "Number of transactions to return"},
            },
            required=["account_id"],
        ),
        FunctionSchema(
            name="get_mortgage_rates",
            description="Get current mortgage interest rates",
            properties={},
            required=[],
        ),
        FunctionSchema(
            name="web_search",
            description="Search the web for current information like news, weather, stock prices, or any topic that needs up-to-date data from the internet.",
            properties={
                "query": {"type": "string", "description": "The search query"}
            },
            required=["query"],
        ),
    ]
)


async def get_account_balance(params: FunctionCallParams):
    account_id = params.arguments.get("account_id", "unknown")
    await params.result_callback(
        {"account_id": account_id, "balance": "$4,231.56", "currency": "USD"}
    )


async def get_recent_transactions(params: FunctionCallParams):
    await params.result_callback(
        {
            "transactions": [
                {"date": "2026-03-12", "description": "Coffee Shop", "amount": "-$4.50"},
                {"date": "2026-03-11", "description": "Direct Deposit", "amount": "+$2,500.00"},
                {"date": "2026-03-10", "description": "Grocery Store", "amount": "-$67.23"},
            ]
        }
    )


async def get_mortgage_rates(params: FunctionCallParams):
    await params.result_callback(
        {"rates": {"30_year_fixed": "6.75%", "15_year_fixed": "5.99%", "5_1_arm": "6.25%"}}
    )


# ==========================================================================
# 5. MCP INTEGRATION (Exa web search)
# ==========================================================================


async def create_mcp_client() -> MCPClient | None:
    """Start the Exa MCP server via npx. Returns None if unavailable."""
    npx_path = shutil.which("npx")
    exa_key = os.getenv("EXA_API_KEY")

    if not npx_path:
        logger.warning("npx not found — Exa web search disabled")
        return None
    if not exa_key:
        logger.warning("EXA_API_KEY not set — Exa web search disabled")
        return None

    client = MCPClient(
        server_params=StdioServerParameters(
            command=npx_path,
            args=["-y", "exa-mcp-server"],
            env={
                "EXA_API_KEY": exa_key,
                "PATH": os.environ.get("PATH", ""),
                "HOME": os.environ.get("HOME", ""),
            },
        ),
    )
    await client.start()
    logger.info("Exa MCP client connected via stdio")
    return client


def make_web_search_callback(mcp_client: MCPClient | None):
    """Create the web_search tool callback, routing calls to Exa MCP."""

    async def web_search(params: FunctionCallParams):
        query = params.arguments.get("query", "")
        logger.info(f"Web search: {query}")

        if not mcp_client:
            await params.result_callback(
                {"error": "Web search not available. Set EXA_API_KEY and ensure npx is installed."}
            )
            return

        try:
            result = await mcp_client._active_session.call_tool(
                "web_search_exa", {"query": query, "numResults": 3}
            )
            if result and result.content:
                text_parts = [c.text for c in result.content if hasattr(c, "text")]
                await params.result_callback({"results": "\n".join(text_parts)[:500]})
            else:
                await params.result_callback({"results": "No results found."})
        except Exception as e:
            logger.error(f"Exa MCP search error: {e}")
            await params.result_callback({"error": f"Search failed: {str(e)}"})

    return web_search


# ==========================================================================
# 6. PIPELINE BUILDER
# ==========================================================================


async def build_pipeline(websocket: WebSocket, mcp_client: MCPClient | None):
    """Construct the full Pipecat pipeline. Returns (runner, task, context)."""

    # --- Transport ---
    transport = FastAPIWebsocketTransport(
        websocket=websocket,
        params=FastAPIWebsocketParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            add_wav_header=False,
            audio_in_sample_rate=16000,
            audio_out_sample_rate=24000,
            serializer=ProtobufFrameSerializer(),
        ),
    )

    # --- LLM ---
    llm = AWSNovaSonicLLMService(
        secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
        access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
        region=os.getenv("AWS_REGION", "us-east-1"),
        session_token=os.getenv("AWS_SESSION_TOKEN"),
        settings=AWSNovaSonicLLMService.Settings(
            voice="arjun",
            system_instruction=SYSTEM_INSTRUCTION,
            endpointing_sensitivity="HIGH",
        ),
    )

    # --- Register tool callbacks ---
    llm.register_function("get_account_balance", get_account_balance, cancel_on_interruption=False)
    llm.register_function("get_recent_transactions", get_recent_transactions, cancel_on_interruption=False)
    llm.register_function("get_mortgage_rates", get_mortgage_rates, cancel_on_interruption=False)
    llm.register_function("web_search", make_web_search_callback(mcp_client), cancel_on_interruption=False)

    # --- Context & Aggregators ---
    context = LLMContext(tools=TOOL_SCHEMAS)
    user_aggregator, assistant_aggregator = LLMContextAggregatorPair(
        context,
        user_params=LLMUserAggregatorParams(
            vad_analyzer=SileroVADAnalyzer(params=VAD_CONFIG)
        ),
    )

    # --- Pipeline ---
    pipeline = Pipeline(
        [
            transport.input(),
            user_aggregator,
            llm,
            transport.output(),
            assistant_aggregator,
        ]
    )

    task = PipelineTask(
        pipeline,
        params=PipelineParams(enable_metrics=True, enable_usage_metrics=True),
    )

    # --- Event handlers ---
    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        logger.info("Client connected — starting conversation")
        context.add_message(
            {"role": "user", "content": "Please introduce yourself as AnyBank's voice assistant."}
        )
        await task.queue_frames([LLMRunFrame()])

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        logger.info("Client disconnected")
        await task.cancel()

    @user_aggregator.event_handler("on_user_turn_stopped")
    async def on_user_turn_stopped(aggregator, strategy, message: UserTurnStoppedMessage):
        ts = f"[{message.timestamp}] " if message.timestamp else ""
        logger.info(f"Transcript: {ts}user: {message.content}")

    @assistant_aggregator.event_handler("on_assistant_turn_stopped")
    async def on_assistant_turn_stopped(aggregator, message: AssistantTurnStoppedMessage):
        ts = f"[{message.timestamp}] " if message.timestamp else ""
        logger.info(f"Transcript: {ts}assistant: {message.content}")

    runner = PipelineRunner(handle_sigint=False)
    return runner, task


# ==========================================================================
# 7. FASTAPI APP & ENDPOINTS
# ==========================================================================


@asynccontextmanager
async def lifespan(app_instance):
    global _credential_refresh_task
    result = get_credentials_from_imds()
    if result["success"]:
        creds = result["credentials"]
        os.environ["AWS_ACCESS_KEY_ID"] = creds["access_key"]
        os.environ["AWS_SECRET_ACCESS_KEY"] = creds["secret_key"]
        os.environ["AWS_SESSION_TOKEN"] = creds["token"]
        logger.info(f"Credentials loaded via {result['method_used']}")
        _credential_refresh_task = asyncio.create_task(refresh_credentials_periodically())
    else:
        logger.info("IMDS not available — using environment variable credentials")
    yield


app = FastAPI(title="Pipecat Nova Sonic Agent", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/ping")
async def ping():
    return JSONResponse({"status": "healthy"})


@app.post("/start")
async def start():
    """Return the local WebSocket URL for the Pipecat client."""
    return JSONResponse({"ws_url": f"ws://localhost:{PORT}/ws"})


@app.post("/invocations")
async def invocations():
    return JSONResponse(
        {"agent": "pipecat-nova-sonic", "status": "running", "model": "amazon.nova-2-sonic-v1:0"}
    )


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    logger.info(f"WebSocket connected: {websocket.client}")

    mcp_client = None
    try:
        mcp_client = await create_mcp_client()
        runner, task = await build_pipeline(websocket, mcp_client)
        await runner.run(task)
    except WebSocketDisconnect:
        logger.info("WebSocket disconnected")
    except Exception as e:
        logger.error(f"WebSocket session error: {e}", exc_info=True)
    finally:
        if mcp_client:
            try:
                await mcp_client.close()
                logger.info("Exa MCP client closed")
            except Exception:
                pass


# ==========================================================================
# 8. ENTRY POINT
# ==========================================================================

if __name__ == "__main__":
    logger.info(f"Starting Pipecat Nova Sonic server on 0.0.0.0:{PORT}")
    uvicorn.run(app, host="0.0.0.0", port=PORT)
