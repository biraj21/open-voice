import asyncio
from contextlib import asynccontextmanager
from json import JSONDecodeError
from typing import Literal, Protocol, cast

from aiortc import RTCIceCandidate
from aiortc.sdp import candidate_from_sdp
from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from starlette.websockets import WebSocketDisconnect

from src.env import env  # noqa: F401
from src.logger import get_logger
from src.schemas.realtime import SessionConfig
from src.webrtc import TurnServer, WebRTCConnection, get_turn_servers

logger = get_logger(__name__)


class AppState(Protocol):
    connections: set[WebRTCConnection]


@asynccontextmanager
async def lifespan(app: FastAPI):
    # will connect to DB and stuff here

    yield

    close_tasks: list[asyncio.Task] = []
    for conn in state.connections:
        close_tasks.append(asyncio.create_task(conn.close()))

    await asyncio.gather(*close_tasks)


app = FastAPI(lifespan=lifespan)

state = cast(AppState, app.state)
"""Stores FastAPI app.state object"""

state.connections = set()


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def get_health():
    return {
        "status": "ok",
        "active_connections": len(state.connections),
    }


class IceCandidateMsg(BaseModel):
    type: Literal["ice_candidate"]
    candidate: str
    sdpMid: str | None = None
    sdpMLineIndex: int | None = None


class SdpOfferMsg(BaseModel):
    type: Literal["offer"]
    sdp: str
    session: SessionConfig


@app.websocket("/call")
async def websocket_endpoint(ws: WebSocket, use_turn: bool = False):
    try:
        await ws.accept()

        turn_servers: list[TurnServer] | None = None
        if use_turn:
            turn_servers = await get_turn_servers()
            await ws.send_json(
                {"type": "turn_servers", "turn_servers": [ts.model_dump() for ts in turn_servers]}
            )

        conn: WebRTCConnection | None = None

        # ice candidates received before we create the WebRTC connection object
        pending_ice_candidates: list[RTCIceCandidate] = []

        while True:
            try:
                data = await ws.receive_json()
                data_type = data.get("type")
                if data_type == "ice_candidate":
                    msg = IceCandidateMsg.model_validate(data)

                    # this logic is based on Modal's WebRTC YOLO example
                    ice_candidate = candidate_from_sdp(msg.candidate)
                    ice_candidate.sdpMid = msg.sdpMid
                    ice_candidate.sdpMLineIndex = msg.sdpMLineIndex

                    if conn is None:
                        pending_ice_candidates.append(ice_candidate)
                    else:
                        await conn.add_or_queue_ice_candidates([ice_candidate])
                elif data_type == "offer":
                    msg = SdpOfferMsg.model_validate(data)

                    # init the connection & send answer SDP
                    conn = WebRTCConnection(
                        offer_sdp=msg.sdp,
                        session_config=msg.session,
                        turn_servers=turn_servers,
                        on_close=state.connections.discard,
                    )

                    await conn.add_or_queue_ice_candidates(pending_ice_candidates)
                    pending_ice_candidates = []

                    answer = await conn.answer()
                    state.connections.add(conn)

                    await ws.send_json({"type": "answer", "sdp": answer.sdp})
            except JSONDecodeError:
                await ws.send_json({"error": "Invalid JSON format"})
                await ws.close()
    except WebSocketDisconnect:
        logger.info("Signalling websocket disconnected")
    except Exception as e:
        logger.error(f"/call error: {e}", exc_info=True)
