from pydantic import BaseModel

from src.clients.http import http_client
from src.env import env
from src.logger import get_logger


class TurnServer(BaseModel):
    urls: list[str]
    username: str | None = None
    credential: str | None = None


URL = f"https://rtc.live.cloudflare.com/v1/turn/keys/{env.CF_TURN_TOKEN_ID}/credentials/generate-ice-servers"

logger = get_logger(__name__)


async def get_turn_servers(ttl: int = 600) -> list[TurnServer]:
    """
    Get a list of TURN servers from Cloudflare Realtime TURN service.

    Args:
        ttl (int, optional): Time-to-live for the generated credentials. Defaults to 86400 seconds (one day).
    """

    try:
        resp = await http_client.post(
            URL,
            headers={
                "Authorization": f"Bearer {env.CF_TURN_API_TOKEN}",
                "Content-Type": "application/json",
            },
            json={
                "ttl": ttl,
            },
        )

        resp.raise_for_status()

        resp = resp.json()
        assert isinstance(resp, dict)
        ice_servers = resp.get("iceServers")
        assert isinstance(ice_servers, list)

        turn_servers: list[TurnServer] = []
        for ts in ice_servers:
            try:
                turn_servers.append(TurnServer.model_validate(ts))
            except Exception as e:
                logger.error(f"Invalid turn server: {e}", exc_info=True)

        return turn_servers
    except Exception as e:
        raise e
