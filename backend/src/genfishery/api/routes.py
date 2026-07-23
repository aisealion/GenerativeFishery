"""GET /fisheries/{id}/state (snapshot) and WS /fisheries/{id}/stream (live
event feed) -- build spec §9. The WS stream carries only what changed *after*
connecting; a client is expected to call the GET snapshot first. Both routes
are pollable/streamable independently per fishery (build spec §1: fisheries
don't share a turn order), looked up by id from `app.state.runners`.
"""

from fastapi import APIRouter, HTTPException, Request, WebSocket, WebSocketDisconnect

from genfishery.api.runner import FisheryRunner
from genfishery.api.serialization import serialize_state

router = APIRouter()


def _find_runner(request_or_ws, fishery_id: str) -> FisheryRunner | None:
    return request_or_ws.app.state.runners.get(fishery_id)


@router.get("/fisheries/{fishery_id}/state")
async def get_fishery_state(fishery_id: str, request: Request) -> dict:
    runner = _find_runner(request, fishery_id)
    if runner is None:
        raise HTTPException(status_code=404, detail=f"unknown fishery_id: {fishery_id!r}")
    return serialize_state(runner.state)


@router.websocket("/fisheries/{fishery_id}/stream")
async def stream_fishery_events(websocket: WebSocket, fishery_id: str) -> None:
    runner = _find_runner(websocket, fishery_id)
    if runner is None:
        await websocket.close(code=4404, reason=f"unknown fishery_id: {fishery_id!r}")
        return

    await websocket.accept()
    notify_bridge = websocket.app.state.notify_bridge
    queue = notify_bridge.subscribe(fishery_id)
    try:
        while True:
            event = await queue.get()
            await websocket.send_json(event)
    except WebSocketDisconnect:
        pass
    finally:
        notify_bridge.unsubscribe(fishery_id, queue)
