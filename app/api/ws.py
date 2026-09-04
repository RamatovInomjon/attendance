"""WebSocket camera streaming, in the shape the original live page expects.

`templates/recognition/live.html` connects to `/ws/camera/<id>/` and draws
`{type:"frame", data:<base64 jpeg>}` messages onto a canvas.  This serves that
contract from the v3 workers.

Each connection renders its own frame from the worker's latest processed frame,
so opening a second viewer does not steal the first one's frames — the previous
implementation had every consumer pulling from one shared queue.
"""
from __future__ import annotations

import asyncio
import base64
import logging

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from app.core.security import COOKIE_NAME, read_session
from app.services import auth as auth_svc

from app.runtime import runtime

log = logging.getLogger(__name__)
router = APIRouter()

FPS = 10


async def _pump(ws: WebSocket, camera_id: int):
    worker = runtime.workers.get(camera_id)
    if worker is None:
        await ws.close(code=4004, reason="camera not running")
        return
    try:
        while True:
            jpg = await asyncio.to_thread(worker.render)
            if jpg:
                await ws.send_json({
                    "type": "frame",
                    "data": base64.b64encode(jpg).decode("ascii"),
                    "camera_id": camera_id,
                    "fps": round(worker.source.fps, 1),
                    "stale": worker.source.is_stale,
                })
            await asyncio.sleep(1 / FPS)
    except WebSocketDisconnect:
        pass
    except Exception as e:                       # client vanished mid-send
        log.debug("ws camera %s ended: %s", camera_id, e)


def _resolve(camera_id: str) -> int | None:
    if camera_id.isdigit():
        return int(camera_id)
    if camera_id in ("primary", "default"):
        ids = sorted(runtime.workers)
        return ids[0] if ids else None
    for w in runtime.workers.values():          # allow /ws/camera/entrance/
        if w.name.lower() == camera_id.lower() or w.role.value.lower() == camera_id.lower():
            return w.camera_id
    return None



async def _authed(ws: WebSocket, *, admin_only: bool = False) -> bool:
    """Reject an unauthenticated socket before accepting it.

    HTTP middleware does not run for WebSocket scopes, so without this check
    the live camera feed and the attendance push stay world-readable even
    though every page around them requires a login. Browsers send cookies on
    the WebSocket handshake, so the same signed session applies.

    1008 is "policy violation"; closing before accept() means no frame is ever
    sent to a client that has not signed in.
    """
    session = read_session(ws.cookies.get(COOKIE_NAME))
    if session is None:
        await ws.close(code=1008, reason="not authenticated")
        return False
    # HTTP middleware does not run for WebSocket scopes, so the admin-only
    # rule that keeps operators off /recognition and /video has to be
    # repeated here - otherwise the live camera feed stays reachable by
    # socket for exactly the accounts the page was hidden from.
    if admin_only and not auth_svc.can_admin(session):
        await ws.close(code=1008, reason="admin only")
        return False
    return True


@router.websocket("/ws/camera/{camera_id}/")
async def camera_ws(ws: WebSocket, camera_id: str):
    if not await _authed(ws, admin_only=True):
        return
    await ws.accept()
    cid = _resolve(camera_id)
    if cid is None:
        await ws.close(code=4004, reason="unknown camera")
        return
    await _pump(ws, cid)


@router.websocket("/ws/camera/{camera_id}")
async def camera_ws_noslash(ws: WebSocket, camera_id: str):
    await camera_ws(ws, camera_id)


@router.websocket("/ws/attendance/")
async def attendance_ws(ws: WebSocket):
    """Pushes recognition events to the live page as they happen."""
    if not await _authed(ws):
        return
    await ws.accept()
    seen: set[tuple] = set()
    try:
        while True:
            for ev in runtime.events(12):
                key = (ev["ts"], ev["name"], ev["camera"])
                if key in seen:
                    continue
                seen.add(key)
                await ws.send_json({"type": "event", **ev})
            if len(seen) > 400:
                seen = set(list(seen)[-200:])
            await asyncio.sleep(2)
    except WebSocketDisconnect:
        pass
    except Exception as e:
        log.debug("attendance ws ended: %s", e)
