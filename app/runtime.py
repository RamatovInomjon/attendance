"""Process-wide runtime: owns the gallery and the camera workers."""
from __future__ import annotations

import logging

from sqlalchemy import select

from app.core.direction import config_from_camera
from app.core.gallery import Gallery
from app.db.models import Camera
from app.db.session import init_db, session_scope
from app.services.enrollment import load_gallery
from app.services.worker import CameraWorker

log = logging.getLogger(__name__)


class Runtime:
    def __init__(self):
        self.gallery: Gallery | None = None
        self.workers: dict[int, CameraWorker] = {}

    def start(self):
        init_db()

        # Fail loudly if we are about to run ~10x slower on CPU.  The failure
        # mode this guards against is silent: onnxruntime just reports
        # CPUExecutionProvider and everything still "works".
        from app.core.onnx_env import available_providers, preload_cuda_libs
        preload_cuda_libs()
        provs = available_providers()
        if "CUDAExecutionProvider" not in provs:
            log.error("CUDA execution provider NOT available (%s) - see docs/OPERATIONS.md", provs)
        else:
            log.info("onnxruntime providers: %s", provs[:2])
        self.gallery = load_gallery()
        log.info("gallery: %d embeddings / %d people", len(self.gallery), self.gallery.n_people)
        if len(self.gallery) == 0:
            log.warning("gallery is empty - run scripts/enroll.py")

        with session_scope() as s:
            cams = [
                (c.id, c.name, c.role, c.rtsp_url, config_from_camera(c))
                for c in s.execute(select(Camera).where(Camera.enabled.is_(True))).scalars()
            ]
        for cid, name, role, url, dcfg in cams:
            w = CameraWorker(cid, name, role, url, self.gallery, direction_cfg=dcfg)
            self.workers[cid] = w.start()
            if dcfg.configured:
                log.info("started worker %s (%s) with direction line", name, role.value)
            else:
                log.warning("started worker %s (%s) WITHOUT a direction line - "
                            "falling back to camera role, which cannot tell a person "
                            "walking out from one walking in. Run scripts/set_direction.py",
                            name, role.value)

    def stop(self):
        for w in self.workers.values():
            w.stop()
        self.workers.clear()

    def reload_gallery(self):
        self.gallery = load_gallery()
        for w in self.workers.values():
            w.pipeline.gallery = self.gallery
        return self.gallery

    def events(self, limit: int = 40) -> list[dict]:
        out: list[dict] = []
        for w in self.workers.values():
            out.extend(w.recent_events)
        out.sort(key=lambda e: e["ts"], reverse=True)
        return out[:limit]


runtime = Runtime()
