"""Build the local face gallery from the `face_id_users/` export.

Layout consumed:

    face_id_users/
      users.json                     master index
      NNN_<user_id>/metadata.json    per-person record
      NNN_<user_id>/image_01..05.png 1280x720 enrolment frames

One `FaceEmbedding` row per usable image — not an averaged centroid.  Matching
on max-similarity across a person's images handles pose and lighting spread
better than one mean vector, and it keeps a bad enrolment photo visible instead
of quietly dragging the centroid toward a different face.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
from sqlalchemy import delete, select

from app.config import settings
from app.core.aligner import FaceAligner
from app.core.detector import build_detector
from app.core.gallery import Gallery
from app.core.quality import estimate_pose, sharpness_of
from app.core.recognizer import FaceRecognizer
from app.db.models import Employee, FaceEmbedding
from app.db.session import session_scope

log = logging.getLogger(__name__)


@dataclass
class EnrollReport:
    people: int = 0
    images_seen: int = 0
    embedded: int = 0
    no_face: list[str] = field(default_factory=list)
    no_align: list[str] = field(default_factory=list)
    outliers: list[tuple[str, float]] = field(default_factory=list)

    def summary(self) -> str:
        return (f"{self.people} people, {self.embedded}/{self.images_seen} images embedded; "
                f"{len(self.no_face)} no-face, {len(self.no_align)} no-align, "
                f"{len(self.outliers)} outliers")


class Enroller:
    def __init__(self, detector=None, aligner=None, recognizer=None):
        self.detector = detector or build_detector(
            settings.detector_kind,
            settings.model_path(settings.detector_model),
            imgsz=settings.detect_width,
            conf=settings.detect_conf,
        )
        self.aligner = aligner or FaceAligner(
            settings.model_path(settings.aligner_model),
            crop_size=settings.align_crop_size,
            margin=settings.align_margin,
            mode=settings.align_mode,
        )
        self.recognizer = recognizer or FaceRecognizer(
            settings.model_path(settings.recognizer_model), batch_size=settings.embed_batch
        )

    # -- one image --------------------------------------------------------
    def embed_file(self, path: Path):
        """-> (embedding, aligner_score, sharpness, face_px) or None."""
        bgr = cv2.imread(str(path))
        if bgr is None:
            return None
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        dets = self.detector.detect(bgr)
        if not dets:
            return None
        d = max(dets, key=lambda x: (x.box[2] - x.box[0]) * (x.box[3] - x.box[1]))

        faces = self.aligner.align(rgb, [d.box])
        if not faces:
            return None
        f = faces[0]

        emb = self.recognizer.embed(f.aligned[None])[0]
        face_px = float(max(d.box[2] - d.box[0], d.box[3] - d.box[1]))
        return emb, f.score, sharpness_of(f.aligned), face_px

    # -- full gallery -----------------------------------------------------
    def run(self, gallery_dir: Path | None = None, outlier_threshold: float = 0.35,
            wipe: bool = True) -> EnrollReport:
        gallery_dir = Path(gallery_dir or settings.gallery_dir)
        rep = EnrollReport()
        model_name = settings.recognizer_model

        folders = sorted(
            p for p in gallery_dir.iterdir()
            if p.is_dir() and not p.name.startswith("_")
        )

        with session_scope() as s:
            if wipe:
                s.execute(delete(FaceEmbedding))
                s.flush()

            for folder in folders:
                meta_path = folder / "metadata.json"
                meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
                ext_id = meta.get("user_id") or folder.name.split("_", 1)[-1]

                emp = s.execute(
                    select(Employee).where(Employee.external_id == ext_id)
                ).scalar_one_or_none()
                if emp is None:
                    emp = Employee(external_id=ext_id)
                    s.add(emp)
                emp.folder = folder.name
                emp.full_name = meta.get("full_name") or ext_id
                emp.department = meta.get("department", "") or ""
                emp.position = meta.get("position", "") or ""
                emp.phone = meta.get("phone", "") or ""
                emp.user_type = meta.get("user_type", "") or ""
                emp.is_active = True
                s.flush()

                rows, vecs = [], []
                for img in sorted(folder.glob("*.png")):
                    rep.images_seen += 1
                    out = self.embed_file(img)
                    if out is None:
                        rep.no_face.append(f"{folder.name}/{img.name}")
                        continue
                    emb, ascore, sharp, face_px = out
                    rows.append((img.name, emb, ascore))
                    vecs.append(emb)

                # Flag images that disagree with the rest of their own folder.
                if len(vecs) >= 3:
                    V = np.stack(vecs)
                    S = V @ V.T
                    np.fill_diagonal(S, np.nan)
                    cohesion = np.nanmean(S, axis=1)
                    for (name, _e, _q), c in zip(rows, cohesion):
                        if c < outlier_threshold:
                            rep.outliers.append((f"{folder.name}/{name}", float(c)))

                for name, emb, ascore in rows:
                    s.add(FaceEmbedding(
                        employee_id=emp.id, source_file=name,
                        vector=emb.astype(np.float32).tobytes(), dim=int(emb.shape[0]),
                        model_name=model_name, quality=float(ascore),
                    ))
                    rep.embedded += 1
                rep.people += 1
                log.info("enrolled %-28s %d images", folder.name, len(rows))

        return rep


def load_gallery() -> Gallery:
    """Read every embedding out of the database into one matrix."""
    with session_scope() as s:
        rows = s.execute(
            select(FaceEmbedding.employee_id, FaceEmbedding.vector,
                   Employee.full_name, Employee.is_active)
            .join(Employee, Employee.id == FaceEmbedding.employee_id)
            .where(Employee.is_active.is_(True))
        ).all()

    if not rows:
        return Gallery(np.zeros((0, 512), np.float32), np.zeros((0,), np.int64), {})

    vecs = np.stack([np.frombuffer(r[1], dtype=np.float32) for r in rows])
    owners = np.array([r[0] for r in rows], dtype=np.int64)
    names = {int(r[0]): r[2] for r in rows}
    return Gallery(vecs, owners, names)
