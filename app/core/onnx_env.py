"""Make onnxruntime-gpu find its CUDA libraries, regardless of import order.

onnxruntime-gpu dlopens `libonnxruntime_providers_cuda.so`, which links against
`libcublasLt.so.12`, `libcudnn.so.9` and friends.  On this machine those live
inside the pip `nvidia-*` wheels rather than on the system linker path, so the
dlopen only succeeds if something has already pulled them into the process.

Importing torch happens to do that, which meant the pipeline worked purely
because `detector.py` (-> ultralytics -> torch) was imported before any ONNX
session was created.  Import the aligner on its own and the CUDA provider
silently fails, leaving onnxruntime on CPU - a ~10x slowdown that reports
itself only as "CPUExecutionProvider" in the logs.

Preloading the libraries explicitly removes the ordering dependency, and
`require_cuda()` turns a silent CPU fallback into a loud failure.
"""
from __future__ import annotations

import ctypes
import logging
import os
import site
import sys
from pathlib import Path

log = logging.getLogger(__name__)

# Order matters: dependencies before dependents.
_PRELOAD = (
    "nvidia/cuda_runtime/lib/libcudart.so.12",
    "nvidia/cublas/lib/libcublas.so.12",
    "nvidia/cublas/lib/libcublasLt.so.12",
    "nvidia/cufft/lib/libcufft.so.11",
    "nvidia/curand/lib/libcurand.so.10",
    "nvidia/cusolver/lib/libcusolver.so.11",
    "nvidia/cusparse/lib/libcusparse.so.12",
    "nvidia/nvjitlink/lib/libnvJitLink.so.12",
    "nvidia/cudnn/lib/libcudnn.so.9",
)

_done = False


def _site_dirs() -> list[Path]:
    dirs = []
    try:
        dirs += [Path(p) for p in site.getsitepackages()]
    except Exception:
        pass
    usr = site.getusersitepackages() if hasattr(site, "getusersitepackages") else None
    if isinstance(usr, str):
        dirs.append(Path(usr))
    dirs.append(Path(sys.prefix) / "lib" / f"python{sys.version_info.major}.{sys.version_info.minor}" / "site-packages")
    return [d for d in dirs if d.is_dir()]


def preload_cuda_libs() -> int:
    """Load the CUDA shared objects into this process. Returns how many loaded."""
    global _done
    if _done:
        return 0
    loaded = 0
    for base in _site_dirs():
        for rel in _PRELOAD:
            p = base / rel
            if not p.exists():
                continue
            try:
                ctypes.CDLL(str(p), mode=ctypes.RTLD_GLOBAL)
                loaded += 1
            except OSError as e:
                log.debug("preload failed for %s: %s", p, e)
        if loaded:
            # Also help the provider .so find its own siblings.
            extra = os.pathsep.join(str((base / rel).parent) for rel in _PRELOAD
                                    if (base / rel).exists())
            if extra:
                os.environ["LD_LIBRARY_PATH"] = extra + os.pathsep + os.environ.get("LD_LIBRARY_PATH", "")
            break
    _done = True
    return loaded


def available_providers() -> list[str]:
    preload_cuda_libs()
    import onnxruntime as ort
    return ort.get_available_providers()


def best_providers(prefer_cuda: bool = True) -> list:
    preload_cuda_libs()
    import onnxruntime as ort
    avail = ort.get_available_providers()
    out = []
    if prefer_cuda and "CUDAExecutionProvider" in avail:
        out.append(("CUDAExecutionProvider", {"device_id": 0}))
    out.append("CPUExecutionProvider")
    return out


def require_cuda(session, name: str = "model") -> None:
    """Fail loudly rather than running 10x slower without anyone noticing."""
    provs = session.get_providers()
    if "CUDAExecutionProvider" not in provs:
        raise RuntimeError(
            f"{name}: CUDA execution provider unavailable (got {provs}). "
            "Check that onnxruntime-gpu is installed WITHOUT the CPU-only "
            "'onnxruntime' package alongside it, and that the nvidia-* CUDA 12 "
            "and cuDNN 9 wheels are present. See docs/OPERATIONS.md."
        )
