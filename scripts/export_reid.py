#!/usr/bin/env python3
"""Export a trained person-ReID checkpoint to a deployable ONNX model.

    python scripts/export_reid.py /media/inomjon/T7/reid/e1_modellar/h100/\
resnet101_ibn_256x128_s0_e2048_es/model_best.pth
    python scripts/export_reid.py <ckpt> --out models/reid_r101ibn_fp16.onnx
    python scripts/export_reid.py <ckpt> --no-fp16

The research checkpoints are `torch.save` dicts carrying everything needed to
rebuild the graph - `arch`, `height`, `width`, `embed_dim`, `num_classes` - so no
config file travels with them. The network is rebuilt with `init="none"`, which
skips the pretrained-backbone lookup: at inference the weights come entirely from
the checkpoint, so neither `zoo/` nor a torch hub cache is needed.

WHY THE PREPROCESSING IS PINNED HERE AND NOT LEFT TO THE CALLER
---------------------------------------------------------------
Nothing in an ONNX graph records that its input must be RGB, squashed to
256x128 without preserving aspect, scaled to [0,1] and then normalised by the
ImageNet mean and standard deviation. Feed it BGR, or letterbox it, or skip the
normalisation, and it returns embeddings of exactly the right shape and norm
that are quietly wrong - and every stored feature and every match is wrong with
them, with nothing in the numbers to say so.

So the contract is written into the exported model's metadata and re-read by
`app/core/reid.py`, and `--verify` checks the exported model against the torch
one on real crops rather than on noise.

The eval-mode graph is a single tensor in, a single L2-normalised tensor out,
with no dynamic control flow, so a plain `torch.onnx.export` is enough - none of
the Reshape/LayerNormalization surgery the face recognizer needed applies.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "vendor_reid"))

# The preprocessing contract, from tools/reid/reid_data.py:120-136.
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


def build(ckpt_path: Path):
    """Rebuild the trained network from the checkpoint alone."""
    import torch
    from reid_model import ReIDNet

    ck = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    for key in ("model", "arch", "height", "width", "num_classes"):
        if key not in ck:
            raise SystemExit(f"  checkpoint is missing {key!r}; not a ReIDNet export")
    arch, h, w = ck["arch"], int(ck["height"]), int(ck["width"])
    emb, ncls = int(ck.get("embed_dim", 512)), int(ck["num_classes"])

    # pretrained=False as well as init="none": the osnet branch would otherwise
    # try to fetch ImageNet weights it is about to overwrite.
    net = ReIDNet(arch, num_classes=ncls, embed_dim=emb, init="none",
                  pretrained=False)
    missing, unexpected = net.load_state_dict(ck["model"], strict=False)
    if missing or unexpected:
        raise SystemExit(
            f"  state_dict does not fit the rebuilt graph:\n"
            f"    missing:    {list(missing)[:6]}\n"
            f"    unexpected: {list(unexpected)[:6]}\n"
            f"  the vendored model definition is out of step with the checkpoint.")
    net.eval()
    return net, {"arch": arch, "height": h, "width": w,
                 "embed_dim": emb, "epoch": ck.get("epoch")}


def preprocess(crops_bgr, h: int, w: int) -> np.ndarray:
    """The one true preprocessing. BGR uint8 crops -> (N,3,h,w) float32.

    Mirrors `T.Compose([Resize((h, w)), ToTensor(), Normalize(mean, std)])` on a
    PIL RGB image. `Resize((h, w))` squashes; it does not preserve aspect ratio.
    """
    import cv2
    out = np.empty((len(crops_bgr), 3, h, w), np.float32)
    mean = np.asarray(MEAN, np.float32).reshape(3, 1, 1)
    std = np.asarray(STD, np.float32).reshape(3, 1, 1)
    for i, bgr in enumerate(crops_bgr):
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        # cv2.resize takes (width, height); torchvision Resize takes (h, w).
        rgb = cv2.resize(rgb, (w, h), interpolation=cv2.INTER_LINEAR)
        chw = rgb.transpose(2, 0, 1).astype(np.float32) / 255.0
        out[i] = (chw - mean) / std
    return out


def real_crops(n: int, h: int, w: int):
    """Verify on real person crops, not noise - a preprocessing error can look
    harmless on random input and still wreck the embedding of an actual body."""
    import cv2
    pool = sorted((ROOT / "data").glob("reid_passes/*/*.jpg"))
    if not pool:
        pool = sorted((ROOT / "data").glob("persons*/*/*.jpg"))
    crops = []
    for p in pool[:: max(1, len(pool) // max(n, 1))]:
        img = cv2.imread(str(p))
        if img is not None:
            crops.append(img)
        if len(crops) >= n:
            break
    return crops


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("ckpt")
    ap.add_argument("--out", default=None)
    ap.add_argument("--no-fp16", action="store_true")
    ap.add_argument("--opset", type=int, default=17)
    args = ap.parse_args()

    import torch

    ckpt = Path(args.ckpt)
    if not ckpt.is_file():
        raise SystemExit(f"  no such checkpoint: {ckpt}")
    net, meta = build(ckpt)
    h, w, emb = meta["height"], meta["width"], meta["embed_dim"]
    print(f"  {ckpt}")
    print(f"  {meta['arch']}  {h}x{w}  embed={emb}  epoch={meta['epoch']}")

    out = Path(args.out) if args.out else (
        ROOT / "models" /
        f"reid_{meta['arch']}_{h}x{w}_e{emb}"
        f"{'' if args.no_fp16 else '_fp16'}.onnx")
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".fp32.onnx") if not args.no_fp16 else out

    torch.onnx.export(
        net, torch.randn(1, 3, h, w), str(tmp),
        input_names=["images"], output_names=["embedding"],
        dynamic_axes={"images": {0: "batch"}, "embedding": {0: "batch"}},
        opset_version=args.opset, do_constant_folding=True)
    print(f"  exported fp32 -> {tmp.name}")

    import onnx
    model = onnx.load(str(tmp))
    if not args.no_fp16:
        from onnxconverter_common import float16
        del model.graph.value_info[:]
        model = float16.convert_float_to_float16(model, keep_io_types=True)
        del model.graph.value_info[:]
        print("  converted to fp16 (fp32 in/out)")

    # The preprocessing convention travels WITH the weights. A reader that has
    # to guess it will guess wrong silently.
    for k, v in (("reid_arch", meta["arch"]), ("reid_height", str(h)),
                 ("reid_width", str(w)), ("reid_embed_dim", str(emb)),
                 ("reid_colour", "RGB"), ("reid_resize", "squash"),
                 ("reid_mean", json.dumps(MEAN)), ("reid_std", json.dumps(STD)),
                 ("reid_normalised_output", "true"),
                 ("reid_source_checkpoint", str(ckpt))):
        e = model.metadata_props.add()
        e.key, e.value = k, v
    onnx.save(model, str(out))
    if tmp != out:
        tmp.unlink(missing_ok=True)
    print(f"  -> {out}  ({out.stat().st_size / 1e6:.0f} MB)")

    # ---- verify against torch, on real crops ---------------------------
    crops = real_crops(32, h, w)
    if not crops:
        print("  NO CROPS FOUND to verify against - do not deploy this unverified")
        return 1
    batch = preprocess(crops, h, w)
    with torch.no_grad():
        ref = net(torch.from_numpy(batch)).numpy()

    from app.core.onnx_env import preload_cuda_libs
    preload_cuda_libs()
    import onnxruntime as ort
    so = ort.SessionOptions()
    so.log_severity_level = 3
    sess = ort.InferenceSession(str(out), sess_options=so,
                                providers=["CUDAExecutionProvider",
                                           "CPUExecutionProvider"])
    got = sess.run(None, {"images": batch})[0].astype(np.float32)

    cos = np.einsum("ij,ij->i", ref / np.linalg.norm(ref, axis=1, keepdims=True),
                    got / np.linalg.norm(got, axis=1, keepdims=True))
    print(f"  verify on {len(crops)} real crops: min cosine vs torch "
          f"{cos.min():.6f}  mean {cos.mean():.6f}")
    print(f"  output norms: {np.linalg.norm(got, axis=1).min():.4f} .. "
          f"{np.linalg.norm(got, axis=1).max():.4f}  (should be ~1.0)")
    if cos.min() < 0.999:
        print("  EXPORT REJECTED. Removing the output so it cannot be shipped.")
        out.unlink(missing_ok=True)
        return 1

    import time
    for b in (1, 4, 8):
        feed = {"images": np.repeat(batch[:1], b, axis=0)}
        for _ in range(5):
            sess.run(None, feed)
        t = time.perf_counter()
        for _ in range(30):
            sess.run(None, feed)
        ms = (time.perf_counter() - t) / 30 * 1000
        print(f"    batch {b}: {ms:6.2f} ms  ({ms / b:5.2f} ms/crop)")
    print("  verified")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
