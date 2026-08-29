#!/usr/bin/env python3
"""Make an exported recognizer deployable: dynamic batch, then fp16.

    python scripts/convert_recognizer.py models/adaface_vit_kprpe.onnx
    python scripts/convert_recognizer.py <src> --out models/foo_fp16.onnx
    python scripts/convert_recognizer.py <src> --no-fp16      # batch only

Research exports are shaped for a single image at a time in fp32. That is the
worst case for this pipeline, which embeds a batch of faces per frame: batch 1
forfeits all GPU parallelism, and fp32 doubles both the weights and the
bandwidth. Converting `adaface_vit_kprpe.onnx` took it from 7.12 ms/face and
461 MB to 2.98 ms/face and 231 MB - 2.4x faster - with d-prime moving 12.222 ->
12.219, which is nothing.

Both steps have a trap that fails in a way you would not guess from the error.

DYNAMIC BATCH is not just renaming the input dimension. A ViT export bakes the
batch size into its Reshape targets - 74 of them here, inside attention - so
renaming the graph input alone produces:

    Input shape:{4,196,1536}, requested shape:{1,196,3,16,32}

The targets have to be repinned. The obvious fix, setting the leading value to
-1, is wrong: many targets already contain a -1 elsewhere and ONNX rejects two
in one shape. Use **0**, which in Reshape means "copy this dimension from the
input" - exact, and legal beside an existing -1. It is only valid while
`allowzero` is unset, so that is checked.

FP16 must exclude LayerNormalization. The converter rewrites its weights but
mishandles the type binding of the fused op, and the result does not load:

    Type parameter (T) of Optype (LayerNormalization) bound to different types

Blocking it leaves 49 small ops in fp32, which costs nothing measurable.

Everything is verified against the original graph before it is written, because
both failure modes CAN produce a model that loads and returns silently wrong
embeddings - which would corrupt every enrolment and every match, invisibly.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def make_batch_dynamic(model, verbose: bool = True):
    """Free the batch dimension, including the Reshape targets that pin it."""
    import onnx
    from onnx import numpy_helper as nh

    blocked = [n.name for n in model.graph.node if n.op_type == "Reshape"
               for a in n.attribute if a.name == "allowzero" and a.i == 1]
    if blocked:
        raise SystemExit(
            f"  {len(blocked)} Reshape node(s) set allowzero=1, so a leading 0 "
            f"would mean a literal zero rather than 'copy the input dim'. "
            f"This model needs a different approach.")

    for t in list(model.graph.input) + list(model.graph.output):
        d = t.type.tensor_type.shape.dim[0]
        d.ClearField("dim_value")
        d.dim_param = "batch"

    init = {i.name: i for i in model.graph.initializer}
    const = {n.output[0]: a for n in model.graph.node if n.op_type == "Constant"
             for a in n.attribute if a.name == "value"}
    targets = {n.input[1] for n in model.graph.node
               if n.op_type == "Reshape" and len(n.input) > 1}

    patched = 0
    for name in targets:
        if name in init:
            arr = nh.to_array(init[name]).copy()
            if arr.ndim == 1 and len(arr) and arr[0] == 1:
                arr[0] = 0
                init[name].CopyFrom(nh.from_array(arr, name))
                patched += 1
        elif name in const:
            attr = const[name]
            arr = nh.to_array(attr.t).copy()
            if arr.ndim == 1 and len(arr) and arr[0] == 1:
                arr[0] = 0
                attr.t.CopyFrom(nh.from_array(arr, attr.t.name))
                patched += 1
    if verbose:
        print(f"  batch: freed; {patched} of {len(targets)} Reshape targets repinned")
    return model


def to_fp16(model, verbose: bool = True):
    import onnx
    from onnxconverter_common import float16

    # Stale fp32 type annotations make ORT reject the converted graph.
    del model.graph.value_info[:]
    out = float16.convert_float_to_float16(
        model, keep_io_types=True,
        # LayerNormalization: see the module docstring. Without this the model
        # does not load at all.
        op_block_list=list(float16.DEFAULT_OP_BLOCK_LIST) + ["LayerNormalization"])
    del out.graph.value_info[:]
    if verbose:
        print("  fp16: converted (LayerNormalization left in fp32)")
    return out


def _feeds(sess, batch: int, seed: int = 0):
    """Random inputs matching whatever the graph declares."""
    rng = np.random.default_rng(seed)
    out = {}
    for i in sess.get_inputs():
        shape = [batch] + [d if isinstance(d, int) else 1 for d in i.shape[1:]]
        dt = np.float16 if "float16" in i.type else np.float32
        # keypoints live in [0, 1]; images are standardised
        arr = (rng.random(shape) if "key" in i.name.lower()
               else rng.standard_normal(shape))
        out[i.name] = arr.astype(dt)
    return out


def verify(src: str, dst: str, batches=(1, 4, 8)) -> bool:
    """Prove the converted graph answers the same as the original.

    Both conversions can yield a model that loads and returns wrong numbers, so
    this is not optional: a silent change here corrupts every enrolment vector
    and every live match at once, with nothing in the data to reveal it.
    """
    from app.core.onnx_env import preload_cuda_libs
    preload_cuda_libs()
    import onnxruntime as ort

    so = ort.SessionOptions()
    so.log_severity_level = 3
    prov = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    ref = ort.InferenceSession(src, sess_options=so, providers=prov)
    try:
        got = ort.InferenceSession(dst, sess_options=so, providers=prov)
    except Exception as e:
        print(f"  VERIFY FAILED - converted model does not load:\n    {str(e)[:200]}")
        return False

    ok = True
    for b in batches:
        feeds = _feeds(got, b)
        # the reference takes one row at a time
        single = []
        for i in range(b):
            one = {k: v[i:i + 1].astype(np.float32) for k, v in feeds.items()}
            single.append(ref.run(None, one)[0][0])
        single = np.stack(single).astype(np.float32)
        try:
            batched = got.run(None, feeds)[0].astype(np.float32)
        except Exception as e:
            print(f"  batch {b}: FAILED {str(e)[:140]}")
            ok = False
            continue
        cos = min(float(a @ b_ / (np.linalg.norm(a) * np.linalg.norm(b_) + 1e-9))
                  for a, b_ in zip(single, batched))
        verdict = "ok" if cos > 0.999 else "** DIFFERS"
        print(f"  batch {b}: min cosine vs original {cos:.6f}  {verdict}")
        ok &= cos > 0.999
    return ok


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("src")
    ap.add_argument("--out", default=None)
    ap.add_argument("--no-fp16", action="store_true")
    ap.add_argument("--no-verify", action="store_true")
    args = ap.parse_args()

    import onnx
    src = Path(args.src)
    if not src.exists():
        raise SystemExit(f"  no such model: {src}")
    dst = Path(args.out) if args.out else src.with_name(
        src.stem + ("_dyn" if args.no_fp16 else "_fp16") + ".onnx")

    print(f"  {src}  ({src.stat().st_size/1e6:.0f} MB)")
    m = onnx.load(str(src))
    m = make_batch_dynamic(m)
    if not args.no_fp16:
        m = to_fp16(m)
    onnx.save(m, str(dst))
    print(f"  -> {dst}  ({dst.stat().st_size/1e6:.0f} MB)")

    if args.no_verify:
        print("  verification SKIPPED - do not deploy an unverified conversion")
        return 0
    if not verify(str(src), str(dst)):
        print("  CONVERSION REJECTED. Removing the output so it cannot be shipped.")
        dst.unlink(missing_ok=True)
        return 1
    print("  verified")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
