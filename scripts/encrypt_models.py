#!/usr/bin/env python3
"""Encrypt the runtime models for one licence.

    python scripts/encrypt_models.py --licence licence.key           # write .enc
    python scripts/encrypt_models.py --licence licence.key --remove-plain
    python scripts/encrypt_models.py --verify --licence licence.key  # round-trip

Writes `<model>.onnx.enc` beside each model. `load_model()` prefers the .enc, so
a release directory containing ONLY .enc files works with no config change.

--remove-plain deletes the originals. Do that when building a release bundle,
never on the machine holding your only copy of the weights.
"""
import argparse, base64, json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from app.core.model_vault import encrypt_bytes, decrypt_bytes, ENC_SUFFIX

RUNTIME_MODELS = ("yolov8n_head_960x544_fp16.onnx", "dfa_mobilenet_aligner.onnx",
                  "adaface_ir101_finetune_fp16.onnx")

def key_material_of(licence_path: Path) -> bytes:
    blob = json.loads(licence_path.read_text())
    data = json.loads(base64.b64decode(blob["payload"]))
    km = base64.b64decode(data.get("key_material", ""))
    if not km:
        raise SystemExit("  licence carries no key material")
    return km

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--licence", required=True, type=Path)
    ap.add_argument("--models-dir", type=Path, default=ROOT / "models")
    ap.add_argument("--remove-plain", action="store_true")
    ap.add_argument("--verify", action="store_true")
    a = ap.parse_args()

    km = key_material_of(a.licence)
    total = 0
    for name in RUNTIME_MODELS:
        src = a.models_dir / name
        if not src.is_file():
            print(f"  missing, skipped: {name}")
            continue
        plain = src.read_bytes()
        dst = src.with_name(src.name + ENC_SUFFIX)
        if a.verify:
            if not dst.is_file():
                print(f"  {name}: no .enc to verify"); continue
            back = decrypt_bytes(dst.read_bytes(), km, name)
            ok = back == plain
            print(f"  {'OK  ' if ok else 'FAIL'} {name}: round-trip {'matches' if ok else 'DIFFERS'}")
            if not ok: return 1
            continue
        dst.write_bytes(encrypt_bytes(plain, km, name))
        total += dst.stat().st_size
        print(f"  encrypted {name} -> {dst.name}  ({dst.stat().st_size/1e6:.1f} MB)")
        if a.remove_plain:
            src.unlink(); print(f"    removed plaintext {name}")
    if not a.verify:
        print(f"  {total/1e6:.1f} MB encrypted")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
