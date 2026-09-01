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

def runtime_models() -> tuple[str, ...]:
    """The models the app actually loads, read from the config.

    This was a hardcoded tuple, and a hardcoded tuple silently goes stale: it
    still named the old recognizer after `recognizer_model` had been changed,
    so the new one was never encrypted. On a licensed deployment only `.enc`
    files are shipped, so the service would have started, found no recognizer
    it could load, and failed - at the customer's site, not here.

    Derived from `settings` instead, so it cannot fall behind the config.
    """
    from app.config import settings
    names = [settings.head_model, settings.aligner_model, settings.recognizer_model]
    if settings.reid_model:
        names.append(settings.reid_model)
    # Refresh any OTHER recognizer that has already been encrypted once, so a
    # rollback target stays deployable. Deliberately not "every model with a
    # calibrated threshold": that list includes the fp32 research masters,
    # which are 460 MB, exist nowhere else, and must never reach a customer
    # bundle - `package_release.py` refuses plaintext .onnx for that reason.
    for extra in settings.recognizer_thresholds:
        if extra not in names and (settings.models_dir / (extra + ".enc")).is_file():
            names.append(extra)
    seen, out = set(), []
    for n in names:
        n = str(n)
        if n and n not in seen:
            seen.add(n)
            out.append(n)
    return tuple(out)

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
    for name in runtime_models():
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
