#!/usr/bin/env python3
"""Vendor-side tooling. Keep the private key OFF customer machines.

    # once, ever - creates the vendor key pair and embeds the public half
    python scripts/make_license.py --init

    # the customer runs show_fingerprint.py and sends you the value
    python scripts/make_license.py --issue \
        --fingerprint 4f2c... --customer "AIRI HQ" --days 365 \
        --out licence.key

    # encrypt the models with THIS licence's key material
    python scripts/encrypt_models.py --licence licence.key

The private key signs licences. Anyone holding it can mint a licence for any
machine, which is the whole security boundary - it must never be shipped, and
`.gitignore` excludes it. The public half is embedded in
`app/core/licensing.py`, so a customer cannot substitute their own key without
editing the compiled module.

`key_material` inside the licence is what unwraps the encrypted models. It is
random per licence, so two customers cannot share model files, and it is why a
copied installation is useless: the models are ciphertext and their licence
will not verify on the new hardware.
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import re
import secrets
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

PRIVATE_KEY_PATH = ROOT / "vendor_private_key.pem"
LICENSING_PY = ROOT / "app" / "core" / "licensing.py"


def cmd_init(args) -> int:
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

    if PRIVATE_KEY_PATH.exists() and not args.force:
        print(f"  {PRIVATE_KEY_PATH} already exists. --force to replace it, but "
              f"every licence signed with the old key stops verifying.")
        return 1

    priv = Ed25519PrivateKey.generate()
    PRIVATE_KEY_PATH.write_bytes(priv.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    ))
    os.chmod(PRIVATE_KEY_PATH, 0o600)

    pub_b64 = base64.b64encode(priv.public_key().public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )).decode()

    src = LICENSING_PY.read_text()
    src = re.sub(r'^EMBEDDED_PUBKEY_B64 = ".*"$',
                 f'EMBEDDED_PUBKEY_B64 = "{pub_b64}"', src, flags=re.M)
    LICENSING_PY.write_text(src)

    print(f"  private key -> {PRIVATE_KEY_PATH} (mode 600, NEVER ship this)")
    print(f"  public key embedded in app/core/licensing.py")
    print(f"  rebuild the native modules so the new key is compiled in:")
    print(f"    python scripts/build_native.py")
    return 0


def _load_private():
    from cryptography.hazmat.primitives import serialization
    if not PRIVATE_KEY_PATH.is_file():
        raise SystemExit("  no vendor key; run --init first")
    return serialization.load_pem_private_key(PRIVATE_KEY_PATH.read_bytes(),
                                              password=None)


def cmd_issue(args) -> int:
    priv = _load_private()
    now = int(time.time())
    payload = {
        "customer": args.customer,
        "fingerprint": args.fingerprint,
        "issued": now,
        "expires": 0 if args.days <= 0 else now + args.days * 86400,
        "features": args.features.split(",") if args.features else [],
        # 32 random bytes; every model key is derived from this, so it is the
        # difference between a copied models/ directory being useful or not.
        "key_material": base64.b64encode(
            base64.b64decode(args.key_material) if args.key_material
            else secrets.token_bytes(32)
        ).decode(),
    }
    raw = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode()
    blob = {
        "payload": base64.b64encode(raw).decode(),
        "signature": base64.b64encode(priv.sign(raw)).decode(),
    }
    out = Path(args.out)
    out.write_text(json.dumps(blob, indent=2) + "\n")
    exp = "perpetual" if not payload["expires"] else \
        time.strftime("%Y-%m-%d", time.gmtime(payload["expires"]))
    print(f"  licence -> {out}")
    print(f"    customer    {args.customer}")
    print(f"    fingerprint {args.fingerprint}")
    print(f"    expires     {exp}")
    print(f"  encrypt the models for this licence:")
    print(f"    python scripts/encrypt_models.py --licence {out}")
    return 0


def cmd_inspect(args) -> int:
    """Read a licence WITHOUT checking it against this machine."""
    blob = json.loads(Path(args.licence).read_text())
    data = json.loads(base64.b64decode(blob["payload"]))
    data["key_material"] = f"<{len(base64.b64decode(data['key_material']))} bytes, hidden>"
    for k, v in data.items():
        if k in ("issued", "expires") and v:
            v = f"{v}  ({time.strftime('%Y-%m-%d', time.gmtime(v))})"
        print(f"    {k:14s} {v}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd")
    ap.add_argument("--init", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--issue", action="store_true")
    ap.add_argument("--inspect", metavar="LICENCE")
    ap.add_argument("--fingerprint")
    ap.add_argument("--customer", default="")
    ap.add_argument("--days", type=int, default=365, help="0 = perpetual")
    ap.add_argument("--features", default="")
    ap.add_argument("--key-material", default="",
                    help="reuse an existing licence's key material (base64), so "
                         "already-encrypted models keep working")
    ap.add_argument("--out", default="licence.key")
    args = ap.parse_args()

    if args.init:
        return cmd_init(args)
    if args.inspect:
        args.licence = args.inspect
        return cmd_inspect(args)
    if args.issue:
        if not args.fingerprint:
            print("  --issue needs --fingerprint", file=sys.stderr)
            return 2
        return cmd_issue(args)
    ap.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
