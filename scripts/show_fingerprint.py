#!/usr/bin/env python3
"""Print this machine's fingerprint. Send it to the vendor to get a licence.

    python scripts/show_fingerprint.py

Reveals no secrets: the fingerprint is a salted SHA-256 of the GPU UUID and
/etc/machine-id, so it identifies the host without publishing its serials.
"""
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.core import licensing

def main() -> int:
    try:
        rep = licensing.fingerprint_report()
    except licensing.LicenceError as e:
        print(f"  {e}", file=sys.stderr)
        return 1
    print(json.dumps(rep, indent=2))
    print("\n  Send the 'fingerprint' value to the vendor.")
    p = licensing.default_path()
    if p.is_file():
        try:
            lic = licensing.verify(p)
            left = "perpetual" if not lic.expires else f"{lic.days_left:.0f} days left"
            print(f"  Licence at {p}: VALID for {lic.customer!r} ({left})")
        except licensing.LicenceError as e:
            print(f"  Licence at {p}: INVALID - {e}")
    else:
        print(f"  No licence installed at {p}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
