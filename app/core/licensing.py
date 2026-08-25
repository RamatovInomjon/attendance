"""Machine binding: fingerprint this host, verify a signed licence.

## What this does and does not achieve

It stops a customer copying the install to a second machine. That is the
realistic threat and this handles it completely: the licence is Ed25519-signed
by a private key that never leaves the vendor, so a customer cannot mint one for
new hardware.

It does **not** stop a determined analyst. The check runs on their CPU, so it
can be patched out - the branch is a conditional jump like any other. Nothing
that runs on hardware someone else controls can prevent that. What raises the
cost is that the *models are encrypted* with a key derived from the licence
(see `model_vault.py`), so patching the check leaves them with ciphertext they
still cannot decrypt.

## Why the GPU UUID, not the MAC address

MAC was the obvious choice and it is the wrong one:

* This host has two NICs (`eno1` down, `wlp4s0` up) - there is no single "the"
  MAC, and picking one means the licence breaks when a cable moves.
* `ip link set dev eth0 address ...` spoofs it in one command.
* Docker, VPNs, dock stations and USB adapters add and remove interfaces, which
  turns into support calls where the software stopped working for no visible
  reason.

The GPU UUID (`GPU-bd3851ec-...`) is stable, readable without root, and already
a hard requirement - the pipeline refuses to start without CUDA. Changing it
means physically swapping the card. `/etc/machine-id` is mixed in as a second
factor; it is editable, but it costs an attacker nothing to have to notice it.
"""
from __future__ import annotations

import base64
import hashlib
import json
import logging
import os
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger(__name__)

LICENCE_ENV = "EMATSY_LICENCE"          # path override, for odd deployments
LICENCE_NAME = "licence.key"


class LicenceError(RuntimeError):
    """Refused. The message is safe to show an operator."""


# ---- machine fingerprint ----------------------------------------------

def gpu_uuid() -> str | None:
    """The CUDA device UUID, or None when nvidia-smi cannot answer."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10,
        )
        if out.returncode == 0:
            first = out.stdout.strip().splitlines()[0].strip()
            return first or None
    except (OSError, subprocess.SubprocessError, IndexError):
        pass
    return None


def machine_id() -> str | None:
    for p in ("/etc/machine-id", "/var/lib/dbus/machine-id"):
        try:
            v = Path(p).read_text().strip()
            if v:
                return v
        except OSError:
            continue
    return None


def fingerprint() -> str:
    """A stable, non-reversible id for this host.

    Hashed rather than raw so a licence file does not publish the customer's
    hardware serial numbers, and salted with a constant so a fingerprint from
    this product cannot be compared against one from another.
    """
    gpu = gpu_uuid()
    if not gpu:
        raise LicenceError(
            "Cannot read the GPU UUID (nvidia-smi unavailable). This build is "
            "licensed to specific hardware and cannot verify which machine it "
            "is running on."
        )
    parts = ["ematsy-v3", gpu, machine_id() or "no-machine-id"]
    digest = hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()
    return digest[:32]


def fingerprint_report() -> dict:
    """What to send the vendor when requesting a licence."""
    return {
        "fingerprint": fingerprint(),
        "gpu_uuid": gpu_uuid(),
        "machine_id_present": machine_id() is not None,
        "hostname": os.uname().nodename,
    }


# ---- licence file ------------------------------------------------------

@dataclass(frozen=True)
class Licence:
    customer: str
    fingerprint: str
    issued: int
    expires: int              # unix seconds; 0 = perpetual
    features: tuple[str, ...]
    key_material: bytes       # unwraps the encrypted models

    @property
    def days_left(self) -> float | None:
        if not self.expires:
            return None
        return (self.expires - time.time()) / 86400.0


def default_path() -> Path:
    env = os.environ.get(LICENCE_ENV)
    if env:
        return Path(env)
    return Path(__file__).resolve().parent.parent.parent / LICENCE_NAME


def _public_key():
    """The vendor's Ed25519 public key.

    Embedded rather than loaded from a file so it cannot be swapped for an
    attacker's key without editing the binary. Overridable by env only to make
    development and key rotation possible - a release build should have this
    baked in and the env path is itself a documented weak point.
    """
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

    raw = os.environ.get("EMATSY_LICENCE_PUBKEY", "").strip()
    if not raw:
        raw = EMBEDDED_PUBKEY_B64
    if not raw:
        raise LicenceError(
            "No licence public key is embedded in this build. Run "
            "scripts/make_license.py --init to generate a vendor key pair."
        )
    return Ed25519PublicKey.from_public_bytes(base64.b64decode(raw))


# Replaced at release time by scripts/make_license.py --init.
EMBEDDED_PUBKEY_B64 = "hTm2ToQMa6zOIuNMyQhEvFcGD5LFEDIKF4zXyBDDh+8="


def verify(path: Path | None = None, *, expect_fingerprint: str | None = None) -> Licence:
    """Load and check a licence, or raise LicenceError.

    Fails closed on every path: missing file, bad signature, wrong machine,
    expired. There is deliberately no 'warn and continue'.
    """
    from cryptography.exceptions import InvalidSignature

    p = Path(path) if path else default_path()
    if not p.is_file():
        raise LicenceError(
            f"No licence found at {p}. Run scripts/show_fingerprint.py and send "
            f"the fingerprint to the vendor to obtain one."
        )
    try:
        blob = json.loads(p.read_text())
        payload_b64 = blob["payload"]
        sig = base64.b64decode(blob["signature"])
        payload_raw = base64.b64decode(payload_b64)
    except (ValueError, KeyError) as e:
        raise LicenceError(f"Licence file is malformed: {e}") from e

    try:
        _public_key().verify(sig, payload_raw)
    except InvalidSignature as e:
        raise LicenceError(
            "Licence signature is invalid. The file has been altered, or it "
            "was not issued for this product."
        ) from e

    data = json.loads(payload_raw)
    want = expect_fingerprint or fingerprint()
    if data.get("fingerprint") != want:
        raise LicenceError(
            "This licence was issued for different hardware.\n"
            f"  licence : {data.get('fingerprint')}\n"
            f"  this host: {want}"
        )
    exp = int(data.get("expires", 0))
    if exp and time.time() > exp:
        raise LicenceError(
            f"Licence expired on {time.strftime('%Y-%m-%d', time.gmtime(exp))}."
        )
    return Licence(
        customer=data.get("customer", ""),
        fingerprint=data["fingerprint"],
        issued=int(data.get("issued", 0)),
        expires=exp,
        features=tuple(data.get("features", ())),
        key_material=base64.b64decode(data.get("key_material", "")),
    )


_cached: Licence | None = None


def current(path: Path | None = None) -> Licence:
    """Verified licence for this process, checked once.

    Cached because `fingerprint()` shells out to nvidia-smi and the models ask
    for the key on every session build. The cache lives for the process only.
    """
    global _cached
    if _cached is None:
        _cached = verify(path)
        log.info("licence OK for %r%s", _cached.customer,
                 "" if not _cached.expires else
                 f", {_cached.days_left:.0f} days left")
    return _cached


def is_licensed(path: Path | None = None) -> bool:
    try:
        current(path)
        return True
    except LicenceError:
        return False
