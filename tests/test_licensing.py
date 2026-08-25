"""Machine binding and encrypted weights.

The threat these defend against is a customer copying the installation to a
second machine. They do not defend against a determined analyst with root -
see the module docstrings for why nothing that runs on someone else's hardware
can.

Note what is deliberately NOT asserted here: that the licence check cannot be
patched out. It can. What these tests pin down is that it cannot be *bypassed
with data* - no edited payload, no self-signed licence, no renamed model file.
"""
from __future__ import annotations

import base64
import json

import pytest

from app.core import licensing
from app.core.model_vault import (
    ENC_SUFFIX, VaultError, decrypt_bytes, encrypt_bytes, model_available,
)

KEY = b"\x01" * 32
OTHER_KEY = b"\x02" * 32


@pytest.fixture(autouse=True)
def _clear_licence_cache():
    licensing._cached = None
    yield
    licensing._cached = None


# ---- encryption --------------------------------------------------------

def test_roundtrip_returns_the_exact_bytes():
    plain = b"\x08\x01\x12\x00onnx-ish payload" * 500
    blob = encrypt_bytes(plain, KEY, "model.onnx")
    assert blob != plain
    assert decrypt_bytes(blob, KEY, "model.onnx") == plain


def test_ciphertext_does_not_contain_the_plaintext():
    plain = b"SECRET-WEIGHTS-" + b"\x7f" * 4096
    blob = encrypt_bytes(plain, KEY, "m.onnx")
    assert b"SECRET-WEIGHTS-" not in blob


def test_wrong_key_cannot_decrypt():
    blob = encrypt_bytes(b"weights", KEY, "m.onnx")
    with pytest.raises(VaultError):
        decrypt_bytes(blob, OTHER_KEY, "m.onnx")


def test_renaming_a_model_file_breaks_decryption():
    """The filename is authenticated, so swapping two encrypted models cannot
    quietly load the wrong weights under the right name."""
    blob = encrypt_bytes(b"head weights", KEY, "head.onnx")
    with pytest.raises(VaultError):
        decrypt_bytes(blob, KEY, "recognizer.onnx")


def test_flipping_one_ciphertext_byte_is_detected():
    """AES-GCM authenticates, so corruption fails loudly rather than feeding
    ONNX Runtime a subtly damaged graph."""
    blob = bytearray(encrypt_bytes(b"x" * 4096, KEY, "m.onnx"))
    blob[-1] ^= 0x01
    with pytest.raises(VaultError):
        decrypt_bytes(bytes(blob), KEY, "m.onnx")


def test_plain_file_is_rejected_as_ciphertext():
    with pytest.raises(VaultError):
        decrypt_bytes(b"just an onnx file", KEY, "m.onnx")


def test_two_encryptions_of_the_same_bytes_differ():
    """Random nonce per encryption: identical models must not produce identical
    ciphertext, or a customer could tell two releases were the same weights."""
    a = encrypt_bytes(b"same", KEY, "m.onnx")
    b = encrypt_bytes(b"same", KEY, "m.onnx")
    assert a != b
    assert decrypt_bytes(a, KEY, "m.onnx") == decrypt_bytes(b, KEY, "m.onnx")


def test_no_key_material_is_refused():
    with pytest.raises(VaultError):
        encrypt_bytes(b"x", b"", "m.onnx")


# ---- model_available ---------------------------------------------------

def test_model_available_sees_the_encrypted_variant(tmp_path):
    """The regression this exists for: a plain Path.exists() check returned
    False in an encrypted deployment, which silently disabled head tracking
    and dropped the pipeline to face detection."""
    p = tmp_path / "head.onnx"
    assert model_available(p) is False
    (tmp_path / ("head.onnx" + ENC_SUFFIX)).write_bytes(b"enc")
    assert model_available(p) is True


def test_model_available_sees_a_plain_file(tmp_path):
    p = tmp_path / "head.onnx"
    p.write_bytes(b"plain")
    assert model_available(p) is True


# ---- licence -----------------------------------------------------------

def _issue(fingerprint: str, *, expires: int = 0, signer=None) -> dict:
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
    signer = signer or Ed25519PrivateKey.generate()
    payload = {"customer": "T", "fingerprint": fingerprint, "issued": 0,
               "expires": expires, "features": [],
               "key_material": base64.b64encode(KEY).decode()}
    raw = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode()
    return {"payload": base64.b64encode(raw).decode(),
            "signature": base64.b64encode(signer.sign(raw)).decode()}, signer


def _install(tmp_path, blob) -> "Path":
    p = tmp_path / "licence.key"
    p.write_text(json.dumps(blob))
    return p


def _use_key(monkeypatch, signer):
    from cryptography.hazmat.primitives import serialization
    pub = signer.public_key().public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw)
    monkeypatch.setenv("EMATSY_LICENCE_PUBKEY", base64.b64encode(pub).decode())


def test_missing_licence_is_refused(tmp_path):
    with pytest.raises(licensing.LicenceError):
        licensing.verify(tmp_path / "nope.key", expect_fingerprint="abc")


def test_valid_licence_verifies(tmp_path, monkeypatch):
    blob, signer = _issue("machine-a")
    _use_key(monkeypatch, signer)
    lic = licensing.verify(_install(tmp_path, blob), expect_fingerprint="machine-a")
    assert lic.fingerprint == "machine-a"
    assert lic.key_material == KEY


def test_licence_for_another_machine_is_refused(tmp_path, monkeypatch):
    """The copied-installation case."""
    blob, signer = _issue("machine-a")
    _use_key(monkeypatch, signer)
    with pytest.raises(licensing.LicenceError, match="different hardware"):
        licensing.verify(_install(tmp_path, blob), expect_fingerprint="machine-b")


def test_edited_payload_is_refused(tmp_path, monkeypatch):
    """Rewriting the fingerprint while keeping the signature must not work."""
    blob, signer = _issue("machine-a")
    _use_key(monkeypatch, signer)
    data = json.loads(base64.b64decode(blob["payload"]))
    data["fingerprint"] = "machine-b"
    blob["payload"] = base64.b64encode(
        json.dumps(data, separators=(",", ":"), sort_keys=True).encode()).decode()
    with pytest.raises(licensing.LicenceError, match="signature is invalid"):
        licensing.verify(_install(tmp_path, blob), expect_fingerprint="machine-b")


def test_licence_signed_with_another_key_is_refused(tmp_path, monkeypatch):
    """Someone minting their own licence with their own key pair."""
    _, vendor = _issue("x")
    attacker_blob, _ = _issue("machine-b")        # signed by a fresh key
    _use_key(monkeypatch, vendor)                 # but we trust the vendor's
    with pytest.raises(licensing.LicenceError, match="signature is invalid"):
        licensing.verify(_install(tmp_path, attacker_blob),
                         expect_fingerprint="machine-b")


def test_expired_licence_is_refused(tmp_path, monkeypatch):
    blob, signer = _issue("machine-a", expires=1)   # 1970
    _use_key(monkeypatch, signer)
    with pytest.raises(licensing.LicenceError, match="expired"):
        licensing.verify(_install(tmp_path, blob), expect_fingerprint="machine-a")


def test_perpetual_licence_does_not_expire(tmp_path, monkeypatch):
    blob, signer = _issue("machine-a", expires=0)
    _use_key(monkeypatch, signer)
    lic = licensing.verify(_install(tmp_path, blob), expect_fingerprint="machine-a")
    assert lic.days_left is None


def test_malformed_licence_is_refused(tmp_path, monkeypatch):
    p = tmp_path / "licence.key"
    p.write_text("not json at all")
    with pytest.raises(licensing.LicenceError, match="malformed"):
        licensing.verify(p, expect_fingerprint="machine-a")


# ---- fingerprint -------------------------------------------------------

def test_fingerprint_is_stable_and_hides_the_serials(monkeypatch):
    monkeypatch.setattr(licensing, "gpu_uuid", lambda: "GPU-abc-123")
    monkeypatch.setattr(licensing, "machine_id", lambda: "mid-xyz")
    a = licensing.fingerprint()
    b = licensing.fingerprint()
    assert a == b
    assert len(a) == 32
    assert "GPU-abc-123" not in a and "mid-xyz" not in a


def test_fingerprint_changes_with_the_gpu(monkeypatch):
    monkeypatch.setattr(licensing, "machine_id", lambda: "mid")
    monkeypatch.setattr(licensing, "gpu_uuid", lambda: "GPU-one")
    a = licensing.fingerprint()
    monkeypatch.setattr(licensing, "gpu_uuid", lambda: "GPU-two")
    assert licensing.fingerprint() != a


def test_missing_gpu_refuses_rather_than_guessing(monkeypatch):
    """Falling back to a weaker identifier would let the licence follow a disk
    image onto other hardware."""
    monkeypatch.setattr(licensing, "gpu_uuid", lambda: None)
    with pytest.raises(licensing.LicenceError, match="GPU UUID"):
        licensing.fingerprint()
