"""Load model weights, encrypted or plain, without writing plaintext to disk.

The weights are the asset worth protecting. The pipeline is ~1,900 lines that a
competent engineer could rewrite; `adaface_ir101_finetune_fp16.onnx` is a
fine-tune that exists nowhere else, and `yolov8n_head_960x544_fp16.onnx` is 100
epochs on full CrowdHuman. Copying the folder is the realistic theft, and a
compiled .so does nothing to prevent it.

So encrypted models are AES-256-GCM, and the key is unwrapped from the signed
licence - which is bound to this machine's GPU. Copy the folder elsewhere and
you have ciphertext and a licence that will not verify.

`ort.InferenceSession` accepts a serialized model as **bytes**, so the plaintext
never touches the filesystem: it exists in this process's memory and nowhere
else.

## Be honest about the limit

Plaintext weights must exist in RAM for ONNX Runtime to build a session. A
determined attacker with root on the machine can read them out of
`/proc/<pid>/mem`, or LD_PRELOAD a shim over session creation. This raises the
cost from "copy a directory" to "understand the process and dump its memory".
If that is not enough, the only real answer is to stop shipping the weights and
run inference on a server you control.

## Development

Unencrypted `.onnx` files load exactly as before, with no licence required, so
the development workflow is unchanged. Encryption is something a release build
opts into by shipping `.onnx.enc` files instead.
"""
from __future__ import annotations

import hashlib
import logging
import os
from pathlib import Path

log = logging.getLogger(__name__)

ENC_SUFFIX = ".enc"
MAGIC = b"EMATSY1\x00"        # 8 bytes, so a wrong file fails fast and clearly
NONCE_LEN = 12                # AES-GCM standard
_KEY_INFO = b"ematsy-model-key-v1"


class VaultError(RuntimeError):
    pass


def _derive_key(key_material: bytes, model_name: str) -> bytes:
    """One key per model, derived from the licence's key material.

    Per-model rather than one global key so that recovering one model's key -
    by whatever means - does not hand over the rest.
    """
    if not key_material:
        raise VaultError("Licence carries no key material; cannot decrypt models.")
    return hashlib.pbkdf2_hmac(
        "sha256", key_material, _KEY_INFO + model_name.encode("utf-8"), 200_000, dklen=32
    )


def encrypt_bytes(plain: bytes, key_material: bytes, model_name: str) -> bytes:
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM

    key = _derive_key(key_material, model_name)
    nonce = os.urandom(NONCE_LEN)
    # model_name is authenticated, so a renamed file fails to decrypt rather
    # than silently loading the wrong weights.
    ct = AESGCM(key).encrypt(nonce, plain, model_name.encode("utf-8"))
    return MAGIC + nonce + ct


def decrypt_bytes(blob: bytes, key_material: bytes, model_name: str) -> bytes:
    from cryptography.exceptions import InvalidTag
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM

    if not blob.startswith(MAGIC):
        raise VaultError(f"{model_name}: not an encrypted model file")
    nonce = blob[len(MAGIC):len(MAGIC) + NONCE_LEN]
    ct = blob[len(MAGIC) + NONCE_LEN:]
    key = _derive_key(key_material, model_name)
    try:
        return AESGCM(key).decrypt(nonce, ct, model_name.encode("utf-8"))
    except InvalidTag as e:
        raise VaultError(
            f"{model_name}: decryption failed. The licence does not match these "
            f"model files, or the file is corrupt."
        ) from e


def load_model(path: str | Path) -> str | bytes:
    """Return what `ort.InferenceSession` should be given for this model.

    A plain `.onnx` returns its path - ONNX Runtime memory-maps it, which is
    cheaper than handing over 130 MB of bytes. An encrypted model returns
    decrypted bytes, so the plaintext is never on disk.

    Looks for `<name>.enc` first, so a release directory containing only
    encrypted files works without any config change.
    """
    p = Path(path)
    enc = p.with_name(p.name + ENC_SUFFIX)

    if enc.is_file():
        from app.core import licensing
        lic = licensing.current()          # raises LicenceError if not valid
        blob = enc.read_bytes()
        plain = decrypt_bytes(blob, lic.key_material, p.name)
        log.info("loaded %s from vault (%.1f MB, decrypted in memory)",
                 p.name, len(plain) / 1e6)
        return plain

    if p.is_file():
        return str(p)

    raise VaultError(
        f"Model not found: neither {p.name} nor {p.name}{ENC_SUFFIX} exists in "
        f"{p.parent}"
    )


def is_encrypted_deployment(models_dir: Path) -> bool:
    return any(models_dir.glob("*" + ENC_SUFFIX))


def model_available(path: str | Path) -> bool:
    """True when this model can be loaded, plain OR encrypted.

    Callers must not use `Path(p).exists()` to decide whether a model is
    present: in an encrypted deployment only `<name>.enc` is on disk, so the
    plain check returns False and the caller silently falls back to a different
    code path. That is exactly what happened to the head detector - the
    pipeline dropped to face detection, which tracked worse and lost a
    recognition, with nothing in the logs to say why.
    """
    p = Path(path)
    return p.is_file() or p.with_name(p.name + ENC_SUFFIX).is_file()
