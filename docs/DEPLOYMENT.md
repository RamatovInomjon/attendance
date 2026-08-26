# Deploying to another machine

Two halves: what the **vendor** does once per customer, and what the
**customer** does on site. The customer never receives the algorithm sources,
the plaintext weights, or the signing key.

---

## Part 1 — Vendor, once ever

```bash
python scripts/make_license.py --init
```

Generates the Ed25519 vendor key pair, writes `vendor_private_key.pem`
(mode 600) and embeds the public half in `app/core/licensing.py`.

> **This private key is the entire security boundary.** Anyone holding it can
> mint a licence for any machine. It is in `.gitignore`; keep it off customer
> machines, off shared drives, and backed up somewhere you control — losing it
> means no existing installation can ever be re-licensed.

Rebuild the core so the new public key is compiled in:

```bash
python scripts/build_native.py --verify
```

---

## Part 2 — Customer sends their fingerprint

On the target machine:

```bash
python scripts/show_fingerprint.py
```

```json
{
  "fingerprint": "b385055d2cac1c0de16977a6a02e9fe5",
  "gpu_uuid": "GPU-bd3851ec-28ec-4108-b36e-f9f288347bd5",
  "machine_id_present": true,
  "hostname": "legion"
}
```

The fingerprint is a salted SHA-256 of the GPU UUID and `/etc/machine-id`, so
it identifies the host without publishing its serial numbers.

**It is bound to the GPU, not the MAC address.** MAC was the obvious choice and
it is wrong: this host has two NICs, `ip link set dev eth0 address …` spoofs one
in a single command, and docks and VPNs add and remove interfaces — which turns
into support calls where the software stopped working for no visible reason. The
GPU UUID is stable, readable without root, and already a hard requirement.

---

## Part 3 — Vendor builds a bundle for that machine

```bash
python scripts/make_license.py --issue \
    --fingerprint b385055d2cac1c0de16977a6a02e9fe5 \
    --customer "AIRI HQ" --days 365 --out licence.key

python scripts/encrypt_models.py --licence licence.key
python scripts/encrypt_models.py --licence licence.key --verify   # round-trip

python scripts/build_native.py
python scripts/package_release.py --out dist --licence licence.key --tar
```

Result: `dist/ematsy-release.tar.gz`, ~147 MB.

| in the bundle | withheld |
|---|---|
| `app/core/*.so` — compiled, stripped | `app/core/*.py` — the algorithm in readable form |
| `models/*.onnx.enc` — AES-256-GCM | `models/*.onnx` — plaintext weights |
| `licence.key` for that one machine | `vendor_private_key.pem` |
| web layer, templates, operator scripts | `face_id_users/`, `data/`, `.env`, tests, bench |

`package_release.py` refuses to produce a bundle if a plaintext `.onnx`, a core
`.py`, or the private key would end up inside it.

The bundle is **per customer**: model encryption is keyed to their licence, so a
bundle built for one site will not run at another even with a valid licence of
its own.

---

## Part 4 — Customer installs

Requirements on the target machine:

| | |
|---|---|
| GPU | NVIDIA with CUDA 12 + cuDNN 9. The pipeline refuses to start on CPU. |
| Python | 3.10 — the `.so` files are built for one CPython ABI (`cp310`) |
| OS | Linux x86-64 |
| RAM | 8 GB minimum; 16 GB if running two 4K cameras |
| Disk | 20 GB plus recording space (~3 GB per camera-day at 1080p) |

```bash
tar xzf ematsy-release.tar.gz && cd ematsy
pip install -r requirements.txt

# CPU-only onnxruntime silently shadows the GPU build and costs ~10x.
pip uninstall -y onnxruntime onnxruntime-gpu
pip install --no-deps onnxruntime-gpu==1.23.2
pip list | grep -E "^onnxruntime"        # must show onnxruntime-gpu ONLY

python scripts/seed_admin.py             # signing key + default admin
python scripts/seed_cameras.py           # camera URLs and direction geometry
python scripts/enroll.py                 # build the gallery from site photos
./run.sh
```

Console on `http://<host>:8000`, sign in with `inomjon` / `123456` and **change
that password immediately** at `/users`.

### Verify the install

```bash
python scripts/show_fingerprint.py       # should report the licence as VALID
python scripts/benchmark.py --out install-check.json
```

Compare against the figures the bundle shipped with. On the reference machine
(RTX 3070 Laptop): d′ ≈ 10.03, rank-1 100%, ~8.2 ms median per frame.

---

## Debug capture is off in a release

Three switches exist for tuning and are **off by default**, because a customer
site is not a debugging session:

| setting | default | what it does |
|---|---|---|
| `record_clips` | `False` | records a video clip per person-pass |
| `save_all_frames` | `False` | writes every gate-passing frame |
| `debug_max_per_person` | `40` | caps per-person evidence images (`0` = unbounded) |

One day of two cameras produced **1,156 clips / 3.2 GB** of video and 15,724
images — all of it footage of identified people. On a customer site that is a
privacy exposure as much as a disk problem, so `package_release.py` **refuses
to build a bundle** while any of them is enabled.

What a release *does* keep is `debug_capture`: one best-shot image and a JSON
sidecar per recognition, capped, which is the evidence the console shows when
somebody disputes an attendance row.

To collect a replay corpus on a live site, enable it for that session only:

```bash
record_clips=1 save_all_frames=1 debug_max_per_person=0 ./run.sh
```

Then turn it off. `scripts/maintenance.py` prunes what it leaves behind.

---

## Re-licensing an existing install

When a licence expires or hardware changes, reuse the **same key material** so
the already-encrypted models keep working:

```bash
python scripts/make_license.py --inspect old-licence.key      # read key_material
python scripts/make_license.py --issue --fingerprint <new> \
    --key-material <base64 from above> --out licence.key
```

Issuing with fresh key material instead means the customer needs a new bundle.

---

## Troubleshooting

| symptom | cause |
|---|---|
| `No licence found` | `licence.key` missing from the install root, or set `EMATSY_LICENCE` |
| `issued for different hardware` | GPU replaced, or the bundle went to the wrong site |
| `Cannot read the GPU UUID` | `nvidia-smi` unavailable — driver not loaded, or running in a container without `--gpus all` |
| `decryption failed` | licence and model files are from different builds |
| `undefined symbol` / `ImportError` on a `.so` | wrong Python version — the core is built for cp310 |
| Everything runs but slowly | CPU-only `onnxruntime` shadowing the GPU build; re-run the uninstall above |

---

## What this protects, and what it does not

**It stops** a customer copying the installation to a second machine. The
licence is signed by a key they do not have, and the weights are ciphertext
without it. That is the realistic threat, and it is handled.

**It does not stop** a determined analyst with root on the machine. The licence
check runs on their CPU and can be patched out; the decrypted weights exist in
process memory and can be dumped from `/proc/<pid>/mem`. Nothing that executes
on hardware someone else controls can prevent this — compiled or not.

What the layers buy is cost. Copying a directory is minutes. Patching a stripped
`.so` *and* recovering 130 MB of weights from a live process is a different kind
of effort, and one that a customer casually reusing your software will not make.

If the weights genuinely must not leak, the only real answer is to stop shipping
them: run inference on a server you control and ship a thin client. That is a
product decision, not an engineering one.
