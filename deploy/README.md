# Deployment

## Restarting an existing install

```bash
./scripts/restart.sh            # stop, migrate, pre-flight, start, verify
./scripts/restart.sh --status   # what is running, changes nothing
./scripts/restart.sh --stop
```

Prefer it over stopping and starting by hand. It finds the process by its
working directory rather than by a command pattern (`pkill -f scripts/run.py`
has killed an SSH session outright), proves the stop before it starts anything
(a worker blocked on an RTSP read ignores SIGTERM, and two processes on the same
cameras and database look healthy), clears the `__pycache__` that otherwise
shadows a freshly deployed `.so`, and checks the CUDA provider and the
gallery/recognizer pairing before the cameras go down rather than after.

## First install, under systemd

```bash
sudo mkdir -p /var/log/ematsy && sudo chown inomjon /var/log/ematsy
sudo cp deploy/ematsy*.service deploy/ematsy*.timer /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now ematsy.service ematsy-maintenance.timer
systemctl status ematsy
journalctl -u ematsy -f
```

Adjust `User=`, `WorkingDirectory=` and the Python path in the unit files if the
project or conda environment lives elsewhere.

## Moving to another machine

1. Copy the project directory (includes `models/`, 225 MB of real weights).
2. `cd models && sha256sum -c MANIFEST.sha256`
3. Recreate the Python environment — see `docs/OPERATIONS.md`. **Install
   `onnxruntime-gpu` only; never alongside `onnxruntime`.**
4. `python scripts/enroll.py && python scripts/seed_cameras.py`
5. Edit camera IPs/credentials in `scripts/seed_cameras.py` or the `camera` table.
6. Re-run the calibration walkthrough — thresholds are site-specific.
