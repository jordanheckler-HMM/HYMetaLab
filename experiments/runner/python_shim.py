#!/usr/bin/env python3
"""
Lightweight shim to provide the HTTP endpoints the TypeScript gateway expects.
Run this next to a sim instance (adjust PORT env var). It stores enqueued payloads as JSON files
in a local `shim_runs/` directory and can serve a fake healthz response describing the instance.

This is intentionally minimal and dependency-light (uses Python stdlib only).
"""
import datetime
import json
import os
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import urlparse

PORT = int(os.environ.get("SHIM_PORT", "5201"))
INSTANCE_ID = os.environ.get("SIM_INSTANCE_ID", "sim_local_5201")
DATA_DIR = os.environ.get("SIM_DATA_DIR", os.path.abspath("."))
SHIM_DIR = os.path.join(DATA_DIR, "shim_runs")

os.makedirs(SHIM_DIR, exist_ok=True)


class Handler(BaseHTTPRequestHandler):
    def _send(self, code, body, content_type="application/json"):
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body.encode("utf8"))))
        self.end_headers()
        self.wfile.write(body.encode("utf8"))

    def do_GET(self):
        p = urlparse(self.path)
        if p.path == "/healthz":
            body = json.dumps(
                {
                    "simInstanceId": INSTANCE_ID,
                    "metricsPort": None,
                    "dataDir": DATA_DIR,
                    "dbPath": os.path.join(DATA_DIR, "sim.db"),
                    "instanceFingerprint": INSTANCE_ID + "_fingerprint",
                    "rngSeed": 42,
                }
            )
            self._send(200, body)
            return
        if p.path.startswith("/run-status/"):
            run_id = p.path.split("/")[-1]
            pathf = os.path.join(SHIM_DIR, f"{run_id}.result.json")
            if os.path.exists(pathf):
                with open(pathf) as f:
                    self._send(200, f.read())
            else:
                self._send(404, json.dumps({"ok": False, "error": "not ready"}))
            return
        self._send(404, json.dumps({"error": "not found"}))

    def do_POST(self):
        p = urlparse(self.path)
        if p.path == "/enqueue-run":
            length = int(self.headers.get("Content-Length", "0"))
            body = self.rfile.read(length).decode("utf8")
            try:
                payload = json.loads(body)
                rid = (
                    payload.get("runId")
                    or f"run_{int(datetime.datetime.utcnow().timestamp())}"
                )
                outp = os.path.join(SHIM_DIR, f"{rid}.payload.json")
                with open(outp, "w") as f:
                    json.dump(payload, f, indent=2)
                # write a finished result immediately (demo mode) so the collector can pick it up
                outres = os.path.join(SHIM_DIR, f"{rid}.result.json")
                result = {
                    "runId": rid,
                    "simInstanceId": INSTANCE_ID,
                    "startedAt": datetime.datetime.utcnow().isoformat() + "Z",
                    "finishedAt": (
                        datetime.datetime.utcnow() + datetime.timedelta(seconds=1)
                    ).isoformat()
                    + "Z",
                    "ok": True,
                    "survival_rate": 0.5 + (hash(rid) % 50) / 100.0,
                    "collapse_risk": 0.5 - (hash(rid) % 30) / 100.0,
                    "aggression_delta": ((hash(rid) % 100) - 50) / 100.0,
                    "collective_cci_delta": ((hash(rid) % 100) / 100.0),
                    "shock_tolerance": 0.2 + ((hash(rid) % 80) / 100.0),
                }
                with open(outres, "w") as f:
                    json.dump(result, f, indent=2)
                self._send(202, json.dumps({"accepted": True, "runId": rid}))
            except Exception as e:
                self._send(400, json.dumps({"error": str(e)}))
            return
        self._send(404, json.dumps({"error": "not found"}))


if __name__ == "__main__":
    print("Starting shim on port", PORT)
    server = HTTPServer(("0.0.0.0", PORT), Handler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        server.server_close()
        print("Stopped.")
