"""Stdlib HTTP API for cluster schedulers and CI agents.

Endpoints
---------
GET  /healthz            → 200 {"ok": true}
GET  /hardware           → detected hardware profile (this machine)
GET  /models             → full catalog
GET  /models/{id}        → one model
POST /rank               → rank for an arbitrary hardware spec
                           body: {"hardware": {...}, "use_case": "...", "top": N,
                                  "min_context": N, "include_unfit": bool,
                                  "family": "...", "type": "...", "min_quality": N}
                           (omit "hardware" to use this machine's profile)
POST /reverse            → body: {"model": "id", "target_tps": N, "context": N, "quant": "..."}

All responses are JSON. CORS is enabled with * (the API is intended to run
on an internal network or localhost).
"""
from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse

from modelfit import __version__
from modelfit.hardware import detect_hardware, hardware_from_dict
from modelfit.models import get_catalog, find_model
from modelfit.scoring import rank_all, USE_CASE_WEIGHTS
from modelfit.reverse import recommend_hardware


def _json(handler: BaseHTTPRequestHandler, status: int, payload):
    body = json.dumps(payload, default=str).encode()
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json")
    handler.send_header("Content-Length", str(len(body)))
    handler.send_header("Access-Control-Allow-Origin", "*")
    handler.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
    handler.send_header("Access-Control-Allow-Headers", "Content-Type")
    handler.end_headers()
    handler.wfile.write(body)


def _read_json_body(handler: BaseHTTPRequestHandler) -> dict:
    length = int(handler.headers.get("Content-Length", 0))
    if not length:
        return {}
    raw = handler.rfile.read(length)
    if not raw:
        return {}
    try:
        return json.loads(raw)
    except json.JSONDecodeError as e:
        raise ValueError(f"invalid JSON body: {e}")


class Handler(BaseHTTPRequestHandler):
    server_version = f"modelfit/{__version__}"

    def log_message(self, fmt, *args):
        # Quieter default; uvicorn-style one-line log.
        print(f"[modelfit-api] {self.address_string()} - {fmt % args}")

    def do_OPTIONS(self):
        _json(self, 204, {})

    def do_GET(self):
        path = urlparse(self.path).path.rstrip("/") or "/"
        try:
            if path == "/" or path == "/healthz":
                return _json(self, 200, {"ok": True, "service": "modelfit",
                                          "version": __version__,
                                          "model_count": len(get_catalog())})
            if path == "/hardware":
                return _json(self, 200, detect_hardware().to_dict())
            if path == "/models":
                return _json(self, 200, [m.to_dict() for m in get_catalog()])
            if path.startswith("/models/"):
                mid = path[len("/models/"):]
                m = find_model(mid)
                if not m:
                    return _json(self, 404, {"error": "model not found", "id": mid})
                return _json(self, 200, m.to_dict())
            if path == "/use-cases":
                return _json(self, 200, {
                    "use_cases": {
                        k: dict(zip(("quality", "speed", "context", "capability"), v))
                        for k, v in USE_CASE_WEIGHTS.items()
                    }
                })
            return _json(self, 404, {"error": "not found", "path": path})
        except Exception as e:
            return _json(self, 500, {"error": str(e)})

    def do_POST(self):
        path = urlparse(self.path).path.rstrip("/") or "/"
        try:
            body = _read_json_body(self)
        except ValueError as e:
            return _json(self, 400, {"error": str(e)})

        if path == "/rank":
            hw_spec = body.get("hardware")
            hw = hardware_from_dict(hw_spec) if hw_spec else detect_hardware()
            use_case = body.get("use_case", "balanced")
            if use_case not in USE_CASE_WEIGHTS:
                return _json(self, 400, {"error": f"unknown use_case {use_case}",
                                          "valid": list(USE_CASE_WEIGHTS.keys())})
            top = int(body.get("top", 20))
            ranked = rank_all(
                hw,
                use_case=use_case,
                min_context=int(body.get("min_context", 2048)),
                include_unfit=bool(body.get("include_unfit", False)),
                family=body.get("family"),
                type_filter=body.get("type"),
                min_quality=float(body.get("min_quality", 0)),
            )
            return _json(self, 200, {
                "hardware": hw.to_dict(),
                "use_case": use_case,
                "weights": dict(zip(("quality", "speed", "context", "capability"),
                                     USE_CASE_WEIGHTS[use_case])),
                "fitting_models": len(ranked),
                "models": [s.to_dict() for s in ranked[:top]],
            })

        if path == "/reverse":
            model_id = body.get("model")
            if not model_id:
                return _json(self, 400, {"error": "missing 'model'"})
            m = find_model(model_id)
            if not m:
                return _json(self, 404, {"error": "model not found", "id": model_id})
            try:
                rec = recommend_hardware(
                    m,
                    target_tps=float(body.get("target_tps", 30)),
                    context=body.get("context"),
                    quant=body.get("quant", "Q4_K_M"),
                )
            except ValueError as e:
                return _json(self, 400, {"error": str(e)})
            return _json(self, 200, rec)

        return _json(self, 404, {"error": "not found", "path": path})


def serve(host: str = "127.0.0.1", port: int = 8765):
    srv = ThreadingHTTPServer((host, port), Handler)
    print(f"[modelfit] REST API listening on http://{host}:{port}")
    print(f"[modelfit] try:   curl http://{host}:{port}/hardware")
    print(f"[modelfit] also:  curl -X POST http://{host}:{port}/rank "
          f"-H 'content-type: application/json' -d '{{\"use_case\":\"chat\",\"top\":5}}'")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\n[modelfit] shutting down")
    finally:
        srv.server_close()
