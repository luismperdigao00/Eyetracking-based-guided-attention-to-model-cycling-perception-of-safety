"""HTTP server bootstrap for the perceived-safety app."""

from __future__ import annotations

import argparse
import shutil
from http.server import ThreadingHTTPServer
from typing import Iterable, Optional

from perceived_safety_app.request_handlers import TEMP_OUTPUT_ROOT, SafetyAppHandler, clear_model_cache


class ReusableThreadingHTTPServer(ThreadingHTTPServer):
    allow_reuse_address = True


def main(argv: Optional[Iterable[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Run the perceived-safety local deployment app.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args(list(argv) if argv is not None else None)

    shutil.rmtree(TEMP_OUTPUT_ROOT, ignore_errors=True)
    TEMP_OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    server = ReusableThreadingHTTPServer((args.host, args.port), SafetyAppHandler)
    print(f"Perceived Safety Model Inspector running at http://{args.host}:{args.port}")
    print(f"Temporary outputs: {TEMP_OUTPUT_ROOT}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping server.")
    finally:
        server.server_close()
        clear_model_cache()
        shutil.rmtree(TEMP_OUTPUT_ROOT, ignore_errors=True)
    return 0
