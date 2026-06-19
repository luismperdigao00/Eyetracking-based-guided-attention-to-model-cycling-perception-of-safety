#!/usr/bin/env python3
"""Application entrypoint for the perceived-safety inspector."""

from __future__ import annotations

from perceived_safety_app.routes import main


if __name__ == "__main__":
    raise SystemExit(main())
