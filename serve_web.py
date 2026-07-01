#!/usr/bin/env python3
# Local static server for build-web with COOP/COEP headers, so performance.now() (and thus
# RenderGraph's compile/realize/execute CPU timings) gets full precision instead of the
# browser's default clamp. Cross-origin isolation is the unlock; only matters locally --
# GitHub Pages doesn't let us set these headers for the deployed build.
import sys
from http.server import HTTPServer, SimpleHTTPRequestHandler
from functools import partial

class COOPCOEPHandler(SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header("Cross-Origin-Opener-Policy", "same-origin")
        self.send_header("Cross-Origin-Embedder-Policy", "require-corp")
        super().end_headers()

if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8080
    directory = sys.argv[2] if len(sys.argv) > 2 else "."
    handler = partial(COOPCOEPHandler, directory=directory)
    HTTPServer(("", port), handler).serve_forever()
