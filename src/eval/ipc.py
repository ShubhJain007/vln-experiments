"""
Length-prefixed message framing shared by the eval client and habitat worker.

Pure stdlib on purpose: this module is imported by BOTH Python environments --
`latentpilot` (3.10, owns the model) and `habitat_render` (3.9, owns the
simulator) -- so it must not depend on anything either env lacks.

WHY A SOCKETPAIR AND NOT STDOUT: habitat-sim writes a large volume of log lines
to stdout/stderr on startup and on every scene load. Any line- or byte-oriented
protocol sharing stdout would be corrupted by that noise. A dedicated socket
pair sidesteps it entirely and lets the worker's stdout stay as chatty as it
likes.

Message layout, so a frame and its metadata arrive atomically:

    [4 bytes  total length      ]
    [4 bytes  json length J     ]
    [J bytes  utf-8 json header ]
    [rest     optional raw bytes]  <- e.g. an RGB frame, uncompressed
"""

import json
import struct

_HEADER = struct.Struct("!I")


def send_msg(sock, obj, raw: bytes = b"") -> None:
    """Send one JSON header plus an optional raw binary payload."""
    payload = json.dumps(obj).encode("utf-8")
    body = _HEADER.pack(len(payload)) + payload + raw
    sock.sendall(_HEADER.pack(len(body)) + body)


def _recv_exactly(sock, n: int) -> bytes:
    """Read exactly n bytes. recv() may return short reads on a stream socket,
    so looping here is required, not defensive."""
    chunks, got = [], 0
    while got < n:
        chunk = sock.recv(min(n - got, 1 << 20))
        if not chunk:
            raise ConnectionError(
                f"peer closed after {got}/{n} bytes -- the other process "
                f"most likely crashed; check its stderr"
            )
        chunks.append(chunk)
        got += len(chunk)
    return b"".join(chunks)


def recv_msg(sock):
    """Receive one message. Returns (json_obj, raw_bytes)."""
    total = _HEADER.unpack(_recv_exactly(sock, 4))[0]
    body = _recv_exactly(sock, total)
    jlen = _HEADER.unpack(body[:4])[0]
    obj = json.loads(body[4:4 + jlen].decode("utf-8"))
    return obj, body[4 + jlen:]
