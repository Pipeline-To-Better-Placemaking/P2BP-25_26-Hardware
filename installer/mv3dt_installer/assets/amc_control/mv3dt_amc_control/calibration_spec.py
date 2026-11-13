"""The project data a calibration run starts from, signed by the web app.

The workstation cannot reach the web app's database. The Express API reads the
project's record, signs it as a short-lived token with the shared
WORKSTATION_TICKET_SECRET, and the browser hands that token to this service.
Signing stops a user from swapping in another project's data or pointing the
workstation at an arbitrary URL; the floor plan is still only downloaded from
the hosts listed in the config.
"""
from __future__ import annotations

import io
from dataclasses import dataclass
from urllib.parse import urlparse

import httpx
import jwt

from .auth import AUDIENCE

SPEC_SCOPE = "amc-calibration-spec"
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


class SpecError(ValueError):
    pass


@dataclass(frozen=True)
class Floorplan:
    id: str
    url: str
    content_type: str
    width: int
    height: int
    px_per_m: float | None


@dataclass(frozen=True)
class CalibrationSpec:
    project_id: str
    title: str
    floorplan: Floorplan | None
    issued_to: str


def verify_spec(token: str, secret: str, ticket_pid: str, issuer: str | None = None) -> CalibrationSpec:
    kwargs = {"issuer": issuer} if issuer else {}
    try:
        claims = jwt.decode(token, secret, algorithms=["HS256"], audience=AUDIENCE,
                            options={"require": ["exp", "sub"]}, **kwargs)
    except jwt.PyJWTError as e:
        raise SpecError(f"invalid calibration spec: {e}") from e
    if claims.get("scope") != SPEC_SCOPE:
        raise SpecError("token is not a calibration spec")
    if claims.get("pid") != ticket_pid:
        raise SpecError("calibration spec is for a different project than the ticket")
    spec = claims.get("spec")
    if not isinstance(spec, dict):
        raise SpecError("calibration spec has no project data")

    fp = spec.get("floorplan")
    floorplan = None
    if isinstance(fp, dict) and fp.get("url"):
        scale = fp.get("pxPerM")
        floorplan = Floorplan(
            id=str(fp.get("id", "")),
            url=str(fp["url"]),
            content_type=str(fp.get("contentType") or ""),
            width=int(fp.get("width") or 0),
            height=int(fp.get("height") or 0),
            px_per_m=float(scale) if isinstance(scale, (int, float)) and scale > 0 else None,
        )
    title = str(spec.get("title") or "").strip() or claims["pid"]
    return CalibrationSpec(project_id=claims["pid"], title=title, floorplan=floorplan, issued_to=claims["sub"])


def download_floorplan_png(fp: Floorplan, allowed_hosts: list[str], max_bytes: int,
                           client: httpx.Client | None = None) -> bytes:
    """Fetch the floor plan and return it as PNG bytes (AMC's layout must be a PNG)."""
    parsed = urlparse(fp.url)
    host = (parsed.hostname or "").lower()
    allowed = {h.strip().lower() for h in allowed_hosts if h.strip()}
    if parsed.scheme != "https" or host not in allowed:
        raise SpecError(f"floor plan host {host or '?'} is not in layout.allowed_hosts")

    http = client or httpx.Client()
    try:
        with http.stream("GET", fp.url, timeout=60, follow_redirects=False) as r:
            if r.status_code != 200:
                raise SpecError(f"floor plan download failed: HTTP {r.status_code}")
            chunks, size = [], 0
            for chunk in r.iter_bytes():
                size += len(chunk)
                if size > max_bytes:
                    raise SpecError(f"floor plan is larger than {max_bytes} bytes")
                chunks.append(chunk)
    except httpx.HTTPError as e:
        raise SpecError(f"floor plan download failed: {e}") from e
    finally:
        if client is None:
            http.close()
    data = b"".join(chunks)
    return data if data.startswith(PNG_SIGNATURE) else _to_png(data)


def _to_png(data: bytes) -> bytes:
    try:
        from PIL import Image
    except ImportError as e:  # Pillow is in requirements.txt; this guards odd installs
        raise SpecError("floor plan is not a PNG and Pillow is not installed to convert it") from e
    try:
        with Image.open(io.BytesIO(data)) as img:
            out = io.BytesIO()
            img.convert("RGBA" if img.mode in ("RGBA", "LA", "P") else "RGB").save(out, format="PNG")
            return out.getvalue()
    except Exception as e:  # PIL raises several exception types for bad images
        raise SpecError(f"floor plan could not be converted to PNG: {e}") from e
