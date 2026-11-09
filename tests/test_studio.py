"""New loopback UI security, file routing, media bounds and concurrency guards."""

import io
import json
import threading
import urllib.error
import urllib.request

from PIL import Image
import pytest

from src.demo_media import DemoError
from src.studio_server import ASSETS, create_server, detect_media


def photo(format="PNG"):
    buffer = io.BytesIO()
    Image.new("RGB", (12, 8), (30, 80, 120)).save(buffer, format=format)
    return buffer.getvalue()


@pytest.fixture
def server(tmp_path):
    instance = create_server(tmp_path, "a" * 48)
    thread = threading.Thread(target=instance.serve_forever, daemon=True)
    thread.start()
    yield instance
    instance.shutdown()
    instance.server_close()
    thread.join(timeout=2)


def query(server, route, body=None, origin=True, token=True, headers=None):
    url = f"http://127.0.0.1:{server.server_port}"
    meta = {}
    if body is not None:
        if origin:
            meta["Origin"] = url
        if token:
            meta["X-Studio-Token"] = server.token
    meta.update(headers or {})
    request = urllib.request.Request(
        url + "/" + server.token + "/" + route, data=body, headers=meta
    )
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    return opener.open(request, timeout=3)


@pytest.mark.parametrize(
    "name,raw",
    [
        ("image.jpg", photo()),
        ("image.png", b"not image"),
        ("file.exe", photo()),
        ("clip.mp4", b"bad video"),
    ],
)
def test_extension_and_actual_file_content_must_match(name, raw):
    with pytest.raises(DemoError):
        detect_media(raw, name)


def test_correct_jpeg_and_png_automatically_route_to_photo():
    assert detect_media(photo(), "p.png") == "photo"
    assert detect_media(photo("JPEG"), "p.jpg") == "photo"


def test_loopback_static_assets_and_no_path_traversal(server):
    assert server.server_address[0] == "127.0.0.1"
    with query(server, "") as response:
        assert b"Deepfake Analiz" in response.read()
        assert "frame-ancestors 'none'" in response.headers["Content-Security-Policy"]
    with pytest.raises(urllib.error.HTTPError) as exc:
        query(server, "../../app.py")
    assert exc.value.code == 404


@pytest.mark.parametrize(
    "route,file", [("", "index.html"), ("app.css", "app.css"), ("app.js", "app.js")]
)
def test_browser_serves_exact_desktop_studio_assets(server, route, file):
    with query(server, route) as response:
        assert response.read() == (ASSETS / file).read_bytes()


@pytest.mark.parametrize("origin,token", [(False, True), (True, False), (False, False)])
def test_mutation_requires_origin_and_session_token(server, origin, token):
    with pytest.raises(urllib.error.HTTPError) as exc:
        query(server, "clear", b"", origin=origin, token=token)
    assert exc.value.code == 403


def test_upload_real_preview_replace_clears_result_and_seek_range(server):
    with query(server, "file?name=p.png", photo()) as response:
        first = json.load(response)
    identity = first["file"]["id"]
    server.studio.state["result"] = {"fake_score": 0.5}
    with query(server, "file?name=q.jpg", photo("JPEG")) as response:
        second = json.load(response)
    assert second["result"] is None and second["file"]["id"] != identity
    route = f"media/{second['file']['id']}/preview"
    with query(server, route, headers={"Range": "bytes=0-7"}) as response:
        assert response.status == 206 and response.read() == b"\x89PNG\r\n\x1a\n"
    with pytest.raises(urllib.error.HTTPError) as exc:
        query(server, route, headers={"Range": "bytes=999999-"})
    assert exc.value.code == 416


def test_busy_analysis_rejects_replacement_and_second_start(server):
    server.studio.state["status"] = "running"
    with pytest.raises(DemoError):
        server.studio.upload(photo(), "p.png")
    with pytest.raises(DemoError):
        server.studio.start()
    with pytest.raises(DemoError):
        server.studio.clear()


def test_oversized_upload_rejected_before_reading_body(server):
    with pytest.raises(urllib.error.HTTPError) as exc:
        query(
            server,
            "file?name=video.mp4",
            b"",
            headers={"Content-Length": str(101 * 1024 * 1024)},
        )
    assert exc.value.code == 400
