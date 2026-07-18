import base64
import json
import tempfile
import threading
import unittest
import urllib.error
import urllib.request
from pathlib import Path

from gemma_photo_story.cli import StoryError, run_checked
from gemma_photo_story.web import (
    INDEX_PATH,
    create_server,
    is_allowed_host_header,
    is_allowed_origin,
    sanitize_upload_name,
)


TINY_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII="
)


class UploadValidationTests(unittest.TestCase):
    def test_keeps_only_a_safe_supported_basename(self) -> None:
        self.assertEqual(sanitize_upload_name("trip/day-1/IMG_0001.HEIC"), "IMG_0001.HEIC")
        self.assertEqual(sanitize_upload_name(r"trip\IMG_0002.jpg"), "IMG_0002.jpg")

    def test_rejects_missing_or_unsupported_names(self) -> None:
        for value in ("", "..", "notes.txt"):
            with self.subTest(value=value), self.assertRaises(StoryError):
                sanitize_upload_name(value)

    def test_accepts_only_loopback_host_headers(self) -> None:
        self.assertTrue(is_allowed_host_header("127.0.0.1:8765"))
        self.assertTrue(is_allowed_host_header("localhost:8765"))
        self.assertTrue(is_allowed_host_header("[::1]:8765"))
        self.assertFalse(is_allowed_host_header("example.com"))
        self.assertFalse(is_allowed_host_header(None))

    def test_accepts_only_same_port_loopback_origins(self) -> None:
        self.assertTrue(is_allowed_origin(None, 8765))
        self.assertTrue(is_allowed_origin("http://127.0.0.1:8765", 8765))
        self.assertTrue(is_allowed_origin("http://localhost:8765", 8765))
        self.assertFalse(is_allowed_origin("https://127.0.0.1:8765", 8765))
        self.assertFalse(is_allowed_origin("http://127.0.0.1:9000", 8765))
        self.assertFalse(is_allowed_origin("https://example.com", 8765))


class StaticPageTests(unittest.TestCase):
    def test_single_page_has_drop_story_and_no_remote_assets(self) -> None:
        page = INDEX_PATH.read_text(encoding="utf-8")
        self.assertIn('id="dropZone"', page)
        self.assertIn('id="storyContent"', page)
        self.assertIn("webkitdirectory", page)
        self.assertIn("Preparing local preview", page)
        self.assertNotIn("<script src=", page)
        self.assertNotIn("<link rel=\"stylesheet\"", page)
        self.assertNotIn("https://", page)


class LoopbackServerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.server = create_server(
            port=0,
            model="gemma4:26b",
            ollama_url="http://127.0.0.1:11434",
            story_words=150,
            max_image_dimension=512,
            workspace=Path(self.temporary_directory.name),
        )
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.origin = f"http://127.0.0.1:{self.server.server_port}"
        self.opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    def tearDown(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)
        self.temporary_directory.cleanup()

    def request_json(
        self,
        path: str,
        *,
        method: str = "GET",
        data: bytes | None = None,
    ) -> tuple[int, dict]:
        request = urllib.request.Request(
            f"{self.origin}{path}",
            data=data,
            method=method,
        )
        with self.opener.open(request, timeout=2) as response:
            return response.status, json.loads(response.read().decode("utf-8"))

    def test_serves_page_with_security_headers(self) -> None:
        with self.opener.open(f"{self.origin}/", timeout=2) as response:
            page = response.read().decode("utf-8")
            self.assertEqual(response.status, 200)
            self.assertIn("default-src 'self'", response.headers["Content-Security-Policy"])
            self.assertEqual(response.headers["X-Content-Type-Options"], "nosniff")
            self.assertIn("Gemma Photo Story", page)

    def test_creates_session_and_accepts_local_photo_bytes(self) -> None:
        status, session = self.request_json("/api/sessions", method="POST", data=b"")
        self.assertEqual(status, 201)
        status, uploaded = self.request_json(
            f"/api/sessions/{session['id']}/files?name=IMG_0001.png",
            method="PUT",
            data=TINY_PNG,
        )
        self.assertEqual(status, 201)
        self.assertEqual(uploaded["file_name"], "IMG_0001.png")
        self.assertTrue(uploaded["preview_url"].endswith("IMG_0001.png.preview.jpg"))
        with self.opener.open(f"{self.origin}{uploaded['preview_url']}", timeout=2) as response:
            preview = response.read()
            self.assertEqual(response.status, 200)
            self.assertEqual(response.headers["Content-Type"], "image/jpeg")
            self.assertEqual(preview[:2], b"\xff\xd8")
        _, snapshot = self.request_json(f"/api/sessions/{session['id']}")
        self.assertEqual(snapshot["file_count"], 1)

    def test_rejects_invalid_image_bytes_without_retaining_file(self) -> None:
        _, session = self.request_json("/api/sessions", method="POST", data=b"")
        request = urllib.request.Request(
            f"{self.origin}/api/sessions/{session['id']}/files?name=broken.HEIC",
            data=b"not an image",
            method="PUT",
        )
        with self.assertRaises(urllib.error.HTTPError) as context:
            self.opener.open(request, timeout=2)
        response = context.exception
        try:
            self.assertEqual(response.code, 400)
            error = json.loads(response.read().decode("utf-8"))
            self.assertIn("Could not prepare a local preview", error["error"])
        finally:
            response.close()
        _, snapshot = self.request_json(f"/api/sessions/{session['id']}")
        self.assertEqual(snapshot["file_count"], 0)

    def test_generates_jpeg_preview_for_heic_upload(self) -> None:
        source = Path(self.temporary_directory.name) / "source.png"
        heic = Path(self.temporary_directory.name) / "source.heic"
        source.write_bytes(TINY_PNG)
        run_checked(
            ["sips", "-s", "format", "heic", str(source), "--out", str(heic)]
        )
        _, session = self.request_json("/api/sessions", method="POST", data=b"")
        status, uploaded = self.request_json(
            f"/api/sessions/{session['id']}/files?name=vacation.HEIC",
            method="PUT",
            data=heic.read_bytes(),
        )
        self.assertEqual(status, 201)
        with self.opener.open(f"{self.origin}{uploaded['preview_url']}", timeout=2) as response:
            self.assertEqual(response.headers["Content-Type"], "image/jpeg")
            self.assertEqual(response.read(2), b"\xff\xd8")

    def test_rejects_processing_an_empty_session(self) -> None:
        _, session = self.request_json("/api/sessions", method="POST", data=b"")
        request = urllib.request.Request(
            f"{self.origin}/api/sessions/{session['id']}/process",
            data=b"",
            method="POST",
        )
        with self.assertRaises(urllib.error.HTTPError) as context:
            self.opener.open(request, timeout=2)
        response = context.exception
        try:
            self.assertEqual(response.code, 400)
            error = json.loads(response.read().decode("utf-8"))
            self.assertIn("Drop at least one", error["error"])
        finally:
            response.close()


if __name__ == "__main__":
    unittest.main()
