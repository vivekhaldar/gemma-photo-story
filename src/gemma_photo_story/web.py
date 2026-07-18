from __future__ import annotations

import argparse
import json
import tempfile
import threading
import urllib.parse
import uuid
import webbrowser
from dataclasses import dataclass, field
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from .cli import (
    DEFAULT_MODEL,
    DEFAULT_OLLAMA_URL,
    IMAGE_EXTENSIONS,
    StoryError,
    check_prerequisites,
    run,
)


DEFAULT_PORT = 8765
MAX_FILE_BYTES = 100 * 1024 * 1024
LOOPBACK_HOSTS = {"127.0.0.1", "localhost", "::1"}
INDEX_PATH = Path(__file__).with_name("web").joinpath("index.html")


def is_allowed_host_header(value: str | None) -> bool:
    if not value:
        return False
    try:
        return urllib.parse.urlsplit(f"//{value}").hostname in LOOPBACK_HOSTS
    except ValueError:
        return False


def is_allowed_origin(value: str | None, port: int) -> bool:
    if value is None:
        return True
    try:
        parsed = urllib.parse.urlsplit(value)
        return (
            parsed.scheme == "http"
            and parsed.hostname in LOOPBACK_HOSTS
            and parsed.port == port
            and parsed.username is None
            and parsed.password is None
        )
    except ValueError:
        return False


def sanitize_upload_name(value: str) -> str:
    base_name = value.replace("\\", "/").rsplit("/", 1)[-1]
    base_name = "".join(
        character
        for character in base_name
        if character.isprintable() and character not in {"/", "\\", "\0"}
    ).strip()
    if base_name in {"", ".", ".."}:
        raise StoryError("The dropped photo has no usable filename")
    if Path(base_name).suffix.lower() not in IMAGE_EXTENSIONS:
        raise StoryError(f"Unsupported image type: {base_name}")
    return base_name


@dataclass
class SessionState:
    identifier: str
    root: Path
    model: str
    status: str = "selecting"
    logs: list[str] = field(default_factory=list)
    analysis: list[dict[str, Any]] | None = None
    story: str | None = None
    error: str | None = None
    lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    @property
    def photos_dir(self) -> Path:
        return self.root / "photos"

    @property
    def output_dir(self) -> Path:
        return self.root / "output"

    def add_log(self, message: str) -> None:
        with self.lock:
            self.logs.append(message)

    def snapshot(self) -> dict[str, Any]:
        with self.lock:
            return {
                "id": self.identifier,
                "status": self.status,
                "model": self.model,
                "file_count": len(list(self.photos_dir.iterdir())),
                "logs": list(self.logs),
                "analysis": self.analysis,
                "story": self.story,
                "error": self.error,
            }


class StoryHTTPServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(
        self,
        server_address: tuple[str, int],
        *,
        model: str,
        ollama_url: str,
        story_words: int,
        max_image_dimension: int,
        workspace: Path | None = None,
    ):
        super().__init__(server_address, StoryRequestHandler)
        self.model = model
        self.ollama_url = ollama_url
        self.story_words = story_words
        self.max_image_dimension = max_image_dimension
        self.sessions: dict[str, SessionState] = {}
        self.sessions_lock = threading.Lock()
        self._temporary_workspace: tempfile.TemporaryDirectory[str] | None = None
        if workspace is None:
            self._temporary_workspace = tempfile.TemporaryDirectory(
                prefix="gemma-photo-story-web-"
            )
            self.workspace = Path(self._temporary_workspace.name)
        else:
            self.workspace = workspace.resolve()
            self.workspace.mkdir(parents=True, exist_ok=True)

    def create_session(self) -> SessionState:
        identifier = uuid.uuid4().hex
        root = self.workspace / identifier
        (root / "photos").mkdir(parents=True)
        session = SessionState(identifier, root, self.model)
        with self.sessions_lock:
            self.sessions[identifier] = session
        return session

    def get_session(self, identifier: str) -> SessionState | None:
        with self.sessions_lock:
            return self.sessions.get(identifier)

    def start_session(self, session: SessionState) -> None:
        with session.lock:
            if session.status != "selecting":
                raise StoryError("This photo session has already started")
            if not any(session.photos_dir.iterdir()):
                raise StoryError("Drop at least one supported photo before starting")
            session.status = "running"
            session.logs.append(f"Starting local analysis with model {self.model}")
        threading.Thread(
            target=self._process_session,
            args=(session,),
            daemon=True,
            name=f"photo-story-{session.identifier[:8]}",
        ).start()

    def _process_session(self, session: SessionState) -> None:
        args = argparse.Namespace(
            images=session.photos_dir,
            output_dir=session.output_dir,
            model=self.model,
            ollama_url=self.ollama_url,
            max_image_dimension=self.max_image_dimension,
            story_words=self.story_words,
        )
        try:
            analysis_path, story_path = run(args, logger=session.add_log)
            analysis = json.loads(analysis_path.read_text(encoding="utf-8"))
            story = story_path.read_text(encoding="utf-8")
            with session.lock:
                session.analysis = analysis
                session.story = story
                session.status = "complete"
        except Exception as exc:
            with session.lock:
                session.error = str(exc)
                session.status = "error"

    def server_close(self) -> None:
        super().server_close()
        if self._temporary_workspace is not None:
            self._temporary_workspace.cleanup()


class StoryRequestHandler(BaseHTTPRequestHandler):
    server: StoryHTTPServer

    def do_GET(self) -> None:
        if not self._guard_local_host():
            return
        route = urllib.parse.urlsplit(self.path)
        if route.path == "/":
            self._send_html(INDEX_PATH.read_bytes())
            return
        if route.path == "/favicon.ico":
            self.send_response(HTTPStatus.NO_CONTENT)
            self.end_headers()
            return
        if route.path == "/api/config":
            self._send_json(
                {
                    "model": self.server.model,
                    "privacy": (
                        "Photos stay on this Mac. Only GPS coordinates are sent "
                        "to OpenStreetMap Nominatim."
                    ),
                }
            )
            return
        segments = self._segments(route.path)
        if len(segments) == 3 and segments[:2] == ["api", "sessions"]:
            session = self.server.get_session(segments[2])
            if session is None:
                self._send_error(HTTPStatus.NOT_FOUND, "Photo session not found")
                return
            self._send_json(session.snapshot())
            return
        self._send_error(HTTPStatus.NOT_FOUND, "Page not found")

    def do_POST(self) -> None:
        if not self._guard_state_change():
            return
        route = urllib.parse.urlsplit(self.path)
        if route.path == "/api/sessions":
            session = self.server.create_session()
            self._send_json(session.snapshot(), status=HTTPStatus.CREATED)
            return
        segments = self._segments(route.path)
        if (
            len(segments) == 4
            and segments[:2] == ["api", "sessions"]
            and segments[3] == "process"
        ):
            session = self.server.get_session(segments[2])
            if session is None:
                self._send_error(HTTPStatus.NOT_FOUND, "Photo session not found")
                return
            try:
                self.server.start_session(session)
            except StoryError as exc:
                self._send_error(HTTPStatus.BAD_REQUEST, str(exc))
                return
            self._send_json(session.snapshot(), status=HTTPStatus.ACCEPTED)
            return
        self._send_error(HTTPStatus.NOT_FOUND, "Endpoint not found")

    def do_PUT(self) -> None:
        if not self._guard_state_change():
            return
        route = urllib.parse.urlsplit(self.path)
        segments = self._segments(route.path)
        if not (
            len(segments) == 4
            and segments[:2] == ["api", "sessions"]
            and segments[3] == "files"
        ):
            self._send_error(HTTPStatus.NOT_FOUND, "Endpoint not found")
            return
        session = self.server.get_session(segments[2])
        if session is None:
            self._send_error(HTTPStatus.NOT_FOUND, "Photo session not found")
            return
        with session.lock:
            if session.status != "selecting":
                self._send_error(
                    HTTPStatus.CONFLICT,
                    "Photos cannot be added after analysis has started",
                )
                return
        query = urllib.parse.parse_qs(route.query)
        requested_name = query.get("name", [""])[0]
        try:
            file_name = sanitize_upload_name(requested_name)
            content_length = self._content_length()
        except StoryError as exc:
            self._send_error(HTTPStatus.BAD_REQUEST, str(exc))
            return
        content = self.rfile.read(content_length)
        if len(content) != content_length:
            self._send_error(
                HTTPStatus.BAD_REQUEST,
                "The photo transfer ended before all bytes arrived",
            )
            return
        with session.lock:
            if session.status != "selecting":
                self._send_error(
                    HTTPStatus.CONFLICT,
                    "Photos cannot be added after analysis has started",
                )
                return
            target = self._unique_target(session.photos_dir, file_name)
            target.write_bytes(content)
        self._send_json(
            {"file_name": target.name, "bytes": content_length},
            status=HTTPStatus.CREATED,
        )

    def _content_length(self) -> int:
        value = self.headers.get("Content-Length")
        if value is None:
            raise StoryError("Photo size was not provided")
        try:
            length = int(value)
        except ValueError as exc:
            raise StoryError("Photo size was not valid") from exc
        if length <= 0:
            raise StoryError("The dropped photo is empty")
        if length > MAX_FILE_BYTES:
            raise StoryError("Each photo must be 100 MB or smaller")
        return length

    @staticmethod
    def _unique_target(folder: Path, name: str) -> Path:
        candidate = folder / name
        counter = 2
        while candidate.exists():
            candidate = folder / f"{Path(name).stem}-{counter}{Path(name).suffix}"
            counter += 1
        return candidate

    @staticmethod
    def _segments(path: str) -> list[str]:
        return [segment for segment in path.split("/") if segment]

    def _guard_local_host(self) -> bool:
        if is_allowed_host_header(self.headers.get("Host")):
            return True
        self._send_error(
            HTTPStatus.MISDIRECTED_REQUEST,
            "This app accepts requests only through a loopback hostname",
        )
        return False

    def _guard_state_change(self) -> bool:
        if not self._guard_local_host():
            return False
        if is_allowed_origin(self.headers.get("Origin"), self.server.server_port):
            return True
        self._send_error(
            HTTPStatus.FORBIDDEN,
            "Cross-origin changes to this local app are not allowed",
        )
        return False

    def _send_html(self, content: bytes) -> None:
        self.send_response(HTTPStatus.OK)
        self._security_headers("text/html; charset=utf-8", len(content))
        self.end_headers()
        self.wfile.write(content)

    def _send_json(
        self,
        value: dict[str, Any],
        *,
        status: HTTPStatus = HTTPStatus.OK,
    ) -> None:
        content = json.dumps(value, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self._security_headers("application/json; charset=utf-8", len(content))
        self.end_headers()
        self.wfile.write(content)

    def _send_error(self, status: HTTPStatus, message: str) -> None:
        self._send_json({"error": message}, status=status)

    def _security_headers(self, content_type: str, content_length: int) -> None:
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(content_length))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Cross-Origin-Resource-Policy", "same-origin")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header("Permissions-Policy", "camera=(), microphone=(), geolocation=()")
        self.send_header(
            "Content-Security-Policy",
            "default-src 'self'; img-src 'self' blob: data:; "
            "style-src 'self' 'unsafe-inline'; script-src 'self' 'unsafe-inline'; "
            "connect-src 'self'; object-src 'none'; base-uri 'none'; "
            "frame-ancestors 'none'; form-action 'self'",
        )

    def log_message(self, format: str, *args: Any) -> None:
        return


def create_server(
    *,
    port: int,
    model: str,
    ollama_url: str,
    story_words: int,
    max_image_dimension: int,
    workspace: Path | None = None,
) -> StoryHTTPServer:
    return StoryHTTPServer(
        ("127.0.0.1", port),
        model=model,
        ollama_url=ollama_url,
        story_words=story_words,
        max_image_dimension=max_image_dimension,
        workspace=workspace,
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the loopback-only Gemma Photo Story web app."
    )
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--ollama-url", default=DEFAULT_OLLAMA_URL)
    parser.add_argument("--story-words", type=int, default=650)
    parser.add_argument("--max-image-dimension", type=int, default=1600)
    parser.add_argument(
        "--no-open",
        action="store_true",
        help="Do not open the app in the default browser",
    )
    args = parser.parse_args(argv)
    if not 1 <= args.port <= 65535:
        parser.error("--port must be between 1 and 65535")
    if args.story_words < 150:
        parser.error("--story-words must be at least 150")
    if args.max_image_dimension < 256:
        parser.error("--max-image-dimension must be at least 256")
    return args


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    try:
        check_prerequisites(args.ollama_url, args.model)
    except StoryError as exc:
        raise SystemExit(f"error: {exc}") from exc
    server = create_server(
        port=args.port,
        model=args.model,
        ollama_url=args.ollama_url,
        story_words=args.story_words,
        max_image_dimension=args.max_image_dimension,
    )
    url = f"http://127.0.0.1:{server.server_port}"
    print(f"Gemma Photo Story is ready at {url}")
    print(f"Model: {args.model}")
    print("Privacy: photos stay local; only GPS coordinates reach Nominatim.")
    if not args.no_open:
        threading.Timer(0.2, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping Gemma Photo Story")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
