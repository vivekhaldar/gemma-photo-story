from __future__ import annotations

import argparse
import base64
import json
import math
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


DEFAULT_MODEL = "gemma4:26b"
DEFAULT_OLLAMA_URL = "http://127.0.0.1:11434"
NOMINATIM_ORIGIN = "https://nominatim.openstreetmap.org"
NOMINATIM_REVERSE_URL = f"{NOMINATIM_ORIGIN}/reverse"
NOMINATIM_SEARCH_URL = f"{NOMINATIM_ORIGIN}/search"
USER_AGENT = "gemma-photo-story/0.1 (github.com/vivekhaldar/gemma-photo-story)"
IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
    ".heic",
    ".heif",
    ".tif",
    ".tiff",
}
LOCAL_HOSTS = {"127.0.0.1", "localhost", "::1"}
NOMINATIM_PATHS = {"/reverse", "/search"}


@dataclass
class PhotoMetadata:
    file_name: str
    source_path: str
    captured_at: str | None
    latitude: float | None
    longitude: float | None
    altitude_meters: float | None


@dataclass
class PhotoAnalysis:
    metadata: PhotoMetadata
    place: dict[str, Any] | None
    visual: dict[str, Any]


class StoryError(RuntimeError):
    pass


class NoRedirectHandler(urllib.request.HTTPRedirectHandler):
    """Reject redirects so an allowlisted request cannot escape its destination."""

    def redirect_request(
        self,
        req: urllib.request.Request,
        fp: Any,
        code: int,
        msg: str,
        headers: Any,
        newurl: str,
    ) -> urllib.request.Request | None:
        raise StoryError(f"HTTP redirect rejected: {req.full_url} -> {newurl}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Describe a folder of photos with local Gemma, reverse-geocode EXIF GPS "
            "coordinates, and ask local Gemma to write a grounded travel story."
        )
    )
    parser.add_argument("images", nargs="?", type=Path, help="Folder containing photos")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output"),
        help="Destination for analysis.json, story.md, and the geocode cache",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Local Ollama model tag")
    parser.add_argument(
        "--ollama-url",
        default=DEFAULT_OLLAMA_URL,
        help="Local Ollama origin; non-local hosts are rejected",
    )
    parser.add_argument(
        "--max-image-dimension",
        type=int,
        default=1600,
        help="Longest JPEG preview edge sent to local Ollama",
    )
    parser.add_argument(
        "--story-words",
        type=int,
        default=650,
        help="Approximate target length for the final narrative",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Check local prerequisites and model availability, then exit",
    )
    args = parser.parse_args(argv)
    if not args.check and args.images is None:
        parser.error("images is required unless --check is used")
    if args.max_image_dimension < 256:
        parser.error("--max-image-dimension must be at least 256")
    if args.story_words < 150:
        parser.error("--story-words must be at least 150")
    return args


def validate_network_url(url: str, *, purpose: str) -> None:
    parsed = urllib.parse.urlparse(url)
    host = parsed.hostname
    try:
        port = parsed.port
    except ValueError as exc:
        raise StoryError(f"Invalid network URL: {url!r}") from exc
    if parsed.username is not None or parsed.password is not None or parsed.fragment:
        raise StoryError(f"Credentials and fragments are not allowed in URLs: {url!r}")
    if purpose == "ollama":
        if parsed.scheme != "http" or host not in LOCAL_HOSTS:
            raise StoryError(
                f"Ollama must use plain HTTP on a local host; rejected {url!r}"
            )
        return
    if purpose == "reverse_geocode":
        if (
            parsed.scheme != "https"
            or host != "nominatim.openstreetmap.org"
            or port not in (None, 443)
            or parsed.path not in NOMINATIM_PATHS
        ):
            raise StoryError(
                "Reverse geocoding is restricted to the HTTPS /reverse and /search "
                "endpoints at nominatim.openstreetmap.org"
            )
        return
    raise StoryError(f"Unknown network purpose: {purpose}")


def direct_json_request(
    url: str,
    *,
    purpose: str,
    method: str = "GET",
    payload: dict[str, Any] | None = None,
    headers: dict[str, str] | None = None,
    timeout: float = 300,
) -> dict[str, Any]:
    validate_network_url(url, purpose=purpose)
    body = None if payload is None else json.dumps(payload).encode("utf-8")
    request_headers = {"Accept": "application/json"}
    if body is not None:
        request_headers["Content-Type"] = "application/json"
    if headers:
        request_headers.update(headers)
    request = urllib.request.Request(
        url,
        data=body,
        headers=request_headers,
        method=method,
    )
    # Ignore proxy environment variables so the only remote destination is the
    # exact allowlisted Nominatim host and Ollama remains a direct local call.
    opener = urllib.request.build_opener(
        urllib.request.ProxyHandler({}),
        NoRedirectHandler(),
    )
    try:
        with opener.open(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise StoryError(f"HTTP {exc.code} from {url}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise StoryError(f"Could not reach {url}: {exc.reason}") from exc
    except json.JSONDecodeError as exc:
        raise StoryError(f"Non-JSON response from {url}: {exc}") from exc


def discover_images(folder: Path) -> list[Path]:
    folder = folder.expanduser().resolve()
    if not folder.is_dir():
        raise StoryError(f"Image folder does not exist: {folder}")
    images = sorted(
        path
        for path in folder.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not images:
        raise StoryError(f"No supported images found in {folder}")
    return images


def run_checked(command: list[str]) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError as exc:
        raise StoryError(f"Required command not found: {command[0]}") from exc
    except subprocess.CalledProcessError as exc:
        detail = exc.stderr.strip() or exc.stdout.strip()
        raise StoryError(f"Command failed: {' '.join(command)}\n{detail}") from exc


def extract_metadata(images: list[Path]) -> list[PhotoMetadata]:
    command = [
        "exiftool",
        "-n",
        "-json",
        "-FileName",
        "-DateTimeOriginal",
        "-CreateDate",
        "-GPSLatitude",
        "-GPSLongitude",
        "-GPSAltitude",
        *[str(path) for path in images],
    ]
    records = json.loads(run_checked(command).stdout)
    metadata: list[PhotoMetadata] = []
    for record in records:
        captured_at = record.get("DateTimeOriginal") or record.get("CreateDate")
        metadata.append(
            PhotoMetadata(
                file_name=str(record.get("FileName") or Path(record["SourceFile"]).name),
                source_path=str(Path(record["SourceFile"]).resolve()),
                captured_at=captured_at,
                latitude=_optional_float(record.get("GPSLatitude")),
                longitude=_optional_float(record.get("GPSLongitude")),
                altitude_meters=_optional_float(record.get("GPSAltitude")),
            )
        )
    metadata.sort(key=lambda item: (_date_sort_key(item.captured_at), item.file_name))
    return metadata


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


def _date_sort_key(value: str | None) -> tuple[int, str]:
    if not value:
        return (1, "")
    return (0, value)


def prepare_jpeg(source: Path, destination: Path, max_dimension: int) -> None:
    run_checked(
        [
            "sips",
            "-s",
            "format",
            "jpeg",
            "-s",
            "formatOptions",
            "85",
            "-Z",
            str(max_dimension),
            str(source),
            "--out",
            str(destination),
        ]
    )


def image_as_base64(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("ascii")


def ollama_chat(
    ollama_url: str,
    *,
    model: str,
    system: str,
    prompt: str,
    images: list[str] | None = None,
    json_mode: bool = False,
    temperature: float = 0.2,
) -> str:
    origin = ollama_url.rstrip("/")
    url = f"{origin}/api/chat"
    user_message: dict[str, Any] = {"role": "user", "content": prompt}
    if images:
        user_message["images"] = images
    payload: dict[str, Any] = {
        "model": model,
        "stream": False,
        "think": False,
        "messages": [
            {"role": "system", "content": system},
            user_message,
        ],
        "options": {"temperature": temperature},
    }
    if json_mode:
        payload["format"] = "json"
    result = direct_json_request(
        url,
        purpose="ollama",
        method="POST",
        payload=payload,
        timeout=600,
    )
    try:
        content = result["message"]["content"].strip()
    except (KeyError, TypeError, AttributeError) as exc:
        raise StoryError(f"Unexpected Ollama response shape: {result}") from exc
    if not content:
        raise StoryError(f"Ollama returned an empty response for model {model}")
    return content


def parse_json_object(content: str) -> dict[str, Any]:
    stripped = content.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        stripped = "\n".join(lines).strip()
    try:
        value = json.loads(stripped)
    except json.JSONDecodeError as exc:
        raise StoryError(f"Gemma did not return valid JSON: {exc}\n{content}") from exc
    if not isinstance(value, dict):
        raise StoryError("Gemma JSON response must be an object")
    return value


def describe_image(
    jpeg_path: Path,
    *,
    metadata: PhotoMetadata,
    place: dict[str, Any] | None,
    ollama_url: str,
    model: str,
) -> dict[str, Any]:
    prompt = f"""
Describe this vacation photograph as grounded evidence for a later travel story.

Known local metadata:
- filename: {metadata.file_name}
- captured_at: {metadata.captured_at or "unknown"}
- reverse_geocoded_place: {json.dumps(place, ensure_ascii=False) if place else "unavailable"}

The reverse-geocoded place is contextual evidence from the nearest suitable map
object, not proof that the exact named feature is visible. Use it to resolve broad
setting and plausible landmarks, but keep pixels and map evidence distinct.

Return exactly one JSON object with these keys:
- concise_description: 1-2 factual sentences describing the whole image
- subjects: array of visible subjects, objects, artworks, buildings, or scenery
- setting: concise description of the apparent setting
- mood: visual mood based only on composition, light, color, and activity
- story_details: array of 2-5 concrete visual details useful in a narrative
- readable_text: array containing only text actually legible in the image
- landmark_hypotheses: array of possible named landmarks or artworks, each with
  name, confidence from 0 to 1, and evidence
- uncertainty: anything important that cannot be determined from pixels alone

Do not identify private people. Do not invent conversations, relationships,
intentions, travel companions, or off-camera events. Treat named landmarks and
artworks as hypotheses unless established jointly by visual and map evidence.
Do not quote stylized signage in concise_description unless every character is
clear; uncertain lettering belongs in readable_text and uncertainty.
""".strip()
    content = ollama_chat(
        ollama_url,
        model=model,
        system=(
            "You are a careful visual archivist. You have no internet access and must "
            "separate direct visual evidence, reverse-geocoded context, and inference."
        ),
        prompt=prompt,
        images=[image_as_base64(jpeg_path)],
        json_mode=True,
        temperature=0.1,
    )
    return parse_json_object(content)


class ReverseGeocoder:
    def __init__(self, cache_path: Path):
        self.cache_path = cache_path
        self.cache = self._load_cache()
        self.last_request_at: float | None = None

    def _load_cache(self) -> dict[str, Any]:
        if not self.cache_path.exists():
            return {}
        try:
            value = json.loads(self.cache_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise StoryError(f"Invalid geocode cache {self.cache_path}: {exc}") from exc
        if not isinstance(value, dict):
            raise StoryError(f"Geocode cache must be a JSON object: {self.cache_path}")
        return value

    def reverse(self, latitude: float, longitude: float) -> dict[str, Any]:
        key = f"{latitude:.7f},{longitude:.7f}"
        cached = self.cache.get(key)
        if isinstance(cached, dict) and cached.get("lookup_complete") is True:
            return cached
        if isinstance(cached, dict):
            place = cached
        else:
            query = urllib.parse.urlencode(
                {
                    "lat": f"{latitude:.8f}",
                    "lon": f"{longitude:.8f}",
                    "format": "geocodejson",
                    "addressdetails": "1",
                    "extratags": "1",
                    "namedetails": "1",
                    "zoom": "18",
                    "accept-language": "en",
                }
            )
            result = self._request(f"{NOMINATIM_REVERSE_URL}?{query}")
            place = parse_geocode_result(result)
        named_place = self._search_exact_address(place, latitude, longitude)
        if named_place is not None:
            place["named_place"] = named_place
        place["lookup_complete"] = True
        self.cache[key] = place
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.cache_path.write_text(
            json.dumps(self.cache, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return place

    def _search_exact_address(
        self,
        place: dict[str, Any],
        latitude: float,
        longitude: float,
    ) -> dict[str, Any] | None:
        if not place.get("housenumber") or not place.get("street"):
            return None
        parts = [
            place.get("housenumber"),
            place.get("street"),
            place.get("city") or place.get("locality"),
            place.get("state"),
            place.get("postcode"),
            place.get("country"),
        ]
        address = ", ".join(str(part) for part in parts if part)
        query = urllib.parse.urlencode(
            {
                "q": address,
                "format": "geocodejson",
                "addressdetails": "1",
                "namedetails": "1",
                "limit": "5",
                "accept-language": "en",
            }
        )
        result = self._request(f"{NOMINATIM_SEARCH_URL}?{query}")
        return select_named_candidate(result, latitude, longitude)

    def _request(self, url: str) -> dict[str, Any]:
        self._respect_rate_limit()
        result = direct_json_request(
            url,
            purpose="reverse_geocode",
            headers={"User-Agent": USER_AGENT},
            timeout=30,
        )
        self.last_request_at = time.monotonic()
        return result

    def _respect_rate_limit(self) -> None:
        if self.last_request_at is None:
            return
        elapsed = time.monotonic() - self.last_request_at
        if elapsed < 1.05:
            time.sleep(1.05 - elapsed)


def parse_geocode_result(result: dict[str, Any]) -> dict[str, Any]:
    features = result.get("features")
    if not isinstance(features, list) or not features:
        return {"label": "Unknown place", "source": "OpenStreetMap Nominatim"}
    properties = features[0].get("properties", {})
    geocoding = properties.get("geocoding", {})
    if not isinstance(geocoding, dict):
        geocoding = {}
    keep = {
        "name",
        "label",
        "type",
        "housenumber",
        "street",
        "locality",
        "district",
        "city",
        "county",
        "state",
        "postcode",
        "country",
    }
    place = {key: geocoding[key] for key in keep if geocoding.get(key) is not None}
    place["source"] = "OpenStreetMap Nominatim"
    return place


def select_named_candidate(
    result: dict[str, Any], latitude: float, longitude: float
) -> dict[str, Any] | None:
    candidates: list[dict[str, Any]] = []
    for feature in result.get("features", []):
        if not isinstance(feature, dict):
            continue
        geocoding = feature.get("properties", {}).get("geocoding", {})
        coordinates = feature.get("geometry", {}).get("coordinates")
        if not isinstance(geocoding, dict) or not geocoding.get("name"):
            continue
        if not isinstance(coordinates, list) or len(coordinates) < 2:
            continue
        distance = haversine_meters(
            latitude,
            longitude,
            float(coordinates[1]),
            float(coordinates[0]),
        )
        if distance > 250:
            continue
        candidates.append(
            {
                "name": geocoding["name"],
                "label": geocoding.get("label"),
                "type": geocoding.get("type"),
                "distance_meters": round(distance, 1),
            }
        )
    if not candidates:
        return None
    candidates.sort(key=lambda item: item["distance_meters"])
    return candidates[0]


def haversine_meters(
    latitude_a: float,
    longitude_a: float,
    latitude_b: float,
    longitude_b: float,
) -> float:
    radius = 6_371_000.0
    phi_a = math.radians(latitude_a)
    phi_b = math.radians(latitude_b)
    delta_phi = math.radians(latitude_b - latitude_a)
    delta_lambda = math.radians(longitude_b - longitude_a)
    value = (
        math.sin(delta_phi / 2) ** 2
        + math.cos(phi_a) * math.cos(phi_b) * math.sin(delta_lambda / 2) ** 2
    )
    return radius * 2 * math.atan2(math.sqrt(value), math.sqrt(1 - value))


def location_chapters(analyses: list[PhotoAnalysis]) -> list[dict[str, Any]]:
    chapters: list[dict[str, Any]] = []
    for analysis in analyses:
        place = analysis.place or {}
        names = []
        named_place = place.get("named_place")
        if isinstance(named_place, dict) and named_place.get("name"):
            names.append(named_place["name"])
        if place.get("name"):
            names.append(place["name"])
        for field in ("locality", "district", "city", "state"):
            if place.get(field):
                names.append(place[field])
        anchor = {
            "file_name": analysis.metadata.file_name,
            "supported_names": list(dict.fromkeys(names)),
            "map_label": place.get("label"),
        }
        if chapters and same_photo_area(chapters[-1]["metadata"], analysis.metadata):
            chapters[-1]["files"].append(analysis.metadata.file_name)
            chapters[-1]["anchors"].append(anchor)
            continue
        chapters.append(
            {
                "files": [analysis.metadata.file_name],
                "anchors": [anchor],
                "metadata": analysis.metadata,
            }
        )
    for chapter in chapters:
        chapter.pop("metadata", None)
    return chapters


def same_photo_area(first: PhotoMetadata, second: PhotoMetadata) -> bool:
    if (
        first.latitude is None
        or first.longitude is None
        or second.latitude is None
        or second.longitude is None
    ):
        return False
    return (
        haversine_meters(
            first.latitude,
            first.longitude,
            second.latitude,
            second.longitude,
        )
        <= 1_000
    )


def write_story(
    analyses: list[PhotoAnalysis],
    *,
    ollama_url: str,
    model: str,
    target_words: int,
) -> str:
    evidence = []
    for index, analysis in enumerate(analyses, start=1):
        evidence.append(
            {
                "sequence": index,
                "file_name": analysis.metadata.file_name,
                "captured_at": analysis.metadata.captured_at,
                "place": analysis.place,
                "visual": analysis.visual,
            }
        )
    chapters = location_chapters(analyses)
    prompt = f"""
Write a warm, polished vacation-photo narrative of about {target_words} words from
the chronological evidence below.

The result must be Markdown with:
1. A specific, evocative title.
2. One coherent story rather than a list of captions.
3. A natural progression through the places and times.
4. Place names woven in only where supported by reverse geocoding.
5. Concrete visual details from the photographs.
6. A brief final note titled "Location notes" explaining that GPS-derived labels
   identify nearby map features and may not prove the exact photographed subject.

Never invent companions, conversations, emotions experienced by the photographer,
transportation, meals, weather beyond what is visible, or events outside the frames.
Do not convert a low-confidence landmark hypothesis into fact. When the visual
description and geocoder differ in specificity, prefer cautious language such as
"near," "in the area of," or "the coordinates place the photograph at."

The narrative must explicitly name at least one supported place from every
location chapter below. A `named_place` came from an exact-address Nominatim
search near the original coordinate and is stronger than a generic street label,
but it still establishes nearby place context rather than what the camera depicts.
Do not quote uncertain stylized signage from the visual descriptions.

Location chapters:
{json.dumps(chapters, indent=2, ensure_ascii=False)}

Evidence:
{json.dumps(evidence, indent=2, ensure_ascii=False)}
""".strip()
    return ollama_chat(
        ollama_url,
        model=model,
        system=(
            "You are an elegant but strictly evidence-grounded travel writer. "
            "You have no internet access."
        ),
        prompt=prompt,
        json_mode=False,
        temperature=0.5,
    )


def check_prerequisites(ollama_url: str, model: str) -> None:
    missing = [name for name in ("exiftool", "sips") if shutil.which(name) is None]
    if missing:
        raise StoryError(f"Missing required commands: {', '.join(missing)}")
    result = direct_json_request(
        f"{ollama_url.rstrip('/')}/api/tags",
        purpose="ollama",
        timeout=10,
    )
    names = {
        item.get("name")
        for item in result.get("models", [])
        if isinstance(item, dict)
    }
    if model not in names:
        raise StoryError(
            f"Local Ollama model {model!r} is not installed. Available: "
            f"{', '.join(sorted(name for name in names if isinstance(name, str)))}"
        )


def print_json_log(label: str, value: Any) -> None:
    rendered = json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True)
    print(f"{label}:\n{rendered}", flush=True)


def run(args: argparse.Namespace) -> tuple[Path, Path]:
    check_prerequisites(args.ollama_url, args.model)
    images = discover_images(args.images)
    metadata = extract_metadata(images)
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    geocoder = ReverseGeocoder(output_dir / "geocode-cache.json")
    analyses: list[PhotoAnalysis] = []

    with tempfile.TemporaryDirectory(prefix="gemma-photo-story-") as temp:
        temp_dir = Path(temp)
        for index, item in enumerate(metadata, start=1):
            place = None
            if item.latitude is not None and item.longitude is not None:
                print(f"[{index}/{len(metadata)}] Reverse-geocoding GPS", flush=True)
                place = geocoder.reverse(item.latitude, item.longitude)
                print_json_log(
                    f"[{index}/{len(metadata)}] Reverse-geocode result",
                    place,
                )
            else:
                print(
                    f"[{index}/{len(metadata)}] Reverse-geocode skipped: no GPS metadata",
                    flush=True,
                )
            print(
                f"[{index}/{len(metadata)}] Describing {item.file_name} "
                f"with model {args.model}",
                flush=True,
            )
            jpeg_path = temp_dir / f"{index:04d}-{Path(item.file_name).stem}.jpg"
            prepare_jpeg(Path(item.source_path), jpeg_path, args.max_image_dimension)
            visual = describe_image(
                jpeg_path,
                metadata=item,
                place=place,
                ollama_url=args.ollama_url,
                model=args.model,
            )
            print_json_log(
                f"[{index}/{len(metadata)}] Image description from {args.model}",
                visual,
            )
            analyses.append(PhotoAnalysis(metadata=item, place=place, visual=visual))

    analysis_path = output_dir / "analysis.json"
    analysis_path.write_text(
        json.dumps([asdict(item) for item in analyses], indent=2, ensure_ascii=False)
        + "\n",
        encoding="utf-8",
    )
    print(f"Writing narrative with model {args.model}", flush=True)
    story = write_story(
        analyses,
        ollama_url=args.ollama_url,
        model=args.model,
        target_words=args.story_words,
    )
    story_path = output_dir / "story.md"
    story_path.write_text(story.rstrip() + "\n", encoding="utf-8")
    return analysis_path, story_path


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    try:
        if args.check:
            check_prerequisites(args.ollama_url, args.model)
            print(f"Ready: {args.model} is installed and local prerequisites are present.")
            return
        analysis_path, story_path = run(args)
    except StoryError as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
    print(f"Analysis: {analysis_path}")
    print(f"Story:    {story_path}")


if __name__ == "__main__":
    main()
