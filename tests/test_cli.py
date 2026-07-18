import tempfile
import unittest
import urllib.request
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

from gemma_photo_story.cli import (
    NoRedirectHandler,
    PhotoMetadata,
    StoryError,
    discover_images,
    haversine_meters,
    parse_geocode_result,
    parse_json_object,
    print_json_log,
    same_photo_area,
    select_named_candidate,
    validate_network_url,
)


class NetworkPolicyTests(unittest.TestCase):
    def test_accepts_only_local_ollama(self) -> None:
        validate_network_url("http://127.0.0.1:11434/api/chat", purpose="ollama")
        validate_network_url("http://localhost:11434/api/tags", purpose="ollama")
        with self.assertRaises(StoryError):
            validate_network_url("https://example.com/api/chat", purpose="ollama")
        with self.assertRaises(StoryError):
            validate_network_url("https://127.0.0.1:11434/api/chat", purpose="ollama")
        with self.assertRaises(StoryError):
            validate_network_url(
                "http://user:secret@127.0.0.1:11434/api/chat",
                purpose="ollama",
            )

    def test_accepts_only_nominatim_for_remote_access(self) -> None:
        validate_network_url(
            "https://nominatim.openstreetmap.org/reverse?lat=1&lon=2",
            purpose="reverse_geocode",
        )
        with self.assertRaises(StoryError):
            validate_network_url(
                "https://maps.googleapis.com/maps/api/geocode/json",
                purpose="reverse_geocode",
            )
        with self.assertRaises(StoryError):
            validate_network_url(
                "http://nominatim.openstreetmap.org/reverse",
                purpose="reverse_geocode",
            )
        with self.assertRaises(StoryError):
            validate_network_url(
                "https://nominatim.openstreetmap.org/other",
                purpose="reverse_geocode",
            )
        with self.assertRaises(StoryError):
            validate_network_url(
                "https://nominatim.openstreetmap.org:444/reverse",
                purpose="reverse_geocode",
            )

    def test_rejects_http_redirects(self) -> None:
        handler = NoRedirectHandler()
        request = urllib.request.Request(
            "https://nominatim.openstreetmap.org/reverse"
        )
        with self.assertRaises(StoryError):
            handler.redirect_request(
                request,
                None,
                302,
                "Found",
                {},
                "https://example.com/escaped",
            )


class ParsingTests(unittest.TestCase):
    def test_prints_readable_json_log(self) -> None:
        stream = StringIO()
        with redirect_stdout(stream):
            print_json_log(
                "Reverse-geocode result",
                {"city": "Santa Bárbara", "name": "Courthouse"},
            )
        self.assertEqual(
            stream.getvalue(),
            'Reverse-geocode result:\n{\n  "city": "Santa Bárbara",\n'
            '  "name": "Courthouse"\n}\n',
        )

    def test_parses_plain_and_fenced_json(self) -> None:
        self.assertEqual(parse_json_object('{"mood": "quiet"}'), {"mood": "quiet"})
        self.assertEqual(
            parse_json_object('```json\n{"mood": "quiet"}\n```'),
            {"mood": "quiet"},
        )

    def test_rejects_non_object_json(self) -> None:
        with self.assertRaises(StoryError):
            parse_json_object("[]")

    def test_extracts_stable_geocode_fields(self) -> None:
        result = {
            "features": [
                {
                    "properties": {
                        "geocoding": {
                            "label": "Example Trail, Example City",
                            "name": "Example Trail",
                            "type": "street",
                            "city": "Example City",
                            "extra": "discard me",
                        }
                    }
                }
            ]
        }
        parsed = parse_geocode_result(result)
        self.assertEqual(parsed["name"], "Example Trail")
        self.assertEqual(parsed["city"], "Example City")
        self.assertNotIn("extra", parsed)
        self.assertEqual(parsed["source"], "OpenStreetMap Nominatim")

    def test_selects_nearby_named_place_and_rejects_distant_one(self) -> None:
        result = {
            "features": [
                {
                    "geometry": {"coordinates": [-119.7021, 34.4240]},
                    "properties": {
                        "geocoding": {
                            "name": "Historic Courthouse",
                            "label": "Historic Courthouse, Example Street",
                            "type": "house",
                        }
                    },
                },
                {
                    "geometry": {"coordinates": [-120.0, 35.0]},
                    "properties": {
                        "geocoding": {
                            "name": "Distant Place",
                            "label": "Distant Place",
                            "type": "house",
                        }
                    },
                },
            ]
        }
        selected = select_named_candidate(result, 34.4241, -119.7022)
        self.assertIsNotNone(selected)
        self.assertEqual(selected["name"], "Historic Courthouse")
        self.assertLess(selected["distance_meters"], 20)

    def test_distance_and_photo_area(self) -> None:
        self.assertLess(haversine_meters(34.0, -118.0, 34.0001, -118.0001), 20)
        first = PhotoMetadata("a.jpg", "/a.jpg", None, 34.0, -118.0, None)
        near = PhotoMetadata("b.jpg", "/b.jpg", None, 34.001, -118.001, None)
        far = PhotoMetadata("c.jpg", "/c.jpg", None, 35.0, -119.0, None)
        self.assertTrue(same_photo_area(first, near))
        self.assertFalse(same_photo_area(first, far))


class DiscoveryTests(unittest.TestCase):
    def test_discovers_supported_images_without_recursing(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "b.HEIC").write_bytes(b"fixture")
            (root / "a.jpg").write_bytes(b"fixture")
            (root / "notes.txt").write_text("not an image", encoding="utf-8")
            nested = root / "nested"
            nested.mkdir()
            (nested / "ignored.png").write_bytes(b"fixture")
            self.assertEqual(
                [path.name for path in discover_images(root)],
                ["a.jpg", "b.HEIC"],
            )

    def test_rejects_empty_folder(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaises(StoryError):
                discover_images(Path(temp))


if __name__ == "__main__":
    unittest.main()
