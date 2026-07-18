# Gemma Photo Story

Turn a folder of vacation photos into a grounded travel narrative using a local
Gemma 4 vision model and EXIF metadata.

The pipeline:

1. Reads capture time and GPS coordinates locally with ExifTool.
2. Converts each image to a temporary, resized JPEG with macOS `sips`.
3. Sends the JPEG only to local Ollama for a factual visual description.
4. Sends only GPS coordinates to OpenStreetMap Nominatim for reverse geocoding,
   then may query the returned exact address on the same host to recover a nearby
   named landmark.
5. Sends the local descriptions, timestamps, and returned place labels to local
   Ollama for a chronological Markdown travel story.

Raw images are never sent to an internet service. The code rejects non-local
Ollama URLs and rejects every remote host except
`nominatim.openstreetmap.org`.

## Requirements

- macOS
- [`uv`](https://docs.astral.sh/uv/)
- [Ollama](https://ollama.com/) running locally
- `gemma4:26b` installed in Ollama
- ExifTool
- `sips` (included with macOS)

Confirm the local setup:

```sh
uv run gemma-photo-story --check
```

## Run

### Single-page web app

Start the loopback-only app and open it in your browser:

```sh
uv run gemma-photo-story-web
```

Then drag a folder or a set of photos onto the page. The app shows local
JPEG thumbnails generated locally with `sips` (including for HEIC originals),
local progress, the resolved place and Gemma description for each image, and
the finished narrative with its photographs placed inline, plus copy and
Markdown-download controls. Each original is transferred to the loopback
server once and reused for Gemma analysis.

The page caches small JSON results in browser `localStorage`: reverse-geocodes
are keyed by GPS coordinates, descriptions by photo SHA-256, model, location
context, and prompt version, and narratives by the ordered photo set, model,
target length, and prompt version. A repeated selection therefore reuses saved
place names, descriptions, and narrative instead of calling Nominatim or
running Gemma inference again. Photo bytes and thumbnail data are never stored
in `localStorage`. Use **Clear saved results** on the page to remove this cache.

Normal browser security does not reveal absolute paths for dropped files. The
page therefore transfers the selected bytes to the Python server over
`127.0.0.1`, where they are held in a temporary directory until the server
stops. This is local loopback traffic, not an Internet upload. The server
accepts only loopback `Host` headers, applies a restrictive content security
policy, and sends only GPS coordinates to Nominatim.
Cached place names and generated text remain in that browser profile until
cleared; they can be sensitive in the same way as the generated story.

Use a different port or prevent the automatic browser launch if needed:

```sh
uv run gemma-photo-story-web --port 9000 --no-open
```

### Command-line app

```sh
uv run gemma-photo-story \
  /Users/haldar/Downloads/testpics \
  --output-dir output/testpics
```

Generated files:

- `analysis.json`: EXIF metadata, reverse-geocoded place data, and Gemma's
  evidence-grounded description of every image.
- `story.md`: the final narrative.
- `geocode-cache.json`: cached Nominatim results, preventing duplicate remote
  lookups on reruns.

During a run, the CLI also prints each reverse-geocode result and structured
Gemma image description as readable JSON. Every image-description line and the
final narrative-writing line identify the exact Ollama model being used.

The default model is `gemma4:26b`. Override it with `--model` if another local
Ollama vision model is installed.

## Network boundary

The only allowed network destinations are:

- `http://127.0.0.1:11434` or another loopback hostname for local Ollama.
- `https://nominatim.openstreetmap.org/reverse` and `/search` for reverse
  geocoding and exact-address place-name enrichment.

Proxy environment variables are ignored by the HTTP client. The public
client also rejects HTTP redirects, preventing an allowlisted request from
being redirected to another destination. The public
Nominatim service is limited to light use, requires attribution and an
identifying user agent, and must not receive bulk or systematic requests. This
script performs uncached requests sequentially at no more than one per second.

For a production application, use a replaceable geocoder with explicit user
consent, durable caching, and either a suitable hosted plan or a self-hosted
Nominatim instance. Exact GPS coordinates and generated output can be sensitive.

Map data © OpenStreetMap contributors, available under the ODbL.

## Tests

```sh
uv run python -m unittest discover -s tests -v
```

The tests exercise the strict network allowlist, model-output parsing, stable
geocoder-field extraction, image discovery, upload validation, and a real
loopback HTTP session without mocking external APIs.
