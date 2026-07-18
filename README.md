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
geocoder-field extraction, and image discovery without mocking external APIs.
