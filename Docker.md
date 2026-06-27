# Running `audiocpp_server` in Docker

This runbook runs the current `audiocpp_server` binary inside a CUDA
Docker runtime image. The repository is mounted read-only into the container, so
the server uses the exact local `build/` binary and local model assets.

The commands below assume your normal CMake/CUDA build environment is already
active and Docker is configured with NVIDIA Container Toolkit.

## 0. Verify Docker GPU Access

Check Docker GPU access before building or starting the server:

```bash
docker run --rm --gpus all nvidia/cuda:12.9.0-runtime-ubuntu24.04 nvidia-smi
```

If this fails with `could not select device driver "" with capabilities:
[[gpu]]`, Docker is not configured with the NVIDIA container runtime yet.
Configure NVIDIA Container Toolkit first, then rerun the same command.

## 1. Build the Server

Use the same CUDA build tree as the README:

```bash
cmake \
  -S . \
  -B build \
  -DENGINE_ENABLE_CUDA=ON

cmake \
  --build build \
  --parallel "$(nproc)" \
  --target audiocpp_server
```

The server binary should exist at:

```bash
build/bin/audiocpp_server
```

## 2. Create a Smoke-Test Config

This config uses the checked-in Silero VAD asset so the server can be tested
without downloading a large model:

```bash
mkdir -p build/docker-server-smoke

cat > build/docker-server-smoke/server.json <<'JSON'
{
  "host": "0.0.0.0",
  "port": 8080,
  "device": 0,
  "threads": 1,
  "models": [
    {
      "id": "silero-vad",
      "family": "silero_vad",
      "path": "/workspace/assets/framework/models/silero_vad",
      "task": "vad",
      "mode": "offline"
    }
  ]
}
JSON
```

`host` must be `0.0.0.0` inside Docker. Binding to `127.0.0.1` would make the
server listen only inside the container.

## 3. Build the Runtime Image

Build the small runtime image from stdin. This sends only the Dockerfile text to
Docker, not the repository, `build/`, or model directories:

```bash
docker build -t audiocpp-server-runtime:cuda12.9 - <<'DOCKERFILE'
FROM nvidia/cuda:12.9.0-runtime-ubuntu24.04
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates curl libgomp1 libvulkan1 \
 && rm -rf /var/lib/apt/lists/*
WORKDIR /workspace
ENTRYPOINT ["/workspace/build/bin/audiocpp_server"]
DOCKERFILE
```

## 4. Start the Server

Start the server with the standard NVIDIA Docker runtime path:

```bash
docker rm -f audiocpp-server-smoke >/dev/null 2>&1 || true

docker run -d --name audiocpp-server-smoke \
  --gpus all \
  -p 18080:8080 \
  -v "$PWD:/workspace:ro" \
  -w /workspace \
  audiocpp-server-runtime:cuda12.9 \
  --config /workspace/build/docker-server-smoke/server.json
```

Do not replace `--gpus all` with manual driver-library mounts in normal usage.

## 5. Wait for Health

The first request can arrive before model loading finishes, so use a retry loop:

```bash
for i in $(seq 1 30); do
  if curl -fsS http://127.0.0.1:18080/health; then
    echo
    exit 0
  fi
  sleep 1
done

echo "server did not become healthy" >&2
docker logs audiocpp-server-smoke >&2
exit 1
```

Expected response:

```json
{"status":"ok","backend":"cuda","models":1}
```

## 6. Verify the Registered Model

```bash
curl -fsS http://127.0.0.1:18080/v1/models
```

Expected response:

```json
{"object":"list","data":[{"id":"silero-vad","object":"model","owned_by":"engine","family":"silero_vad","task":"vad","mode":"offline"}]}
```

## 7. Run a Real Task Request

```bash
curl -fsS http://127.0.0.1:18080/v1/tasks/run \
  -H 'Content-Type: application/json' \
  -d '{"model":"silero-vad","request":{"audio":"/workspace/assets/resources/sample_16k.wav"}}'
```

Expected response shape:

```json
{"segments":[{"start_sample":6688,"end_sample":33760,"confidence":1},{"start_sample":47136,"end_sample":81376,"confidence":1},{"start_sample":94240,"end_sample":156640,"confidence":1},{"start_sample":170528,"end_sample":222176,"confidence":1}]}
```

This is the validated response for the checked-in `assets/resources/sample_16k.wav`
smoke input.

## 8. Stop the Smoke Server

```bash
docker rm -f audiocpp-server-smoke
```

## Using Other Models

For larger models, keep the same Docker image and server command, but change
`build/docker-server-smoke/server.json` to point at the model path as it appears
inside the container. For example, if the repo-local `models/` directory is
mounted through `/workspace`, use paths like:

```json
"path": "/workspace/models/Qwen3-ASR-0.6B"
```

If models live outside the repo, mount them explicitly and use that container
path in `server.json`, for example:

```bash
-v /absolute/host/models:/models:ro
```

```json
"path": "/models/Qwen3-ASR-0.6B"
```
