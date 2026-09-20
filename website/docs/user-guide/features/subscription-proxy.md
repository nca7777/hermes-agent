---
sidebar_position: 15
title: "Subscription Proxy"
description: "Use your Nous Portal subscription (or other OAuth provider) as an OpenAI-compatible endpoint for external apps"
---

# Subscription Proxy

The subscription proxy is a local HTTP server that lets external apps —
OpenViking, Karakeep, Open WebUI, anything that speaks OpenAI-compatible
chat completions — use your Hermes-managed provider subscription as their
LLM endpoint. The proxy attaches the right credentials (refreshing them
automatically) so the app never needs a static API key.

This is different from the [API server](./api-server.md):

| | API server | Subscription proxy |
|---|---|---|
| What it serves | Your agent (full toolset, memory, skills) | Raw model inference |
| Use case | "Use Hermes as a chat backend" | "Use my Portal sub from another app" |
| Auth | Your `API_SERVER_KEY` | `SUBSCRIPTION_PROXY_KEY` (optional on loopback; required for other binds) |
| Tool calls | Yes — the agent runs tools | Passed through unchanged; the calling app executes them |

Use the API server when you want the **agent** as a backend. Use the
proxy when you just want **the model** through your subscription.

## Quick Start

### 1. Log into your provider (one-time)

```bash
hermes portal
```

This opens your browser for the Nous Portal OAuth flow. Hermes stores
the refresh token in `~/.hermes/auth.json` — the same place all Hermes
provider logins live.

### 2. Start the proxy

```bash
export SUBSCRIPTION_PROXY_KEY="$(openssl rand -hex 32)"
hermes config set SUBSCRIPTION_PROXY_KEY "$SUBSCRIPTION_PROXY_KEY"
hermes proxy start
```

```
Starting Hermes proxy for Nous Portal
  Listening on:  http://127.0.0.1:8645/v1
  Forwarding to: (resolved per-request from your subscription)
  Downstream auth: required (SUBSCRIPTION_PROXY_KEY)
```

Leave this running in the foreground. Use `tmux`, `nohup`, or a systemd
unit if you want it to survive logout.

### 3. Point your app at it

Any OpenAI-compatible app config takes the same triple:

```
Base URL:   http://127.0.0.1:8645/v1
API key:    the exact value of SUBSCRIPTION_PROXY_KEY
Model:      Hermes-4-70B    # or Hermes-4.3-36B, Hermes-4-405B
```

On a loopback-only bind, `SUBSCRIPTION_PROXY_KEY` may be omitted. In that
mode the proxy accepts any downstream bearer. When the secret is configured,
the API key must exactly match it. Every non-loopback bind is refused unless
the key contains at least 32 non-whitespace characters. In either mode, the
downstream `Authorization` value is never forwarded: the proxy replaces it
with your provider credential. Refreshes happen automatically when the
upstream bearer approaches expiry.

Store the secret through Hermes' secret configuration flow:

```bash
hermes config set SUBSCRIPTION_PROXY_KEY "$SUBSCRIPTION_PROXY_KEY"
```

This writes it to the active Hermes profile's `.env`, not `config.yaml`.
Restart the proxy after changing it. The startup message reports only whether
authentication is required; it never prints the secret.

## Available providers

```bash
hermes proxy providers
```

Currently shipped:

| Provider | Upstream protocol | Login |
|---|---|---|
| `nous` | OpenAI Chat Completions | `hermes auth add nous` |
| `openai-codex` | OpenAI Responses | `hermes auth add openai-codex` |
| `xai` | OpenAI Chat Completions and Responses | `hermes auth add xai-oauth --type oauth` |

### Share the default profile's Codex subscription

Authenticate the **default** Hermes profile once, then start one loopback
proxy:

```bash
hermes auth add openai-codex
hermes proxy start --provider openai-codex
```

The Codex adapter always resolves and rotates credentials in the default
Hermes home's `openai-codex` OAuth pool, even if the command is launched while
a named profile is active. Named profiles never read that `auth.json`, and the
proxy never copies access or refresh tokens into them.

Codex uses the Responses API, not Chat Completions. Store the same downstream
key in each client profile, then reference the secret by name:

```bash
hermes -p work config set SUBSCRIPTION_PROXY_KEY "$SUBSCRIPTION_PROXY_KEY"
```

```yaml
providers:
  shared-codex:
    api: http://127.0.0.1:8645/v1
    key_env: SUBSCRIPTION_PROXY_KEY
    transport: codex_responses
    default_model: gpt-5.6-sol

model:
  provider: shared-codex
  default: gpt-5.6-sol
```

Requests go to `/v1/responses`. Tool definitions, function-call items, and SSE
events pass through unchanged. The proxy replaces the client bearer and Codex
identity/workspace headers with values derived from the selected default-home
OAuth credential.

## Check status

```bash
hermes proxy status
```

```
Hermes proxy upstream adapters

  [nous    ] Nous Portal — ready (bearer expires 2026-05-15T06:43:21Z)
```

If you see `not logged in`, run `hermes portal`. If you see
`credentials need attention`, your refresh token was revoked (rare —
happens if you signed out from the Portal web UI) — just re-run
`hermes portal`.

## Allowed paths

The proxy only forwards paths the selected upstream actually serves:

| Provider | Paths and methods |
|---|---|
| Nous Portal | `POST /v1/chat/completions`, `POST /v1/completions`, `POST /v1/embeddings`, `GET`/`HEAD /v1/models` |
| OpenAI Codex | `POST /v1/responses`, `GET`/`HEAD /v1/models` |
| xAI | `POST /v1/responses`, `POST /v1/chat/completions`, `POST /v1/completions`, `POST /v1/embeddings`, `GET`/`HEAD /v1/models` |

Other paths (`/v1/images/generations`, `/v1/audio/speech`, etc.) return
404 with a clear error pointing at the allowed paths. This keeps stray
clients from leaking weird requests to the upstream.

## Configuring OpenViking to use Portal

[OpenViking](https://github.com/volcengine/OpenViking) is a context
database that needs an LLM provider for its VLM (vision/language model
used to extract memories) and embedding model. With the proxy, you can
point its `vlm.api_base` at your local proxy:

Edit `~/.openviking/ov.conf`:

```json
{
  "vlm": {
    "provider": "openai",
    "model": "Hermes-4-70B",
    "api_base": "http://127.0.0.1:8645/v1",
    "api_key": "${SUBSCRIPTION_PROXY_KEY}"
  }
}
```

OpenViking expands environment variables in `ov.conf`. Export the same key,
then start your proxy in a terminal alongside `openviking-server`:

```bash
# Terminal 1
hermes proxy start

# Terminal 2 (with SUBSCRIPTION_PROXY_KEY injected by your secret manager)
openviking-server
```

OpenViking's VLM calls now flow through your Portal subscription. The
embedding model side still needs its own provider — Portal does serve
`/v1/embeddings` but the model selection depends on what your tier
supports; check `portal.nousresearch.com/models`.

## Configuring Karakeep (or any bookmark/summarizer app)

[Karakeep](https://karakeep.app/) takes an OpenAI-compatible API for
bookmark summarization. Pass the proxy key through the process environment:

```bash
export OPENAI_API_BASE_URL=http://127.0.0.1:8645/v1
export OPENAI_API_KEY="$SUBSCRIPTION_PROXY_KEY"
export INFERENCE_TEXT_MODEL=Hermes-4-70B
```

Same pattern works for Open WebUI, LobeChat, NextChat, or any other
OpenAI-compatible client.

## Docker and Coolify access

By default the proxy binds `127.0.0.1` (localhost only). A container that does
not share the host network cannot reach that loopback listener. Before binding
to an interface reachable by Docker, Coolify, or another machine, configure a
strong downstream secret:

```bash
export SUBSCRIPTION_PROXY_KEY="$(openssl rand -hex 32)"
hermes config set SUBSCRIPTION_PROXY_KEY "$SUBSCRIPTION_PROXY_KEY"
hermes proxy start --host 0.0.0.0 --port 8645
```

Set the consuming app's OpenAI API key to the exact same value. A missing or
wrong `Authorization: Bearer <SUBSCRIPTION_PROXY_KEY>` header receives `401`
before Hermes resolves an upstream credential or forwards a body.

`GET /health` intentionally remains unauthenticated so Docker and Coolify can
probe it; it reports proxy/upstream readiness and never forwards upstream.
`hermes proxy status` is also a local CLI operation and does not use the
downstream secret.

The bearer protects subscription use but does not encrypt traffic. Keep a
firewall around the listener and use a private Docker network, VPN, or TLS
reverse proxy when traffic can leave the host.

## Rate limits

Your provider's subscription limits apply across the whole proxy. Nous uses
its current inference bearer. Codex and xAI use their Hermes credential pools,
including established refresh, cooldown, and one-shot rotation behavior.
Monitor Nous usage at
[portal.nousresearch.com](https://portal.nousresearch.com); Codex usage follows
the ChatGPT account selected by the default profile's pool.

## Architecture

The proxy is intentionally minimal. Per request:

1. Authenticate the downstream bearer when `SUBSCRIPTION_PROXY_KEY` is set
2. Enforce the allowed path/method and the 10 MB request-body cap
3. Look up the adapter's current credential (refresh if expiring)
4. Forward the request body verbatim with the resolved authorization and
   provider-required headers
5. Stream the response back unchanged (SSE preserved)

No transformation. No logging of request bodies or bearer secrets. No agent
loop. The proxy is a credential-attaching pass-through.

## Future: more OAuth providers

The adapter system is pluggable. Adding a new provider (e.g.
HuggingFace, GitHub Copilot's chat endpoint, Anthropic via OAuth)
requires implementing `UpstreamAdapter` in
`hermes_cli/proxy/adapters/<provider>.py` and registering it in
`adapters/__init__.py`. Providers that aren't OpenAI-compatible at the
protocol level (Anthropic Messages API, for example) would need a
transformation layer, which is out of scope for the current shape.
