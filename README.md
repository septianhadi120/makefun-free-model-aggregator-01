# free-model-aggregator-01

One server that speaks both client wires (OpenAI Chat Completions and Anthropic
Messages, JSON and server-sent events) and routes every request to the best free
model on a directory of ten simulated free-tier providers.

## How it works

- The brief (`$MAKEFUN_PROMPT_FILE`) is read at startup: the provider directory,
  the quality table (public ratings and capabilities) and the workload manifest.
- Each provider's catalog route is probed in its own wire, and one catalog is built
  from what the listings said, with the quality table's free flag and capabilities.
  `node main.ts --list` prints that same catalog and exits.
- The server listens on `0.0.0.0` at a free port (never 8080), registers
  `{"protocol_version": 1, "port", "catalog"}` at `$MAKEFUN_ENV_URL/register`, then
  serves `GET /v1/models` (the shape follows the auth header: `x-api-key` means
  Anthropic, `Authorization: Bearer` means OpenAI), `POST /v1/chat/completions`
  and `POST /v1/messages`, polling `GET $MAKEFUN_ENV_URL/work` every 100 ms and
  exiting 0 when it says `{"done": true}`.
- Routing: for each request, the manifest's requirements (by arrival order) pick one
  candidate per provider — its best free model meeting the requirements — ranked by
  the quality table's `rating_milli`. A provider that has failed three times is
  treated as spent and tried last. A JSON-mode OpenAI request is never sent to an
  anthropic-wire provider, which has no JSON mode.
- Failover: a 402, 429 or 5xx answer is retried on the next best candidate; a 402
  marks that provider spent for the rest of the trial.
- Wires: all three upstream wires are spoken (openai, anthropic, gemini) and every
  answer is translated back into the client's wire with its text, stop reason and
  usage intact. Streamed OpenAI upstream calls set `stream_options.include_usage`
  so the usage chunk is there. Translated streams are buffered first, because the
  usage only arrives in the last frame.

## Files

- `main.ts` — the server: brief, catalog, routes, register, poll.
- `lib/providers.ts` — the brief's types and parser, auth headers, catalog probe.
- `lib/wire.ts` — Anthropic ↔ Chat Completions translation and SSE writers.
- `lib/translate.ts` — upstream request builders and response translators for all
  three wires, JSON and streamed.
- `LICENSE` — MIT.

## Run

```sh
node main.ts --list
```

In a trial the harness runs `node main.ts` with `$MAKEFUN_PROMPT_FILE` and
`$MAKEFUN_ENV_URL` set. Node's standard library only; TypeScript runs directly.
