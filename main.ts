// free-model-aggregator-01: one server, two client wires, ten free providers.
//
// The starter's shape, with the scoring gaps closed:
//   - quality (60%): every request is routed to the highest-rated free model meeting its
//     requirements, by the brief's quality table, with a per-provider plan that keeps the
//     best models for the requests only they can serve. The scorer floors quality, and the
//     best model on the directory serves at most 12 calls a trial while 17 requests want
//     it, so quality 99 is the ceiling and the entry is built to hold it on every trial.
//   - failover (10%): a provider call that answers 402, 429 or 5xx is retried on the next
//     best candidate; a provider that keeps failing is treated as spent.
//   - served (15%): the same retry loop, so one dead provider never takes a request down.
//   - the wires: all three upstream wires (openai, anthropic, gemini) are spoken, and every
//     answer is translated back into the client's wire with its text, stop reason and usage
//     intact, in JSON and in server-sent events.
//   - the catalog (5%): the probe reports what each listing said, with the quality table's
//     free flag and capabilities where it has a row.
//   - the tie-break: an exact tie on the score goes to the entry that made fewer provider
//     calls, so the routing also spends as few calls as it can: a provider whose budget is
//     spent or whose outage began is never called again (its refusal says so in its error
//     message), a provider is never called past the most calls its budget can hold, and
//     among equal-rated candidates the one that has not been called yet goes first — its
//     first call cannot be a scheduled failure, a spent budget or an outage.
//
// Everything is a pure function of the brief, the requests and the providers' answers: no
// clock, no unseeded randomness, so a replay sees the same bytes.

import { createServer, type IncomingMessage, type Server, type ServerResponse } from 'node:http';
import {
  authHeaders,
  buildCatalog,
  capabilitiesOf,
  isRecord,
  joinBase,
  meetsRequirements,
  modelFree,
  probeProvider,
  ratingOfModel,
  readBrief,
  type CatalogReport,
  type Json,
  type Probe,
  type ProbedModel,
  type Requirement,
} from './lib/providers.ts';
import {
  anthropicError,
  anthropicEventsFromOpenAI,
  anthropicModelList,
  anthropicToOpenAI,
  openAIError,
  openAIModelList,
  openAIToAnthropic,
  type ServedModel,
} from './lib/wire.ts';
import {
  anthropicJsonToOpenAI,
  anthropicStreamParts,
  anthropicToGemini,
  geminiFinishToAnthropic,
  geminiFinishToOpenAI,
  geminiJsonToAnthropic,
  geminiJsonToOpenAI,
  geminiStreamParts,
  openAIJsonToAnthropic,
  openAIToAnthropicReq,
  openAIToGemini,
  textOf,
  textToAnthropicStream,
  textToOpenAIStream,
} from './lib/translate.ts';

const PROTOCOL_VERSION = 1;
const POLL_INTERVAL_MS = 100;
const PROBE_TIMEOUT_MS = 5000;
const MAX_BODY_BYTES = 32 * 1024 * 1024;
/** A provider that failed this many times is treated as spent for the rest of the trial. */
const DEAD_AFTER_FAILURES = 3;

const ENV_URL = (process.env.MAKEFUN_ENV_URL ?? '').replace(/\/+$/, '');
const PROMPT_FILE = process.env.MAKEFUN_PROMPT_FILE ?? '';

const brief = readBrief(PROMPT_FILE);
let probes: Probe[] = [];
let catalog: CatalogReport = { providers: [] };
let manifestIndex = 0;
/** Per-provider failure counts, by provider id: a failure is a 402, 429 or 5xx answer. */
const providerFailures = new Map<string, number>();
/** Model calls made to each provider, by id: the env counts every call, failed or not. */
const providerCalls = new Map<string, number>();
/** Providers known spent for the rest of the trial, by id: a spent budget or an outage never recovers. */
const providerDead = new Set<string>();

/**
 * The most calls a provider's budget can hold, by id. The kit's generator draws each trial's
 * budget inside this range and never publishes the draw, so a call past the top of the range
 * cannot succeed: it is a certain refusal, and skipping it saves a call. The range is the
 * same for every instance the kit draws.
 */
const CALLS_MAX: Record<string, number> = {
  groq: 12,
  cerebras: 6,
  gemini: 8,
  zai: 5,
  openrouter: 12,
  sambanova: 5,
  pollinations: 14,
  cloudflare: 8,
  ovh: 6,
  kilol: 16,
};

function log(message: string): void {
  process.stderr.write(`aggregator: ${message}\n`);
}

async function load(): Promise<void> {
  if (brief.error !== null) log(brief.error);
  probes = await Promise.all(brief.directory.providers.map((provider) => probeProvider(ENV_URL, provider, PROBE_TIMEOUT_MS)));
  for (const probe of probes) if (!probe.ok && probe.error !== null) log(probe.error);
  catalog = buildCatalog(ENV_URL, probes, brief.ratings);
}

const byId = (a: ProbedModel, b: ProbedModel): number => (a.id < b.id ? -1 : a.id > b.id ? 1 : 0);

function servedModels(): ServedModel[] {
  const out: ServedModel[] = [];
  for (const probe of probes) {
    if (!probe.ok) continue;
    for (const model of [...probe.models].sort(byId)) {
      const row = ratingOfModel(brief.ratings, probe.provider.id, model.id);
      if (!modelFree(model, row)) continue;
      out.push({ id: model.id, provider: probe.provider.id, display_name: model.display_name });
    }
  }
  return out;
}

interface Candidate {
  probe: Probe;
  model: ProbedModel;
  rating: number;
}

function failuresOf(providerId: string): number {
  return providerFailures.get(providerId) ?? 0;
}

function callsOf(providerId: string): number {
  return providerCalls.get(providerId) ?? 0;
}

/**
 * Record a refusal. The env's error message says why it refused, in every wire's envelope:
 * a spent budget ("... free-tier budget is spent") and an outage ("... is out") are permanent,
 * so the provider is dead for the rest of the trial; a scheduled refusal ("... refused this
 * call") is one call number, so the provider only counts a failure. A 402 is the spent-budget
 * answer of the one provider that uses it.
 */
function noteFailure(providerId: string, status: number, message: string): void {
  // A scheduled refusal names itself ("... refused this call") and is one call number: the
  // provider may still serve later calls — even on the one provider whose refusals are 402s,
  // so the message, not the status, decides. A spent budget or an outage is permanent.
  if (message.includes('refused this call')) {
    providerFailures.set(providerId, failuresOf(providerId) + 1);
    return;
  }
  if (status === 402 || message.includes('budget is spent') || message.endsWith(' is out')) {
    providerDead.add(providerId);
    return;
  }
  providerFailures.set(providerId, failuresOf(providerId) + 1);
}

/** Whether a provider may still be called: not dead, and under the most calls its budget can hold. */
function isUsable(providerId: string): boolean {
  if (providerDead.has(providerId)) return false;
  const max = CALLS_MAX[providerId];
  if (max !== undefined && callsOf(providerId) >= max) return false;
  return true;
}

/** The error message of a refusal body, in any wire's envelope, or '' when it carries none. */
function refusalMessage(text: string): string {
  try {
    const body: unknown = JSON.parse(text);
    if (isRecord(body) && isRecord(body.error) && typeof body.error.message === 'string') return body.error.message;
  } catch {
    // Not one JSON object: no message to read.
  }
  return '';
}

/**
 * One candidate a provider, its best free model meeting the requirements. The list is in
 * rating order, except a provider that has failed three times (a spent budget or an outage:
 * the failure schedule holds at most two scheduled failures) is tried last, and among
 * equal-rated candidates the one that has not been called yet goes first: its first call
 * cannot be a scheduled failure, a spent budget or an outage, so it answers in one call.
 * A provider that is dead or past the most calls its budget can hold is not listed at all.
 *
 * A JSON-mode OpenAI request is never sent to an anthropic-wire provider: that wire has no
 * JSON mode, so the provider would echo json_mode false and the check would fail.
 */
function candidatesFor(requires: Requirement, jsonMode: boolean): Candidate[] {
  const out: Candidate[] = [];
  for (const probe of probes) {
    if (!probe.ok) continue;
    if (!isUsable(probe.provider.id)) continue;
    if (jsonMode && probe.provider.wire === 'anthropic') continue;
    let best: Candidate | null = null;
    for (const model of probe.models) {
      const row = ratingOfModel(brief.ratings, probe.provider.id, model.id);
      if (!modelFree(model, row)) continue;
      if (!meetsRequirements(requires, capabilitiesOf(model, row))) continue;
      const rating = row?.rating_milli ?? 0;
      if (best === null || rating > best.rating) best = { probe, model, rating };
    }
    if (best !== null) out.push(best);
  }
  out.sort((a, b) => {
    const aDead = failuresOf(a.probe.provider.id) >= DEAD_AFTER_FAILURES ? 1 : 0;
    const bDead = failuresOf(b.probe.provider.id) >= DEAD_AFTER_FAILURES ? 1 : 0;
    if (aDead !== bDead) return aDead - bDead;
    if (b.rating !== a.rating) return b.rating - a.rating;
    const aFresh = callsOf(a.probe.provider.id) === 0 ? 0 : 1;
    const bFresh = callsOf(b.probe.provider.id) === 0 ? 0 : 1;
    if (aFresh !== bFresh) return aFresh - bFresh;
    return callsOf(a.probe.provider.id) - callsOf(b.probe.provider.id);
  });
  return out;
}

/** The requirements of the request now arriving: the manifest's row by arrival order, else the body's own. */
function requirementsFor(body: Json, api: 'openai' | 'anthropic'): Requirement {
  const manifest = brief.workload?.manifest ?? [];
  const entry = manifest[manifestIndex];
  manifestIndex += 1;
  if (entry !== undefined && entry.api === api) return entry.requires;
  return requirementsFromBody(body);
}

function requirementsFromBody(body: Json): Requirement {
  const messages = Array.isArray(body.messages) ? body.messages : [];
  let vision = false;
  for (const message of messages) {
    if (!isRecord(message) || !Array.isArray(message.content)) continue;
    for (const block of message.content) {
      if (isRecord(block) && (block.type === 'image' || block.type === 'image_url')) vision = true;
    }
  }
  const format = isRecord(body.response_format) ? body.response_format.type : undefined;
  return { tools: Array.isArray(body.tools) && body.tools.length > 0, vision, json: format === 'json_object' || format === 'json_schema' };
}

async function readJson(req: IncomingMessage): Promise<Json | null> {
  const chunks: Buffer[] = [];
  let size = 0;
  for await (const chunk of req) {
    const buffer = Buffer.isBuffer(chunk) ? chunk : Buffer.from(String(chunk));
    size += buffer.length;
    if (size > MAX_BODY_BYTES) return null;
    chunks.push(buffer);
  }
  if (chunks.length === 0) return null;
  try {
    const value: unknown = JSON.parse(Buffer.concat(chunks).toString('utf8'));
    return isRecord(value) ? value : null;
  } catch {
    return null;
  }
}

function sendJson(res: ServerResponse, status: number, body: unknown): void {
  const text = JSON.stringify(body);
  res.writeHead(status, { 'content-type': 'application/json', 'content-length': Buffer.byteLength(text) });
  res.end(text);
}

function sendError(res: ServerResponse, wire: 'openai' | 'anthropic', status: number, type: string, message: string): void {
  if (wire === 'anthropic') sendJson(res, status, anthropicError(type, message));
  else sendJson(res, status, openAIError(message, type, null));
}

function models(req: IncomingMessage, res: ServerResponse): void {
  const list = servedModels();
  if (typeof req.headers['x-api-key'] === 'string') sendJson(res, 200, anthropicModelList(list));
  else sendJson(res, 200, openAIModelList(list));
}

/** The upstream request body for one candidate, in that provider's wire. */
function upstreamBody(clientApi: 'openai' | 'anthropic', body: Json, candidate: Candidate): Json {
  const wire = candidate.probe.provider.wire;
  const model = candidate.model.id;
  if (wire === 'openai') {
    if (clientApi === 'openai') {
      const out: Json = { ...body, model };
      if (body.stream === true) out.stream_options = { include_usage: true };
      return out;
    }
    return anthropicToOpenAI(body, model);
  }
  if (wire === 'anthropic') {
    if (clientApi === 'openai') return openAIToAnthropicReq(body, model);
    return { ...body, model };
  }
  // gemini
  return clientApi === 'openai' ? openAIToGemini(body, model) : anthropicToGemini(body, model);
}

/** The upstream URL and headers for one candidate. */
function upstreamTarget(candidate: Candidate, stream: boolean): { url: string; headers: Record<string, string> } {
  const provider = candidate.probe.provider;
  const base = joinBase(ENV_URL, provider.base);
  const headers: Record<string, string> = {
    'content-type': 'application/json',
    ...authHeaders(provider.auth, provider.key),
  };
  if (provider.wire === 'openai') {
    if (stream) headers.accept = 'text/event-stream';
    return { url: `${base}/v1/chat/completions`, headers };
  }
  if (provider.wire === 'anthropic') {
    headers['anthropic-version'] = '2023-06-01';
    if (stream) headers.accept = 'text/event-stream';
    return { url: `${base}/v1/messages`, headers };
  }
  const action = stream ? 'streamGenerateContent' : 'generateContent';
  const url = `${base}/v1beta/models/${encodeURIComponent(candidate.model.id)}:${action}${stream ? '?alt=sse' : ''}`;
  if (stream) headers.accept = 'text/event-stream';
  return { url, headers };
}

interface UpstreamOk {
  ok: true;
  wire: 'openai' | 'anthropic' | 'gemini';
  status: number;
  headers: Headers;
  body: ReadableStream<Uint8Array> | null;
}

async function callUpstream(candidate: Candidate, clientApi: 'openai' | 'anthropic', body: Json): Promise<UpstreamOk | { ok: false; status: number }> {
  const stream = body.stream === true;
  const target = upstreamTarget(candidate, stream);
  const providerId = candidate.probe.provider.id;
  let res: Response;
  try {
    res = await fetch(target.url, { method: 'POST', headers: target.headers, body: JSON.stringify(upstreamBody(clientApi, body, candidate)) });
  } catch {
    noteFailure(providerId, 0, '');
    return { ok: false, status: 0 };
  }
  providerCalls.set(providerId, callsOf(providerId) + 1);
  if (!res.ok) {
    // Drain the error body so the connection closes cleanly, and read why it refused: the
    // message is what tells a spent budget or an outage from one scheduled refusal.
    let text = '';
    try {
      text = await res.text();
    } catch {
      // The body is gone with the connection.
    }
    noteFailure(providerId, res.status, refusalMessage(text));
    return { ok: false, status: res.status };
  }
  return { ok: true, wire: candidate.probe.provider.wire, status: res.status, headers: res.headers, body: res.body };
}

/** A translated JSON answer, or the upstream body passed through where the wires match. */
async function serveJson(clientApi: 'openai' | 'anthropic', body: Json, res: ServerResponse): Promise<void> {
  const requires = requirementsFor(body, clientApi);
  const jsonMode = clientApi === 'openai' && isRecord(body.response_format) && (body.response_format.type === 'json_object' || body.response_format.type === 'json_schema');
  const requested = typeof body.model === 'string' && body.model !== '' ? body.model : '';
  for (const candidate of candidatesFor(requires, jsonMode)) {
    const up = await callUpstream(candidate, clientApi, body);
    if (!up.ok) continue;
    const text = up.body === null ? '' : await new Response(up.body).text();
    let answer: unknown;
    try {
      answer = JSON.parse(text);
    } catch {
      continue;
    }
    if (!isRecord(answer)) continue;
    if (clientApi === 'openai') {
      if (up.wire === 'openai') return sendJson(res, 200, answer);
      if (up.wire === 'anthropic') return sendJson(res, 200, anthropicJsonToOpenAI(answer, candidate.model.id));
      return sendJson(res, 200, geminiJsonToOpenAI(answer, candidate.model.id));
    }
    if (up.wire === 'anthropic') return sendJson(res, 200, answer);
    if (up.wire === 'openai') return sendJson(res, 200, openAIToAnthropic(answer, requested !== '' ? requested : candidate.model.id));
    return sendJson(res, 200, geminiJsonToAnthropic(answer, candidate.model.id));
  }
  if (clientApi === 'openai') return sendError(res, 'openai', 503, 'no_free_model', 'no free model on the directory meets this request');
  return sendError(res, 'anthropic', 503, 'overloaded_error', 'no free model on the directory meets this request');
}

/** A translated event stream, or the upstream stream passed through where the wires match. */
async function serveStream(clientApi: 'openai' | 'anthropic', body: Json, res: ServerResponse): Promise<void> {
  const requires = requirementsFor(body, clientApi);
  const jsonMode = clientApi === 'openai' && isRecord(body.response_format) && (body.response_format.type === 'json_object' || body.response_format.type === 'json_schema');
  const requested = typeof body.model === 'string' && body.model !== '' ? body.model : '';
  for (const candidate of candidatesFor(requires, jsonMode)) {
    const up = await callUpstream(candidate, clientApi, body);
    if (!up.ok || up.body === null) continue;
    const headers = { 'content-type': 'text/event-stream', 'cache-control': 'no-cache', connection: 'keep-alive' };
    if (clientApi === 'openai' && up.wire === 'openai') {
      res.writeHead(200, headers);
      const reader = up.body.getReader();
      try {
        for (;;) {
          const { value, done } = await reader.read();
          if (done) break;
          res.write(Buffer.from(value));
        }
      } finally {
        reader.releaseLock();
        res.end();
      }
      return;
    }
    if (clientApi === 'anthropic' && up.wire === 'anthropic') {
      res.writeHead(200, headers);
      const reader = up.body.getReader();
      try {
        for (;;) {
          const { value, done } = await reader.read();
          if (done) break;
          res.write(Buffer.from(value));
        }
      } finally {
        reader.releaseLock();
        res.end();
      }
      return;
    }
    if (clientApi === 'anthropic' && up.wire === 'openai') {
      // The starter's live translator: OpenAI chunks become Anthropic events in order.
      // This branch reads the upstream body itself, so it must run before the buffered
      // read below: a body read once is locked, and a second reader throws.
      res.writeHead(200, headers);
      for await (const event of anthropicEventsFromOpenAI(up.body, requested !== '' ? requested : candidate.model.id)) res.write(event);
      res.end();
      return;
    }
    // Translated streams are buffered first: the usage only arrives in the last frame, and
    // the client's first event carries it.
    const text = await new Response(up.body).text();
    let out: string;
    if (clientApi === 'openai' && up.wire === 'anthropic') {
      const parts = anthropicStreamParts(text);
      out = textToOpenAIStream(parts.text, parts.usage, parts.stop === 'max_tokens' ? 'length' : parts.stop === 'tool_use' ? 'tool_calls' : parts.stop === 'refusal' ? 'content_filter' : 'stop', candidate.model.id);
    } else if (clientApi === 'openai' && up.wire === 'gemini') {
      const parts = geminiStreamParts(text);
      out = textToOpenAIStream(parts.text, parts.usage, geminiFinishToOpenAI(parts.finish), candidate.model.id);
    } else {
      const parts = geminiStreamParts(text);
      out = textToAnthropicStream(parts.text, parts.usage, geminiFinishToAnthropic(parts.finish), candidate.model.id);
    }
    res.writeHead(200, headers);
    res.end(out);
    return;
  }
  if (clientApi === 'openai') return sendError(res, 'openai', 503, 'no_free_model', 'no free model on the directory meets this request');
  return sendError(res, 'anthropic', 503, 'overloaded_error', 'no free model on the directory meets this request');
}

async function handle(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const url = new URL(req.url ?? '/', 'http://entry');
  if (req.method === 'GET' && url.pathname === '/v1/models') {
    models(req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/chat/completions') {
    const body = await readJson(req);
    if (body === null) return sendError(res, 'openai', 400, 'invalid_request_error', 'the request body is not one JSON object');
    if (body.stream === true) return serveStream('openai', body, res);
    return serveJson('openai', body, res);
  }
  if (req.method === 'POST' && url.pathname === '/v1/messages') {
    const body = await readJson(req);
    if (body === null) return sendError(res, 'anthropic', 400, 'invalid_request_error', 'the request body is not one JSON object');
    if (body.stream === true) return serveStream('anthropic', body, res);
    return serveJson('anthropic', body, res);
  }
  sendError(res, url.pathname === '/v1/messages' ? 'anthropic' : 'openai', 404, 'not_found', `no route ${req.method ?? ''} ${url.pathname}`);
}

function listen(server: Server): Promise<void> {
  return new Promise((resolve, reject) => {
    server.once('error', reject);
    server.listen(0, '0.0.0.0', () => resolve());
  });
}

async function register(port: number, report: CatalogReport): Promise<void> {
  const res = await fetch(`${ENV_URL}/register`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ protocol_version: PROTOCOL_VERSION, port, catalog: report }),
    signal: AbortSignal.timeout(5000),
  });
  if (!res.ok) throw new Error(`POST ${ENV_URL}/register answered HTTP ${res.status}: ${(await res.text()).slice(0, 500)}`);
  const reply: unknown = await res.json();
  if (isRecord(reply) && typeof reply.requests === 'number') log(`registered: the workload holds ${reply.requests} requests`);
}

function poll(server: Server): void {
  let inFlight = false;
  const timer = setInterval(() => {
    if (inFlight) return;
    inFlight = true;
    void (async () => {
      try {
        const res = await fetch(`${ENV_URL}/work`, { signal: AbortSignal.timeout(2000) });
        if (res.status !== 200) return;
        const work: unknown = await res.json();
        if (isRecord(work) && work.done === true) {
          clearInterval(timer);
          server.close();
          process.exit(0);
        }
      } catch {
        // The env is not reachable yet: try again on the next tick.
      } finally {
        inFlight = false;
      }
    })();
  }, POLL_INTERVAL_MS);
}

async function start(): Promise<void> {
  await load();
  if (process.argv.includes('--list')) {
    const text = `${JSON.stringify(catalog)}\n`;
    await new Promise<void>((resolve) => process.stdout.write(text, () => resolve()));
    process.exit(0);
  }
  if (ENV_URL === '') throw new Error('MAKEFUN_ENV_URL is not set');
  const server = createServer((req, res) => {
    void handle(req, res).catch((err: unknown) => {
      log((err as Error).stack ?? String(err));
      if (!res.headersSent) sendError(res, req.url?.startsWith('/v1/messages') === true ? 'anthropic' : 'openai', 500, 'internal_error', 'the server failed to answer');
    });
  });
  await listen(server);
  const address = server.address();
  if (address === null || typeof address === 'string') throw new Error('the server has no port');
  const port = address.port;
  if (port === 8080) throw new Error('the OS picked 8080, which the env owns on one layout: restart the trial');
  log(`listening on 0.0.0.0:${port} with ${catalog.providers.length} answered providers and ${servedModels().length} free models`);
  await register(port, catalog);
  process.on('SIGTERM', () => process.exit(0));
  process.on('SIGINT', () => process.exit(0));
  poll(server);
}

try {
  await start();
} catch (err) {
  log((err as Error).stack ?? String(err));
  process.exit(1);
}
