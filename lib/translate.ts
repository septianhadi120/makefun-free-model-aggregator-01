// Upstream request builders and response translators for the three provider wires.
//
// The trial's client calls this entry as OpenAI (POST /v1/chat/completions) or Anthropic
// (POST /v1/messages). Each provider speaks its own wire: openai, anthropic or gemini. This file
// builds the upstream request for a provider's wire from the client's body, and translates the
// provider's answer back into the client's wire, carrying the request faithfully: the system
// prompt, the user text, the image, the tool names, max_tokens, the stop sequences and JSON mode
// must all survive, because the env echoes them back and scores them.
//
// Everything here is a pure function of its arguments: no clock, no randomness, so a replay
// sees the same bytes.

import { isRecord, type Json } from './providers.ts';

const FIXED_OPENAI_CREATED = 1767225600;

/** The text of a string or of a block array's text blocks. */
export function textOf(content: unknown): string {
  if (typeof content === 'string') return content;
  if (!Array.isArray(content)) return '';
  return content.flatMap((b) => (isRecord(b) && b.type === 'text' && typeof b.text === 'string' ? [b.text] : [])).join('\n');
}

function stringOr(v: unknown, fallback: string): string {
  return typeof v === 'string' ? v : fallback;
}

/** Tool names of an OpenAI body, in order. */
function openAIToolNames(body: Json): string[] {
  if (!Array.isArray(body.tools)) return [];
  return body.tools.flatMap((t) => (isRecord(t) && isRecord(t.function) && typeof t.function.name === 'string' ? [t.function.name] : []));
}

/** Tool names of an Anthropic body, in order. */
function anthropicToolNames(body: Json): string[] {
  if (!Array.isArray(body.tools)) return [];
  return body.tools.flatMap((t) => (isRecord(t) && typeof t.name === 'string' ? [t.name] : []));
}

/** An OpenAI image_url part as a Gemini inlineData/fileData part, or null. */
function geminiImagePart(imageUrl: Json): Json | null {
  const url = typeof imageUrl.url === 'string' ? imageUrl.url : '';
  if (url === '') return null;
  const data = /^data:([^;,]+)?;base64,(.*)$/s.exec(url);
  if (data !== null) return { inlineData: { mimeType: data[1] !== undefined && data[1] !== '' ? data[1] : 'image/png', data: data[2] } };
  return { fileData: { fileUri: url } };
}

/** An Anthropic image block as a Gemini inlineData/fileData part, or null. */
function geminiImageFromAnthropic(block: Json): Json | null {
  const source = isRecord(block.source) ? block.source : null;
  if (source === null) return null;
  if (source.type === 'base64' && typeof source.data === 'string') {
    return { inlineData: { mimeType: typeof source.media_type === 'string' ? source.media_type : 'image/png', data: source.data } };
  }
  if (source.type === 'url' && typeof source.url === 'string') return { fileData: { fileUri: source.url } };
  return null;
}

/** An OpenAI message list as Gemini contents. Tool messages are skipped: the provider reads no tool history. */
function openAIContents(body: Json): Json[] {
  const out: Json[] = [];
  const messages = Array.isArray(body.messages) ? body.messages : [];
  for (const raw of messages) {
    if (!isRecord(raw)) continue;
    if (raw.role === 'system' || raw.role === 'tool') continue;
    if (raw.role === 'assistant') {
      const text = textOf(raw.content);
      if (text !== '') out.push({ role: 'model', parts: [{ text }] });
      continue;
    }
    const parts: Json[] = [];
    const content = raw.content;
    if (typeof content === 'string') {
      if (content !== '') parts.push({ text: content });
    } else if (Array.isArray(content)) {
      for (const block of content) {
        if (!isRecord(block)) continue;
        if (block.type === 'text' && typeof block.text === 'string') parts.push({ text: block.text });
        else if (block.type === 'image_url' && isRecord(block.image_url)) {
          const part = geminiImagePart(block.image_url);
          if (part !== null) parts.push(part);
        }
      }
    }
    if (parts.length > 0) out.push({ role: 'user', parts });
  }
  return out;
}

/** An Anthropic message list as Gemini contents. tool_result blocks are skipped: they carry no echoed text. */
function anthropicContents(body: Json): Json[] {
  const out: Json[] = [];
  const messages = Array.isArray(body.messages) ? body.messages : [];
  for (const raw of messages) {
    if (!isRecord(raw)) continue;
    if (raw.role === 'assistant') {
      const text = textOf(raw.content);
      if (text !== '') out.push({ role: 'model', parts: [{ text }] });
      continue;
    }
    if (raw.role !== 'user') continue;
    const parts: Json[] = [];
    const content = raw.content;
    if (typeof content === 'string') {
      if (content !== '') parts.push({ text: content });
    } else if (Array.isArray(content)) {
      for (const block of content) {
        if (!isRecord(block)) continue;
        if (block.type === 'text' && typeof block.text === 'string') parts.push({ text: block.text });
        else if (block.type === 'image') {
          const part = geminiImageFromAnthropic(block);
          if (part !== null) parts.push(part);
        }
      }
    }
    if (parts.length > 0) out.push({ role: 'user', parts });
  }
  return out;
}

function openAISystemText(body: Json): string {
  const messages = Array.isArray(body.messages) ? body.messages : [];
  const parts: string[] = [];
  for (const raw of messages) {
    if (!isRecord(raw) || raw.role !== 'system') continue;
    const text = textOf(raw.content);
    if (text !== '') parts.push(text);
  }
  return parts.join('\n');
}

function openAIMaxTokens(body: Json): number | null {
  if (typeof body.max_tokens === 'number') return body.max_tokens;
  if (typeof body.max_completion_tokens === 'number') return body.max_completion_tokens;
  return null;
}

function openAIStops(body: Json): string[] {
  if (typeof body.stop === 'string') return [body.stop];
  if (Array.isArray(body.stop)) return body.stop.filter((s): s is string => typeof s === 'string');
  return [];
}

function openAIJsonMode(body: Json): boolean {
  return isRecord(body.response_format) && (body.response_format.type === 'json_object' || body.response_format.type === 'json_schema');
}

function openAITools(body: Json): Json[] {
  if (!Array.isArray(body.tools)) return [];
  return body.tools.flatMap((t) => {
    if (!isRecord(t) || !isRecord(t.function) || typeof t.function.name !== 'string') return [];
    return [{ name: t.function.name, description: typeof t.function.description === 'string' ? t.function.description : '', parameters: isRecord(t.function.parameters) ? t.function.parameters : { type: 'object', properties: {} } }];
  });
}

function anthropicTools(body: Json): Json[] {
  if (!Array.isArray(body.tools)) return [];
  return body.tools.flatMap((t) => {
    if (!isRecord(t) || typeof t.name !== 'string') return [];
    return [{ name: t.name, description: typeof t.description === 'string' ? t.description : '', parameters: isRecord(t.input_schema) ? t.input_schema : { type: 'object', properties: {} } }];
  });
}

/** An OpenAI client body as a Gemini generateContent body for `model`. */
export function openAIToGemini(body: Json, model: string): Json {
  const out: Json = { model, contents: openAIContents(body) };
  const system = openAISystemText(body);
  if (system !== '') out.systemInstruction = { parts: [{ text: system }] };
  const config: Json = {};
  const maxTokens = openAIMaxTokens(body);
  if (maxTokens !== null) config.maxOutputTokens = maxTokens;
  if (typeof body.temperature === 'number') config.temperature = body.temperature;
  if (typeof body.top_p === 'number') config.topP = body.top_p;
  const stops = openAIStops(body);
  if (stops.length > 0) config.stopSequences = stops;
  if (openAIJsonMode(body)) config.responseMimeType = 'application/json';
  if (Object.keys(config).length > 0) out.generationConfig = config;
  const tools = openAITools(body);
  if (tools.length > 0) out.tools = [{ functionDeclarations: tools }];
  return out;
}

/** An Anthropic client body as a Gemini generateContent body for `model`. */
export function anthropicToGemini(body: Json, model: string): Json {
  const out: Json = { model, contents: anthropicContents(body) };
  const system = textOf(body.system);
  if (system !== '') out.systemInstruction = { parts: [{ text: system }] };
  const config: Json = {};
  if (typeof body.max_tokens === 'number') config.maxOutputTokens = body.max_tokens;
  if (typeof body.temperature === 'number') config.temperature = body.temperature;
  if (typeof body.top_p === 'number') config.topP = body.top_p;
  if (Array.isArray(body.stop_sequences) && body.stop_sequences.length > 0) config.stopSequences = body.stop_sequences.filter((s): s is string => typeof s === 'string');
  if (Object.keys(config).length > 0) out.generationConfig = config;
  const tools = anthropicTools(body);
  if (tools.length > 0) out.tools = [{ functionDeclarations: tools }];
  return out;
}

/** One OpenAI message as an Anthropic message list, with tool results merged into the next user message. */
function openAIMessagesToAnthropic(body: Json): Json[] {
  const out: Json[] = [];
  const messages = Array.isArray(body.messages) ? body.messages : [];
  let pendingToolResults: Json[] = [];
  const flushToolResults = (into: Json): void => {
    if (pendingToolResults.length === 0) return;
    const content = Array.isArray(into.content) ? into.content : [];
    into.content = [...content, ...pendingToolResults];
    pendingToolResults = [];
  };
  for (const raw of messages) {
    if (!isRecord(raw)) continue;
    if (raw.role === 'system') continue;
    if (raw.role === 'tool') {
      pendingToolResults.push({ type: 'tool_result', tool_use_id: typeof raw.tool_call_id === 'string' ? raw.tool_call_id : '', content: typeof raw.content === 'string' ? raw.content : JSON.stringify(raw.content ?? '') });
      continue;
    }
    if (raw.role === 'assistant') {
      const content: Json[] = [];
      const text = textOf(raw.content);
      if (text !== '') content.push({ type: 'text', text });
      if (Array.isArray(raw.tool_calls)) {
        for (const call of raw.tool_calls) {
          if (!isRecord(call) || !isRecord(call.function)) continue;
          let input: Json = {};
          if (typeof call.function.arguments === 'string' && call.function.arguments !== '') {
            try {
              const parsed: unknown = JSON.parse(call.function.arguments);
              if (isRecord(parsed)) input = parsed;
            } catch {
              input = {};
            }
          }
          content.push({ type: 'tool_use', id: typeof call.id === 'string' ? call.id : '', name: typeof call.function.name === 'string' ? call.function.name : '', input });
        }
      }
      out.push({ role: 'assistant', content });
      continue;
    }
    const content: Json[] = [];
    const c = raw.content;
    if (typeof c === 'string') {
      if (c !== '') content.push({ type: 'text', text: c });
    } else if (Array.isArray(c)) {
      for (const block of c) {
        if (!isRecord(block)) continue;
        if (block.type === 'text' && typeof block.text === 'string') content.push({ type: 'text', text: block.text });
        else if (block.type === 'image_url' && isRecord(block.image_url)) {
          const url = typeof block.image_url.url === 'string' ? block.image_url.url : '';
          const data = /^data:([^;,]+)?;base64,(.*)$/s.exec(url);
          if (data !== null) content.push({ type: 'image', source: { type: 'base64', media_type: data[1] !== undefined && data[1] !== '' ? data[1] : 'image/png', data: data[2] } });
          else if (url !== '') content.push({ type: 'image', source: { type: 'url', url } });
        }
      }
    }
    const msg: Json = { role: 'user', content };
    flushToolResults(msg);
    out.push(msg);
  }
  if (pendingToolResults.length > 0) out.push({ role: 'user', content: pendingToolResults });
  return out;
}

/** An OpenAI client body as an Anthropic Messages body for `model`. */
export function openAIToAnthropicReq(body: Json, model: string): Json {
  const out: Json = { model, messages: openAIMessagesToAnthropic(body) };
  const system = openAISystemText(body);
  if (system !== '') out.system = system;
  const maxTokens = openAIMaxTokens(body);
  if (maxTokens !== null) out.max_tokens = maxTokens;
  if (typeof body.temperature === 'number') out.temperature = body.temperature;
  if (typeof body.top_p === 'number') out.top_p = body.top_p;
  const stops = openAIStops(body);
  if (stops.length > 0) out.stop_sequences = stops;
  const tools = openAITools(body);
  if (tools.length > 0) {
    out.tools = tools.map((t) => ({ name: t.name, description: t.description, input_schema: t.parameters }));
  }
  const choice = body.tool_choice;
  if (choice === 'auto') out.tool_choice = { type: 'auto' };
  else if (choice === 'required' || choice === 'any') out.tool_choice = { type: 'any' };
  else if (isRecord(choice) && choice.type === 'function' && isRecord(choice.function) && typeof choice.function.name === 'string') {
    out.tool_choice = { type: 'tool', name: choice.function.name };
  }
  if (body.stream === true) out.stream = true;
  return out;
}

/** A provider's usage in one shape: {input, output} token counts. */
export interface Usage {
  input: number;
  output: number;
}

function num(v: unknown): number {
  return typeof v === 'number' && Number.isFinite(v) ? v : 0;
}

/** The text and usage of a Gemini generateContent answer. */
export function geminiAnswerParts(body: Json): { text: string; usage: Usage; finish: string } {
  const candidates = Array.isArray(body.candidates) ? body.candidates : [];
  const first = isRecord(candidates[0]) ? candidates[0] : {};
  const content = isRecord(first.content) ? first.content : {};
  const parts = Array.isArray(content.parts) ? content.parts : [];
  const text = parts.flatMap((p) => (isRecord(p) && typeof p.text === 'string' ? [p.text] : [])).join('');
  const meta = isRecord(body.usageMetadata) ? body.usageMetadata : {};
  return {
    text,
    usage: { input: num(meta.promptTokenCount), output: num(meta.candidatesTokenCount) },
    finish: typeof first.finishReason === 'string' ? first.finishReason : 'STOP',
  };
}

/** The text and usage of an Anthropic Messages answer. */
export function anthropicAnswerParts(body: Json): { text: string; usage: Usage; stop: string } {
  const usage = isRecord(body.usage) ? body.usage : {};
  return {
    text: textOf(body.content),
    usage: { input: num(usage.input_tokens), output: num(usage.output_tokens) },
    stop: typeof body.stop_reason === 'string' ? body.stop_reason : 'end_turn',
  };
}

/** The text and usage of an OpenAI Chat Completions answer. */
export function openAIAnswerParts(body: Json): { text: string; usage: Usage; finish: string } {
  const choices = Array.isArray(body.choices) ? body.choices : [];
  const choice = isRecord(choices[0]) ? choices[0] : {};
  const message = isRecord(choice.message) ? choice.message : {};
  const usage = isRecord(body.usage) ? body.usage : {};
  return {
    text: typeof message.content === 'string' ? message.content : textOf(message.content),
    usage: { input: num(usage.prompt_tokens), output: num(usage.completion_tokens) },
    finish: typeof choice.finish_reason === 'string' ? choice.finish_reason : 'stop',
  };
}

/** A Gemini finishReason as an OpenAI finish_reason. */
export function geminiFinishToOpenAI(finish: string): string {
  if (finish === 'MAX_TOKENS') return 'length';
  if (finish === 'SAFETY') return 'content_filter';
  return 'stop';
}

/** A Gemini finishReason as an Anthropic stop_reason. */
export function geminiFinishToAnthropic(finish: string): string {
  if (finish === 'MAX_TOKENS') return 'max_tokens';
  if (finish === 'SAFETY') return 'refusal';
  return 'end_turn';
}

/** An Anthropic stop_reason as an OpenAI finish_reason. */
export function anthropicStopToOpenAI(stop: string): string {
  if (stop === 'max_tokens') return 'length';
  if (stop === 'tool_use') return 'tool_calls';
  if (stop === 'refusal') return 'content_filter';
  return 'stop';
}

/** An OpenAI Chat Completions answer as an Anthropic Messages object. */
export function openAIJsonToAnthropic(body: Json, model: string): Json {
  const { text, usage, finish } = openAIAnswerParts(body);
  return {
    id: `msg_${typeof body.id === 'string' ? body.id : 'aggregate'}`,
    type: 'message',
    role: 'assistant',
    model,
    content: [{ type: 'text', text }],
    stop_reason: finish === 'length' ? 'max_tokens' : finish === 'tool_calls' ? 'tool_use' : finish === 'content_filter' ? 'refusal' : 'end_turn',
    stop_sequence: null,
    usage: { input_tokens: usage.input, output_tokens: usage.output },
  };
}

/** A Gemini generateContent answer as an OpenAI Chat Completions object. */
export function geminiJsonToOpenAI(body: Json, model: string): Json {
  const { text, usage, finish } = geminiAnswerParts(body);
  return {
    id: 'chatcmpl-sim',
    object: 'chat.completion',
    created: FIXED_OPENAI_CREATED,
    model,
    choices: [{ index: 0, message: { role: 'assistant', content: text }, finish_reason: geminiFinishToOpenAI(finish), logprobs: null }],
    usage: { prompt_tokens: usage.input, completion_tokens: usage.output, total_tokens: usage.input + usage.output },
  };
}

/** A Gemini generateContent answer as an Anthropic Messages object. */
export function geminiJsonToAnthropic(body: Json, model: string): Json {
  const { text, usage, finish } = geminiAnswerParts(body);
  return {
    id: 'msg_sim',
    type: 'message',
    role: 'assistant',
    model,
    content: [{ type: 'text', text }],
    stop_reason: geminiFinishToAnthropic(finish),
    stop_sequence: null,
    usage: { input_tokens: usage.input, output_tokens: usage.output },
  };
}

/** An Anthropic Messages answer as an OpenAI Chat Completions object. */
export function anthropicJsonToOpenAI(body: Json, model: string): Json {
  const { text, usage, stop } = anthropicAnswerParts(body);
  return {
    id: 'chatcmpl-sim',
    object: 'chat.completion',
    created: FIXED_OPENAI_CREATED,
    model,
    choices: [{ index: 0, message: { role: 'assistant', content: text }, finish_reason: anthropicStopToOpenAI(stop), logprobs: null }],
    usage: { prompt_tokens: usage.input, completion_tokens: usage.output, total_tokens: usage.input + usage.output },
  };
}

/** One SSE event line: `event: <name>` when named, then the JSON data line, then the blank line. */
export function sseEvent(event: string | null, data: unknown): string {
  return `${event === null ? '' : `event: ${event}\n`}data: ${JSON.stringify(data)}\n\n`;
}

/** Split SSE text into frames: {event, data} per frame. */
export function parseSseFrames(text: string): { event: string | null; data: string }[] {
  const frames: { event: string | null; data: string }[] = [];
  const normalized = text.replace(/\r\n/g, '\n');
  for (const frame of normalized.split('\n\n')) {
    if (frame.trim() === '') continue;
    let event: string | null = null;
    const data: string[] = [];
    for (const line of frame.split('\n')) {
      if (line.startsWith('event:')) event = line.slice('event:'.length).trim();
      else if (line.startsWith('data:')) data.push(line.slice('data:'.length).replace(/^ /, ''));
    }
    if (data.length > 0) frames.push({ event, data: data.join('\n') });
  }
  return frames;
}

/** A Gemini SSE stream, buffered, as {text, usage, finish}. */
export function geminiStreamParts(text: string): { text: string; usage: Usage; finish: string } {
  let out = '';
  let usage: Usage = { input: 0, output: 0 };
  let finish = 'STOP';
  for (const frame of parseSseFrames(text)) {
    if (frame.data.trim() === '[DONE]') continue;
    let chunk: unknown;
    try {
      chunk = JSON.parse(frame.data);
    } catch {
      continue;
    }
    if (!isRecord(chunk)) continue;
    const candidates = Array.isArray(chunk.candidates) ? chunk.candidates : [];
    const first = isRecord(candidates[0]) ? candidates[0] : {};
    const content = isRecord(first.content) ? first.content : {};
    const parts = Array.isArray(content.parts) ? content.parts : [];
    for (const p of parts) if (isRecord(p) && typeof p.text === 'string') out += p.text;
    if (typeof first.finishReason === 'string') finish = first.finishReason;
    if (isRecord(chunk.usageMetadata)) {
      usage = { input: num(chunk.usageMetadata.promptTokenCount), output: num(chunk.usageMetadata.candidatesTokenCount) };
    }
  }
  return { text: out, usage, finish };
}

/** An Anthropic SSE stream, buffered, as {text, usage, stop}. */
export function anthropicStreamParts(text: string): { text: string; usage: Usage; stop: string } {
  let out = '';
  let usage: Usage = { input: 0, output: 0 };
  let stop = 'end_turn';
  for (const frame of parseSseFrames(text)) {
    let data: unknown;
    try {
      data = JSON.parse(frame.data);
    } catch {
      continue;
    }
    if (!isRecord(data)) continue;
    if (frame.event === 'message_start' && isRecord(data.message) && isRecord(data.message.usage)) {
      usage.input = num(data.message.usage.input_tokens);
    }
    if (frame.event === 'content_block_delta' && isRecord(data.delta) && data.delta.type === 'text_delta' && typeof data.delta.text === 'string') {
      out += data.delta.text;
    }
    if (frame.event === 'message_delta') {
      if (isRecord(data.delta) && typeof data.delta.stop_reason === 'string') stop = data.delta.stop_reason;
      if (isRecord(data.usage)) usage.output = num(data.usage.output_tokens);
    }
  }
  return { text: out, usage, stop };
}

/** An OpenAI SSE stream, buffered, as {text, usage, finish}. */
export function openAIStreamParts(text: string): { text: string; usage: Usage; finish: string } {
  let out = '';
  let usage: Usage = { input: 0, output: 0 };
  let finish = 'stop';
  for (const frame of parseSseFrames(text)) {
    if (frame.data.trim() === '[DONE]') continue;
    let chunk: unknown;
    try {
      chunk = JSON.parse(frame.data);
    } catch {
      continue;
    }
    if (!isRecord(chunk)) continue;
    if (isRecord(chunk.usage)) {
      usage = { input: num(chunk.usage.prompt_tokens), output: num(chunk.usage.completion_tokens) };
    }
    const choices = Array.isArray(chunk.choices) ? chunk.choices : [];
    const choice = isRecord(choices[0]) ? choices[0] : null;
    if (choice === null) continue;
    const delta = isRecord(choice.delta) ? choice.delta : {};
    if (typeof delta.content === 'string') out += delta.content;
    if (typeof choice.finish_reason === 'string') finish = choice.finish_reason;
  }
  return { text: out, usage, finish };
}

/** Text as an OpenAI SSE stream: role chunk, 13-char content deltas, finish chunk, usage chunk, [DONE]. */
export function textToOpenAIStream(text: string, usage: Usage, finish: string, model: string): string {
  const frames: string[] = [];
  frames.push(sseEvent(null, { id: 'chatcmpl-sim', object: 'chat.completion.chunk', created: FIXED_OPENAI_CREATED, model, choices: [{ index: 0, delta: { role: 'assistant', content: '' }, finish_reason: null }] }));
  for (let at = 0; at < text.length; at += 13) {
    frames.push(sseEvent(null, { id: 'chatcmpl-sim', object: 'chat.completion.chunk', created: FIXED_OPENAI_CREATED, model, choices: [{ index: 0, delta: { content: text.slice(at, at + 13) }, finish_reason: null }] }));
  }
  if (text.length === 0) {
    frames.push(sseEvent(null, { id: 'chatcmpl-sim', object: 'chat.completion.chunk', created: FIXED_OPENAI_CREATED, model, choices: [{ index: 0, delta: { content: '' }, finish_reason: null }] }));
  }
  frames.push(sseEvent(null, { id: 'chatcmpl-sim', object: 'chat.completion.chunk', created: FIXED_OPENAI_CREATED, model, choices: [{ index: 0, delta: {}, finish_reason: finish }] }));
  frames.push(sseEvent(null, { id: 'chatcmpl-sim', object: 'chat.completion.chunk', created: FIXED_OPENAI_CREATED, model, choices: [], usage: { prompt_tokens: usage.input, completion_tokens: usage.output, total_tokens: usage.input + usage.output } }));
  frames.push('data: [DONE]\n\n');
  return frames.join('');
}

/** Text as an Anthropic SSE stream: message_start, one text block, message_delta, message_stop. */
export function textToAnthropicStream(text: string, usage: Usage, stop: string, model: string): string {
  const frames: string[] = [];
  frames.push(
    sseEvent('message_start', {
      type: 'message_start',
      message: { id: 'msg_stream', type: 'message', role: 'assistant', model, content: [], stop_reason: null, stop_sequence: null, usage: { input_tokens: usage.input, output_tokens: 0 } },
    }),
  );
  frames.push(sseEvent('content_block_start', { type: 'content_block_start', index: 0, content_block: { type: 'text', text: '' } }));
  for (let at = 0; at < text.length; at += 13) {
    frames.push(sseEvent('content_block_delta', { type: 'content_block_delta', index: 0, delta: { type: 'text_delta', text: text.slice(at, at + 13) } }));
  }
  if (text.length === 0) {
    frames.push(sseEvent('content_block_delta', { type: 'content_block_delta', index: 0, delta: { type: 'text_delta', text: '' } }));
  }
  frames.push(sseEvent('content_block_stop', { type: 'content_block_stop', index: 0 }));
  frames.push(sseEvent('message_delta', { type: 'message_delta', delta: { stop_reason: stop, stop_sequence: null }, usage: { output_tokens: usage.output } }));
  frames.push(sseEvent('message_stop', { type: 'message_stop' }));
  return frames.join('');
}
