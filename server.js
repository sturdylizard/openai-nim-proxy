// server.js - OpenAI to NVIDIA NIM API Proxy
const express = require('express');
const cors    = require('cors');
const axios   = require('axios');

const app  = express();
const PORT = process.env.PORT || 3000;

// ── NVIDIA NIM API configuration ─────────────────────────────────────────────
const NIM_API_BASE = process.env.NIM_API_BASE || 'https://integrate.api.nvidia.com/v1';
const NIM_API_KEY  = process.env.NIM_API_KEY;

// Fail fast – every request would silently fail without a key
if (!NIM_API_KEY) {
  console.error('FATAL: NIM_API_KEY environment variable is not set. Exiting.');
  process.exit(1);
}

// ── Behaviour toggles ────────────────────────────────────────────────────────
// Set to true to wrap chain-of-thought output in <think>…</think> tags
const SHOW_REASONING = false;

// Set to true to pass { thinking: true } to models that support it
const ENABLE_THINKING_MODE = false;

// ── Model registry ───────────────────────────────────────────────────────────
// Add or remove models here. The key is what you type into JanitorAI's
// "Model" field; the value is the real NVIDIA NIM model string.
//
// HOW TO SWITCH MODELS IN JANITORAI:
//   1. Set your API base URL to this proxy's address.
//   2. In the "Model" field, type one of the shorthand keys below
//      (e.g. "fast", "smart") — or paste any full NIM model ID directly.
//   3. The proxy resolves the name and forwards to the right model.
//
const MODEL_REGISTRY = {
  // ── Shorthand aliases ─────────────────────────────────────────────────────
  'default'  : 'deepseek-ai/deepseek-v4-pro',
  'fast'     : 'nvidia/nemotron-3-super-120b-a12b',        // Smaller, quicker
  'smart'    : 'minimaxai/minimax-m3',               // Largest, best reasoning
  'balanced' : 'mistralai/mistral-medium-3.5-128b',  // Good speed/quality mix

  // ── Short brand names ─────────────────────────────────────────────────────
  'glm'        : 'z-ai/glm-5.2',
  'deepseek'   : 'deepseek-ai/deepseek-v4-pro',
  'qwen'       : 'qwen/qwen3.5-397b-a17b',

  // ── Add your own below ────────────────────────────────────────────────────
  // 'mymodel' : 'nvidia/some-other-model-id',
};

// Used when the client sends no model name at all
const DEFAULT_MODEL = MODEL_REGISTRY['default'];

/**
 * Resolves a model name from the client to a real NIM model string.
 *   1. Looks up the name in MODEL_REGISTRY (case-insensitive).
 *   2. Falls through to the raw name so full NIM IDs work without registration.
 *   3. Falls back to DEFAULT_MODEL when nothing is provided.
 */
function resolveModel(requested) {
  if (!requested) return DEFAULT_MODEL;
  const key = requested.trim().toLowerCase();
  return MODEL_REGISTRY[key] ?? requested.trim();
}

// ── Middleware ────────────────────────────────────────────────────────────────
app.use(cors({
  // Tighten to your actual frontend origin in production, e.g.:
  // origin: 'https://janitorai.com'
  origin: process.env.ALLOWED_ORIGIN || '*'
}));
app.use(express.json());

// ── Retry helper ─────────────────────────────────────────────────────────────
/**
 * Calls `fn` up to `maxAttempts` times, retrying only on 429 responses.
 * Waits `baseDelayMs * 2^attempt` ms between tries (exponential backoff).
 * Respects the Retry-After header from NVIDIA if present.
 */
async function withRetry(fn, maxAttempts = 3, baseDelayMs = 1000) {
  let lastError;

  for (let attempt = 0; attempt < maxAttempts; attempt++) {
    try {
      return await fn();
    } catch (err) {
      const status = err.response?.status;
      if (status !== 429) throw err; // Not rate-limited – rethrow immediately

      lastError = err;

      const retryAfter = err.response?.headers?.['retry-after'];
      const waitMs = retryAfter
        ? parseFloat(retryAfter) * 1000
        : baseDelayMs * Math.pow(2, attempt);

      console.warn(
        `Rate limited (429). Attempt ${attempt + 1}/${maxAttempts}. ` +
        `Retrying in ${Math.round(waitMs)}ms…`
      );
      await new Promise((resolve) => setTimeout(resolve, waitMs));
    }
  }

  throw lastError;
}

// ── Health check ──────────────────────────────────────────────────────────────
app.get('/health', (req, res) => {
  res.json({
    status:           'ok',
    service:          'OpenAI to NVIDIA NIM Proxy',
    default_model:    DEFAULT_MODEL,
    available_models: Object.keys(MODEL_REGISTRY),
    reasoning_display: SHOW_REASONING,
    thinking_mode:    ENABLE_THINKING_MODE
  });
});

// ── List models (OpenAI-compatible) ──────────────────────────────────────────
// Exposes every registered alias so JanitorAI's model picker can list them.
app.get('/v1/models', (req, res) => {
  const now = Math.floor(Date.now() / 1000);
  const data = Object.entries(MODEL_REGISTRY).map(([alias, nimId]) => ({
    id:         alias,        // The shorthand the client uses
    nim_model:  nimId,        // The actual NIM model it maps to (informational)
    object:     'model',
    created:    now,
    owned_by:   'nvidia-nim-proxy'
  }));

  res.json({ object: 'list', data });
});

// ── Chat completions (main proxy) ─────────────────────────────────────────────
app.post('/v1/chat/completions', async (req, res) => {
  const { model, messages, temperature, max_tokens, stream } = req.body;

  // ── Input validation ────────────────────────────────────────────────────────
  if (!Array.isArray(messages) || messages.length === 0) {
    return res.status(400).json({
      error: {
        message: '`messages` must be a non-empty array.',
        type:    'invalid_request_error',
        code:    400
      }
    });
  }

  // ── Resolve model ───────────────────────────────────────────────────────────
  const resolvedModel = resolveModel(model);
  console.log(`Model requested: "${model ?? '(none)'}" → resolved: "${resolvedModel}"`);

  // ── Build NIM request ───────────────────────────────────────────────────────
  const nimRequest = {
    model:       resolvedModel,
    messages,
    temperature: temperature !== undefined ? temperature : 0.7, // 0 must be forwarded
    max_tokens:  max_tokens || 9024,
    stream:      !!stream,
    ...(ENABLE_THINKING_MODE && { chat_template_kwargs: { thinking: true } })
  };

  try {
    const response = await withRetry(() =>
      axios.post(
        `${NIM_API_BASE}/chat/completions`,
        nimRequest,
        {
          headers: {
            Authorization:  `Bearer ${NIM_API_KEY}`,
            'Content-Type': 'application/json'
          },
          timeout:      30_000, // Prevent hung connections
          responseType: stream ? 'stream' : 'json'
        }
      )
    );

    // ── Streaming path ──────────────────────────────────────────────────────
    if (stream) {
      res.setHeader('Content-Type', 'text/event-stream');
      res.setHeader('Cache-Control', 'no-cache');
      res.setHeader('Connection',    'keep-alive');

      let buffer           = '';
      let reasoningStarted = false;

      response.data.on('data', (chunk) => {
        buffer += chunk.toString();
        const lines = buffer.split('\n');
        buffer = lines.pop() ?? ''; // Keep incomplete trailing line

        for (const line of lines) {
          if (!line.startsWith('data: ')) continue;

          if (line.includes('[DONE]')) {
            res.write('data: [DONE]\n\n'); // Double newline required by SSE spec
            continue;
          }

          try {
            const data  = JSON.parse(line.slice(6));
            const delta = data.choices?.[0]?.delta;

            if (delta) {
              const reasoning = delta.reasoning_content ?? null; // NIM's CoT field
              const content   = delta.content ?? null;
              delete delta.reasoning_content; // Strip NIM-specific field

              if (SHOW_REASONING) {
                if (reasoning) {
                  delta.content    = (reasoningStarted ? '' : '<think>\n') + reasoning;
                  reasoningStarted = true;
                } else if (content !== null && reasoningStarted) {
                  delta.content    = '</think>\n\n' + content;
                  reasoningStarted = false;
                } else {
                  delta.content = content ?? '';
                }
              } else {
                delta.content = content ?? '';
              }
            }

            res.write(`data: ${JSON.stringify(data)}\n\n`);
          } catch (parseError) {
            console.error('SSE chunk parse error:', parseError.message);
            // Never re-emit a malformed chunk to the client
          }
        }
      });

      response.data.on('end',   ()    => res.end());
      response.data.on('error', (err) => {
        console.error('Upstream stream error:', err.message);
        res.end();
      });

    // ── Non-streaming path ──────────────────────────────────────────────────
    } else {
      const choices = response.data.choices.map((choice) => {
        let content = choice.message?.content ?? '';

        if (SHOW_REASONING && choice.message?.reasoning_content) {
          content = `<think>\n${choice.message.reasoning_content}\n</think>\n\n${content}`;
        }

        return {
          index:         choice.index,
          message:       { role: choice.message.role, content },
          finish_reason: choice.finish_reason
        };
      });

      res.json({
        id:      `chatcmpl-${Date.now()}`,
        object:  'chat.completion',
        created: Math.floor(Date.now() / 1000),
        model:   resolvedModel, // Return the actual model used
        choices,
        usage: response.data.usage ?? {
          prompt_tokens:     0,
          completion_tokens: 0,
          total_tokens:      0
        }
      });
    }

  } catch (error) {
    console.error('Proxy error:', error.message);

    if (error.response) {
      console.error('Upstream status:', error.response.status);

      // When responseType is 'stream', error.response.data is a raw Node.js
      // socket/stream — not parsed JSON — so JSON.stringify() throws a
      // "circular structure" TypeError. Detect and handle it safely.
      const rawData = error.response.data;
      if (rawData && typeof rawData === 'object' && typeof rawData.pipe === 'function') {
        console.error('Upstream body: <stream — not serialisable; check upstream logs>');
      } else {
        try {
          console.error('Upstream body:', JSON.stringify(rawData));
        } catch {
          console.error('Upstream body: <could not serialise response data>');
        }
      }
    }

    const status  = error.response?.status || 500;
    const message = status === 429
      ? 'The upstream NVIDIA NIM API is rate-limiting this proxy. Please try again shortly.'
      : 'The proxy failed to complete the request. Check server logs for details.';

    res.status(status).json({
      error: { message, type: 'proxy_error', code: status }
    });
  }
});

// ── 404 catch-all ─────────────────────────────────────────────────────────────
app.all('*', (req, res) => {
  res.status(404).json({
    error: {
      message: `Endpoint ${req.path} is not supported by this proxy.`,
      type:    'invalid_request_error',
      code:    404
    }
  });
});

// ── Start server ──────────────────────────────────────────────────────────────
app.listen(PORT, () => {
  console.log(`\nOpenAI → NVIDIA NIM Proxy  listening on port ${PORT}`);
  console.log(`Health check    : http://localhost:${PORT}/health`);
  console.log(`Default model   : ${DEFAULT_MODEL}`);
  console.log(`Registered aliases:`);
  Object.entries(MODEL_REGISTRY).forEach(([alias, id]) =>
    console.log(`  "${alias}" → ${id}`)
  );
  console.log(`Reasoning       : ${SHOW_REASONING       ? 'VISIBLE'  : 'HIDDEN'}`);
  console.log(`Thinking mode   : ${ENABLE_THINKING_MODE ? 'ENABLED'  : 'DISABLED'}\n`);
});
