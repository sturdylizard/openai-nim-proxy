// server.js - OpenAI to NVIDIA NIM API Proxy
const express = require('express');
const cors = require('cors');
const axios = require('axios');

const app = express();
const PORT = process.env.PORT || 3000;

// ── NVIDIA NIM API configuration ────────────────────────────────────────────
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

// ── Target model ─────────────────────────────────────────────────────────────
const TARGET_MODEL = 'z-ai/glm-5.2'; // Change to your preferred model

// ── Middleware ───────────────────────────────────────────────────────────────
app.use(cors({
  // Tighten this to your actual frontend origin in production, e.g.:
  // origin: 'https://my-app.example.com'
  origin: process.env.ALLOWED_ORIGIN || '*'
}));
app.use(express.json());

// ── Helper ───────────────────────────────────────────────────────────────────
/**
 * Build a minimal SSE data line by cloning the parsed SSE object and
 * overwriting delta.content, so the rest of the fields (id, model, …) are kept.
 */
function buildSSELine(data, content) {
  data.choices[0].delta.content = content;
  return `data: ${JSON.stringify(data)}\n\n`;
}

// ── Health check ─────────────────────────────────────────────────────────────
app.get('/health', (req, res) => {
  res.json({
    status: 'ok',
    service: 'OpenAI to NVIDIA NIM Proxy',
    model: TARGET_MODEL,
    reasoning_display: SHOW_REASONING,
    thinking_mode: ENABLE_THINKING_MODE
  });
});

// ── List models (OpenAI-compatible) ─────────────────────────────────────────
app.get('/v1/models', (req, res) => {
  res.json({
    object: 'list',
    data: [{
      id: TARGET_MODEL,
      object: 'model',
      created: Math.floor(Date.now() / 1000),
      owned_by: 'nvidia-nim-proxy'
    }]
  });
});

// ── Chat completions (main proxy) ────────────────────────────────────────────
app.post('/v1/chat/completions', async (req, res) => {
  const { messages, temperature, max_tokens, stream } = req.body;

  // ── Input validation ──────────────────────────────────────────────────────
  if (!Array.isArray(messages) || messages.length === 0) {
    return res.status(400).json({
      error: {
        message: '`messages` must be a non-empty array.',
        type: 'invalid_request_error',
        code: 400
      }
    });
  }

  // ── Build NIM request ─────────────────────────────────────────────────────
  const nimRequest = {
    model: TARGET_MODEL,
    messages,
    // Use !== undefined so temperature: 0 is forwarded correctly
    temperature: temperature !== undefined ? temperature : 0.7,
    max_tokens: max_tokens || 9024,
    stream: !!stream,
    // Spread thinking flag at the top level, not inside extra_body
    ...(ENABLE_THINKING_MODE && { chat_template_kwargs: { thinking: true } })
  };

  try {
    const response = await axios.post(
      `${NIM_API_BASE}/chat/completions`,
      nimRequest,
      {
        headers: {
          Authorization: `Bearer ${NIM_API_KEY}`,
          'Content-Type': 'application/json'
        },
        // Prevent hung connections from blocking the server indefinitely
        timeout: 20_000,
        responseType: stream ? 'stream' : 'json'
      }
    );

    // ── Streaming path ──────────────────────────────────────────────────────
    if (stream) {
      res.setHeader('Content-Type', 'text/event-stream');
      res.setHeader('Cache-Control', 'no-cache');
      res.setHeader('Connection', 'keep-alive');

      let buffer           = '';
      let reasoningStarted = false; // tracks whether we opened a <think> tag

      response.data.on('data', (chunk) => {
        buffer += chunk.toString();
        const lines = buffer.split('\n');
        // Keep any incomplete trailing line in the buffer
        buffer = lines.pop() ?? '';

        for (const line of lines) {
          if (!line.startsWith('data: ')) continue;

          // SSE termination signal – must end with double newline
          if (line.includes('[DONE]')) {
            res.write('data: [DONE]\n\n');
            continue;
          }

          try {
            const data  = JSON.parse(line.slice(6));
            const delta = data.choices?.[0]?.delta;

            if (delta) {
              // NIM uses `reasoning_content` for chain-of-thought (not `_content`)
              const reasoning = delta.reasoning_content ?? null;
              const content   = delta.content ?? null;

              // Remove the NIM-specific field before forwarding
              delete delta.reasoning_content;

              if (SHOW_REASONING) {
                if (reasoning) {
                  // First reasoning chunk: open the <think> tag
                  // Subsequent chunks: just append reasoning text
                  delta.content = (reasoningStarted ? '' : '<think>\n') + reasoning;
                  reasoningStarted = true;
                } else if (content !== null && reasoningStarted) {
                  // First content chunk after reasoning: close the tag
                  delta.content = '</think>\n\n' + content;
                  reasoningStarted = false;
                } else {
                  delta.content = content ?? '';
                }
              } else {
                // Reasoning hidden – forward only the main content
                delta.content = content ?? '';
              }
            }

            res.write(`data: ${JSON.stringify(data)}\n\n`);
          } catch (parseError) {
            console.error('SSE chunk parse error:', parseError.message);
            // Do not re-emit the raw malformed line to the client
          }
        }
      });

      response.data.on('end',   ()    => res.end());
      response.data.on('error', (err) => {
        console.error('Upstream stream error:', err.message);
        res.end();
      });

    // ── Non-streaming path ────────────────────────────────────────────────
    } else {
      const choices = response.data.choices.map((choice) => {
        let content = choice.message?.content ?? '';

        if (SHOW_REASONING && choice.message?.reasoning_content) {
          content = `<think>\n${choice.message.reasoning_content}\n</think>\n\n${content}`;
        }

        return {
          index: choice.index,
          message: {
            role: choice.message.role,
            content
          },
          finish_reason: choice.finish_reason
        };
      });

      res.json({
        id:      `chatcmpl-${Date.now()}`,
        object:  'chat.completion',
        created: Math.floor(Date.now() / 1000),
        // Return the model that was actually used, not whatever the client sent
        model:   TARGET_MODEL,
        choices,
        usage: response.data.usage ?? {
          prompt_tokens:     0,
          completion_tokens: 0,
          total_tokens:      0
        }
      });
    }

  } catch (error) {
    // Log full error server-side for debugging
    console.error('Proxy error:', error.message);
    if (error.response) {
      console.error('Upstream status:', error.response.status);
      console.error('Upstream body:',  JSON.stringify(error.response.data));
    }

    // Return a sanitised error to the client – no internal details leaked
    res.status(error.response?.status || 500).json({
      error: {
        message: 'The proxy failed to complete the request. Check server logs for details.',
        type:    'proxy_error',
        code:    error.response?.status || 500
      }
    });
  }
});

// ── 404 catch-all ────────────────────────────────────────────────────────────
app.all('*', (req, res) => {
  res.status(404).json({
    error: {
      message: `Endpoint ${req.path} is not supported by this proxy.`,
      type:    'invalid_request_error',
      code:    404
    }
  });
});

// ── Start server ─────────────────────────────────────────────────────────────
app.listen(PORT, () => {
  console.log(`OpenAI → NVIDIA NIM Proxy  listening on port ${PORT}`);
  console.log(`Health check : http://localhost:${PORT}/health`);
  console.log(`Target model : ${TARGET_MODEL}`);
  console.log(`Reasoning    : ${SHOW_REASONING      ? 'VISIBLE'  : 'HIDDEN'}`);
  console.log(`Thinking mode: ${ENABLE_THINKING_MODE ? 'ENABLED' : 'DISABLED'}`);
});
