/**
 * OpenAI-compatible API shim for Claude Code.
 *
 * Translates Anthropic SDK calls (anthropic.beta.messages.create) into
 * OpenAI-compatible chat completion requests and streams back events
 * in the Anthropic streaming format so the rest of the codebase is unaware.
 *
 * Supports: OpenAI, Azure OpenAI, Ollama, LM Studio, OpenRouter,
 * Together, Groq, Fireworks, DeepSeek, Mistral, and any OpenAI-compatible API.
 *
 * Environment variables:
 *   CLAUDE_CODE_USE_OPENAI=1          — enable this provider
 *   OPENAI_API_KEY=sk-...             — API key (optional for local models)
 *   OPENAI_BASE_URL=http://...        — base URL (default: https://api.openai.com/v1)
 *   OPENAI_MODEL=gpt-4o              — default model override
 */

// ---------------------------------------------------------------------------
// Types — minimal subset of Anthropic SDK types we need to produce
// ---------------------------------------------------------------------------

interface AnthropicUsage {
  input_tokens: number
  output_tokens: number
  cache_creation_input_tokens: number
  cache_read_input_tokens: number
}

interface AnthropicStreamEvent {
  type: string
  message?: Record<string, unknown>
  index?: number
  content_block?: Record<string, unknown>
  delta?: Record<string, unknown>
  usage?: Partial<AnthropicUsage>
}

// ---------------------------------------------------------------------------
// Message format conversion: Anthropic → OpenAI
// ---------------------------------------------------------------------------

interface OpenAIMessage {
  role: 'system' | 'user' | 'assistant' | 'tool'
  content?: string | Array<{ type: string; text?: string; image_url?: { url: string } }>
  tool_calls?: Array<{
    id: string
    type: 'function'
    function: { name: string; arguments: string }
  }>
  tool_call_id?: string
  name?: string
}

interface OpenAITool {
  type: 'function'
  function: {
    name: string
    description: string
    parameters: Record<string, unknown>
    strict?: boolean
  }
}

function convertSystemPrompt(
  system: unknown,
): string {
  if (!system) return ''
  if (typeof system === 'string') return system
  if (Array.isArray(system)) {
    return system
      .map((block: { type?: string; text?: string }) =>
        block.type === 'text' ? block.text ?? '' : '',
      )
      .join('\n\n')
  }
  return String(system)
}

function convertContentBlocks(
  content: unknown,
): string | Array<{ type: string; text?: string; image_url?: { url: string } }> {
  if (typeof content === 'string') return content
  if (!Array.isArray(content)) return String(content ?? '')

  const parts: Array<{ type: string; text?: string; image_url?: { url: string } }> = []
  for (const block of content) {
    switch (block.type) {
      case 'text':
        parts.push({ type: 'text', text: block.text ?? '' })
        break
      case 'image': {
        const src = block.source
        if (src?.type === 'base64') {
          parts.push({
            type: 'image_url',
            image_url: {
              url: `data:${src.media_type};base64,${src.data}`,
            },
          })
        } else if (src?.type === 'url') {
          parts.push({ type: 'image_url', image_url: { url: src.url } })
        }
        break
      }
      case 'tool_use':
        // handled separately
        break
      case 'tool_result':
        // handled separately
        break
      case 'thinking':
        // Append thinking as text with a marker for models that support reasoning
        if (block.thinking) {
          parts.push({ type: 'text', text: `<thinking>${block.thinking}</thinking>` })
        }
        break
      default:
        if (block.text) {
          parts.push({ type: 'text', text: block.text })
        }
    }
  }

  if (parts.length === 0) return ''
  if (parts.length === 1 && parts[0].type === 'text') return parts[0].text ?? ''
  return parts
}

function convertMessages(
  messages: Array<{ role: string; message?: { role?: string; content?: unknown }; content?: unknown }>,
  system: unknown,
): OpenAIMessage[] {
  const result: OpenAIMessage[] = []

  // System message first
  const sysText = convertSystemPrompt(system)
  if (sysText) {
    result.push({ role: 'system', content: sysText })
  }

  for (const msg of messages) {
    // Claude Code wraps messages in { role, message: { role, content } }
    const inner = msg.message ?? msg
    const role = (inner as { role?: string }).role ?? msg.role
    const content = (inner as { content?: unknown }).content

    if (role === 'user') {
      // Check for tool_result blocks in user messages
      if (Array.isArray(content)) {
        const toolResults = content.filter((b: { type?: string }) => b.type === 'tool_result')
        const otherContent = content.filter((b: { type?: string }) => b.type !== 'tool_result')

        // Emit tool results as tool messages
        for (const tr of toolResults) {
          const trContent = Array.isArray(tr.content)
            ? tr.content.map((c: { text?: string }) => c.text ?? '').join('\n')
            : typeof tr.content === 'string'
              ? tr.content
              : JSON.stringify(tr.content ?? '')
          result.push({
            role: 'tool',
            tool_call_id: tr.tool_use_id ?? 'unknown',
            content: tr.is_error ? `Error: ${trContent}` : trContent,
          })
        }

        // Emit remaining user content
        if (otherContent.length > 0) {
          result.push({
            role: 'user',
            content: convertContentBlocks(otherContent),
          })
        }
      } else {
        result.push({
          role: 'user',
          content: convertContentBlocks(content),
        })
      }
    } else if (role === 'assistant') {
      // Check for tool_use blocks
      if (Array.isArray(content)) {
        const toolUses = content.filter((b: { type?: string }) => b.type === 'tool_use')
        const textContent = content.filter(
          (b: { type?: string }) => b.type !== 'tool_use' && b.type !== 'thinking',
        )

        const assistantMsg: OpenAIMessage = {
          role: 'assistant',
          content: convertContentBlocks(textContent) as string,
        }

        if (toolUses.length > 0) {
          assistantMsg.tool_calls = toolUses.map(
            (tu: { id?: string; name?: string; input?: unknown }) => ({
              id: tu.id ?? `call_${Math.random().toString(36).slice(2)}`,
              type: 'function' as const,
              function: {
                name: tu.name ?? 'unknown',
                arguments:
                  typeof tu.input === 'string'
                    ? tu.input
                    : JSON.stringify(tu.input ?? {}),
              },
            }),
          )
        }

        result.push(assistantMsg)
      } else {
        result.push({
          role: 'assistant',
          content: convertContentBlocks(content) as string,
        })
      }
    }
  }

  return result
}

function convertTools(
  tools: Array<{ name: string; description?: string; input_schema?: Record<string, unknown> }>,
): OpenAITool[] {
  return tools
    .filter(t => t.name !== 'ToolSearchTool') // Not relevant for OpenAI
    .map(t => ({
      type: 'function' as const,
      function: {
        name: t.name,
        description: t.description ?? '',
        parameters: t.input_schema ?? { type: 'object', properties: {} },
      },
    }))
}

// ---------------------------------------------------------------------------
// Streaming: OpenAI SSE → Anthropic stream events
// ---------------------------------------------------------------------------

interface OpenAIStreamChunk {
  id: string
  object: string
  model: string
  choices: Array<{
    index: number
    delta: {
      role?: string
      content?: string | null
      tool_calls?: Array<{
        index: number
        id?: string
        type?: string
        function?: { name?: string; arguments?: string }
      }>
    }
    finish_reason: string | null
  }>
  usage?: {
    prompt_tokens?: number
    completion_tokens?: number
    total_tokens?: number
  }
}

function makeMessageId(): string {
  return `msg_${Math.random().toString(36).slice(2)}${Date.now().toString(36)}`
}

/**
 * Async generator that transforms an OpenAI SSE stream into
 * Anthropic-format BetaRawMessageStreamEvent objects.
 */
async function* openaiStreamToAnthropic(
  response: Response,
  model: string,
): AsyncGenerator<AnthropicStreamEvent> {
  const messageId = makeMessageId()
  let contentBlockIndex = 0
  const activeToolCalls = new Map<number, { id: string; name: string; index: number }>()
  let hasEmittedContentStart = false

  // Emit message_start
  yield {
    type: 'message_start',
    message: {
      id: messageId,
      type: 'message',
      role: 'assistant',
      content: [],
      model,
      stop_reason: null,
      stop_sequence: null,
      usage: {
        input_tokens: 0,
        output_tokens: 0,
        cache_creation_input_tokens: 0,
        cache_read_input_tokens: 0,
      },
    },
  }

  const reader = response.body?.getReader()
  if (!reader) return

  const decoder = new TextDecoder()
  let buffer = ''

  while (true) {
    const { done, value } = await reader.read()
    if (done) break

    buffer += decoder.decode(value, { stream: true })
    const lines = buffer.split('\n')
    buffer = lines.pop() ?? ''

    for (const line of lines) {
      const trimmed = line.trim()
      if (!trimmed || trimmed === 'data: [DONE]') continue
      if (!trimmed.startsWith('data: ')) continue

      let chunk: OpenAIStreamChunk
      try {
        chunk = JSON.parse(trimmed.slice(6))
      } catch {
        continue
      }

      for (const choice of chunk.choices ?? []) {
        const delta = choice.delta

        // Text content
        if (delta.content) {
          if (!hasEmittedContentStart) {
            yield {
              type: 'content_block_start',
              index: contentBlockIndex,
              content_block: { type: 'text', text: '' },
            }
            hasEmittedContentStart = true
          }
          yield {
            type: 'content_block_delta',
            index: contentBlockIndex,
            delta: { type: 'text_delta', text: delta.content },
          }
        }

        // Tool calls
        if (delta.tool_calls) {
          for (const tc of delta.tool_calls) {
            if (tc.id && tc.function?.name) {
              // New tool call starting
              if (hasEmittedContentStart) {
                yield {
                  type: 'content_block_stop',
                  index: contentBlockIndex,
                }
                contentBlockIndex++
                hasEmittedContentStart = false
              }

              const toolBlockIndex = contentBlockIndex
              activeToolCalls.set(tc.index, {
                id: tc.id,
                name: tc.function.name,
                index: toolBlockIndex,
              })

              yield {
                type: 'content_block_start',
                index: toolBlockIndex,
                content_block: {
                  type: 'tool_use',
                  id: tc.id,
                  name: tc.function.name,
                  input: {},
                },
              }
              contentBlockIndex++

              // Emit any initial arguments
              if (tc.function.arguments) {
                yield {
                  type: 'content_block_delta',
                  index: toolBlockIndex,
                  delta: {
                    type: 'input_json_delta',
                    partial_json: tc.function.arguments,
                  },
                }
              }
            } else if (tc.function?.arguments) {
              // Continuation of existing tool call
              const active = activeToolCalls.get(tc.index)
              if (active) {
                yield {
                  type: 'content_block_delta',
                  index: active.index,
                  delta: {
                    type: 'input_json_delta',
                    partial_json: tc.function.arguments,
                  },
                }
              }
            }
          }
        }

        // Finish
        if (choice.finish_reason) {
          // Close any open content blocks
          if (hasEmittedContentStart) {
            yield {
              type: 'content_block_stop',
              index: contentBlockIndex,
            }
          }
          // Close active tool calls
          for (const [, tc] of activeToolCalls) {
            yield { type: 'content_block_stop', index: tc.index }
          }

          const stopReason =
            choice.finish_reason === 'tool_calls'
              ? 'tool_use'
              : choice.finish_reason === 'length'
                ? 'max_tokens'
                : 'end_turn'

          yield {
            type: 'message_delta',
            delta: { stop_reason: stopReason, stop_sequence: null },
            usage: {
              output_tokens: chunk.usage?.completion_tokens ?? 0,
            },
          }
        }
      }
    }
  }

  yield { type: 'message_stop' }
}

// ---------------------------------------------------------------------------
// The shim client — duck-types as Anthropic SDK
// ---------------------------------------------------------------------------

interface ShimCreateParams {
  model: string
  messages: Array<Record<string, unknown>>
  system?: unknown
  tools?: Array<Record<string, unknown>>
  max_tokens: number
  stream?: boolean
  temperature?: number
  top_p?: number
  tool_choice?: unknown
  metadata?: unknown
  [key: string]: unknown
}

class OpenAIShimStream {
  private generator: AsyncGenerator<AnthropicStreamEvent>
  // The controller property is checked by claude.ts to distinguish streams from error messages
  controller = new AbortController()

  constructor(generator: AsyncGenerator<AnthropicStreamEvent>) {
    this.generator = generator
  }

  async *[Symbol.asyncIterator]() {
    yield* this.generator
  }
}

class OpenAIShimMessages {
  private baseUrl: string
  private apiKey: string
  private defaultHeaders: Record<string, string>

  constructor(
    baseUrl: string,
    apiKey: string,
    defaultHeaders: Record<string, string>,
  ) {
    this.baseUrl = baseUrl
    this.apiKey = apiKey
    this.defaultHeaders = defaultHeaders
  }

  create(
    params: ShimCreateParams,
    options?: { signal?: AbortSignal; headers?: Record<string, string> },
  ) {
    const self = this

    // Return a thenable that also has .withResponse()
    const promise = (async () => {
      const response = await self._doRequest(params, options)
      if (params.stream) {
        return new OpenAIShimStream(
          openaiStreamToAnthropic(response, params.model),
        )
      }
      // Non-streaming: parse the full response and convert
      const data = await response.json()
      return self._convertNonStreamingResponse(data, params.model)
    })()

    // Add .withResponse() for streaming path (claude.ts uses this)
    ;(promise as unknown as Record<string, unknown>).withResponse =
      async () => {
        const data = await promise
        return {
          data,
          response: new Response(),
          request_id: makeMessageId(),
        }
      }

    return promise
  }

  private async _doRequest(
    params: ShimCreateParams,
    options?: { signal?: AbortSignal; headers?: Record<string, string> },
  ): Promise<Response> {
    const openaiMessages = convertMessages(
      params.messages as Array<{
        role: string
        message?: { role?: string; content?: unknown }
        content?: unknown
      }>,
      params.system,
    )

    const body: Record<string, unknown> = {
      model: params.model,
      messages: openaiMessages,
      max_tokens: params.max_tokens,
      stream: params.stream ?? false,
    }

    if (params.stream) {
      body.stream_options = { include_usage: true }
    }

    if (params.temperature !== undefined) body.temperature = params.temperature
    if (params.top_p !== undefined) body.top_p = params.top_p

    // Convert tools
    if (params.tools && params.tools.length > 0) {
      const converted = convertTools(
        params.tools as Array<{
          name: string
          description?: string
          input_schema?: Record<string, unknown>
        }>,
      )
      if (converted.length > 0) {
        body.tools = converted
        // Convert tool_choice
        if (params.tool_choice) {
          const tc = params.tool_choice as { type?: string; name?: string }
          if (tc.type === 'auto') {
            body.tool_choice = 'auto'
          } else if (tc.type === 'tool' && tc.name) {
            body.tool_choice = {
              type: 'function',
              function: { name: tc.name },
            }
          } else if (tc.type === 'any') {
            body.tool_choice = 'required'
          }
        }
      }
    }

    const url = `${this.baseUrl}/chat/completions`
    const headers: Record<string, string> = {
      'Content-Type': 'application/json',
      ...this.defaultHeaders,
      ...(options?.headers ?? {}),
    }

    if (this.apiKey) {
      headers['Authorization'] = `Bearer ${this.apiKey}`
    }

    const response = await fetch(url, {
      method: 'POST',
      headers,
      body: JSON.stringify(body),
      signal: options?.signal,
    })

    if (!response.ok) {
      const errorBody = await response.text().catch(() => 'unknown error')
      throw new Error(
        `OpenAI API error ${response.status}: ${errorBody}`,
      )
    }

    return response
  }

  private _convertNonStreamingResponse(
    data: {
      id?: string
      model?: string
      choices?: Array<{
        message?: {
          role?: string
          content?: string | null
          tool_calls?: Array<{
            id: string
            function: { name: string; arguments: string }
          }>
        }
        finish_reason?: string
      }>
      usage?: {
        prompt_tokens?: number
        completion_tokens?: number
      }
    },
    model: string,
  ) {
    const choice = data.choices?.[0]
    const content: Array<Record<string, unknown>> = []

    if (choice?.message?.content) {
      content.push({ type: 'text', text: choice.message.content })
    }

    if (choice?.message?.tool_calls) {
      for (const tc of choice.message.tool_calls) {
        let input: unknown
        try {
          input = JSON.parse(tc.function.arguments)
        } catch {
          input = { raw: tc.function.arguments }
        }
        content.push({
          type: 'tool_use',
          id: tc.id,
          name: tc.function.name,
          input,
        })
      }
    }

    const stopReason =
      choice?.finish_reason === 'tool_calls'
        ? 'tool_use'
        : choice?.finish_reason === 'length'
          ? 'max_tokens'
          : 'end_turn'

    return {
      id: data.id ?? makeMessageId(),
      type: 'message',
      role: 'assistant',
      content,
      model: data.model ?? model,
      stop_reason: stopReason,
      stop_sequence: null,
      usage: {
        input_tokens: data.usage?.prompt_tokens ?? 0,
        output_tokens: data.usage?.completion_tokens ?? 0,
        cache_creation_input_tokens: 0,
        cache_read_input_tokens: 0,
      },
    }
  }
}

class OpenAIShimBeta {
  messages: OpenAIShimMessages

  constructor(
    baseUrl: string,
    apiKey: string,
    defaultHeaders: Record<string, string>,
  ) {
    this.messages = new OpenAIShimMessages(baseUrl, apiKey, defaultHeaders)
  }
}

/**
 * Creates an Anthropic SDK-compatible client that routes requests
 * to an OpenAI-compatible API endpoint.
 *
 * Usage:
 *   CLAUDE_CODE_USE_OPENAI=1 OPENAI_API_KEY=sk-... OPENAI_MODEL=gpt-4o
 */
export function createOpenAIShimClient(options: {
  defaultHeaders?: Record<string, string>
  maxRetries?: number
  timeout?: number
}): unknown {
  const baseUrl = (
    process.env.OPENAI_BASE_URL ??
    process.env.OPENAI_API_BASE ??
    'https://api.openai.com/v1'
  ).replace(/\/+$/, '')

  const apiKey = process.env.OPENAI_API_KEY ?? ''

  const headers = {
    ...(options.defaultHeaders ?? {}),
  }

  const beta = new OpenAIShimBeta(baseUrl, apiKey, headers)

  // Duck-type as Anthropic client
  return {
    beta,
    // Some code paths access .messages directly (non-beta)
    messages: beta.messages,
  }
}
