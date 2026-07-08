/**
 * LangGraph Server client.
 *
 * Wraps @langchain/langgraph-sdk and adapts its stream events into the
 * StreamEvent protocol consumed by handleEvent() in SessionsPage.
 */

import { Client } from "@langchain/langgraph-sdk"
import type { StreamEvent, ThreadSummary } from "./types"

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

const LG_BASE_URL = (import.meta.env.VITE_LANGGRAPH_BASE_URL as string | undefined) ?? "http://localhost:2024"
const STREAM_MODES = ["updates", "messages", "custom"] as const

type LgChunk = {
  event: string
  data: unknown
  id?: string | null
}

function asRecord(value: unknown): Record<string, unknown> {
  return value && typeof value === "object" ? (value as Record<string, unknown>) : {}
}

// ---------------------------------------------------------------------------
// Singleton client + assistant
// ---------------------------------------------------------------------------

let _client: Client | null = null
let _routerAssistantId: string | null = null

function getClient(): Client {
  if (!_client) _client = new Client({ apiUrl: LG_BASE_URL })
  return _client
}

export async function getRouterAssistantId(): Promise<string> {
  if (_routerAssistantId) return _routerAssistantId
  const c = getClient()
  // Look for an existing router assistant first (idempotent across page loads)
  const results = await c.assistants.search({ graphId: "router", limit: 1 })
  if (results.length > 0) {
    _routerAssistantId = results[0].assistant_id
    return _routerAssistantId!
  }
  const a = await c.assistants.create({ graphId: "router", config: {} })
  _routerAssistantId = a.assistant_id
  return _routerAssistantId!
}

// ---------------------------------------------------------------------------
// Thread management
// ---------------------------------------------------------------------------

export async function lgCreateThread(conversationId?: string): Promise<string> {
  const metadata = conversationId ? { conversation_id: conversationId } : undefined
  const thread = await getClient().threads.create({ metadata })
  return thread.thread_id
}

export async function lgListThreads(): Promise<ThreadSummary[]> {
  try {
    const results = await getClient().threads.search({ limit: 100 })
    return results
      .map((t) => {
        const thread = asRecord(t)
        const threadId = thread.thread_id as string
        const metadata = asRecord(thread.metadata)
        const conversationId = (metadata.conversation_id as string) ?? threadId
        if (conversationId.includes("::worker::")) return null
        return {
          conversation_id: conversationId,
          thread_id: threadId,
          agent_key: null,
          step_count: 0,
        } as ThreadSummary
      })
      .filter((x): x is ThreadSummary => x !== null)
  } catch {
    return []
  }
}

export async function lgDeleteThread(conversationId: string): Promise<boolean> {
  try {
    const c = getClient()
    let lgThreadId = loadThreadMapping(conversationId)
    if (!lgThreadId) {
      const results = await c.threads.search({ metadata: { conversation_id: conversationId }, limit: 1 })
      lgThreadId = results[0]?.thread_id ?? null
    }
    lgThreadId = lgThreadId ?? conversationId
    if (!lgThreadId) return false
    await c.threads.delete(lgThreadId)
    sessionStorage.removeItem(THREAD_KEY + conversationId)
    return true
  } catch {
    return false
  }
}

export async function lgGetThreadStatus(conversationId: string): Promise<string> {
  try {
    const c = getClient()
    let lgThreadId = loadThreadMapping(conversationId)
    if (!lgThreadId) {
      const results = await c.threads.search({ metadata: { conversation_id: conversationId }, limit: 1 })
      lgThreadId = results[0]?.thread_id ?? null
    }
    if (!lgThreadId) return "idle"
    const thread = await c.threads.get(lgThreadId)
    return (asRecord(thread).status as string) ?? "idle"
  } catch {
    return "idle"
  }
}

export async function lgCancelRun(lgThreadId: string, runId: string): Promise<void> {
  try {
    await getClient().runs.cancel(lgThreadId, runId, false)
  } catch {
    // ignore
  }
}

// ---------------------------------------------------------------------------
// Thread mapping (conversationId → lgThreadId)
// ---------------------------------------------------------------------------

const THREAD_KEY = "lg_thread:"  // lg_thread:<conversationId> → lgThreadId

export function persistThreadMapping(conversationId: string, lgThreadId: string): void {
  sessionStorage.setItem(THREAD_KEY + conversationId, lgThreadId)
}

export function loadThreadMapping(conversationId: string): string | null {
  return sessionStorage.getItem(THREAD_KEY + conversationId)
}

// ---------------------------------------------------------------------------
// Run mapping (lgThreadId → latest runId)
// ---------------------------------------------------------------------------

const RUN_KEY = "lg_run:"  // lg_run:<lgThreadId> → runId

export function persistRunMapping(lgThreadId: string, runId: string): void {
  sessionStorage.setItem(RUN_KEY + lgThreadId, runId)
}

export function loadRunMapping(lgThreadId: string): string | null {
  return sessionStorage.getItem(RUN_KEY + lgThreadId)
}

// ---------------------------------------------------------------------------
// Event mapping: LangGraph chunk → StreamEvent
// ---------------------------------------------------------------------------

function mapChunk(chunk: LgChunk, threadId: string): StreamEvent | null {
  const { event, data } = chunk

  if (event === "metadata") {
    const d = data as Record<string, unknown>
    return {
      event: "start",
      data: { thread_id: threadId, run_id: d.run_id as string },
    } as StreamEvent
  }

  if (event === "updates" && data && typeof data === "object") {
    const entries = Object.entries(data as Record<string, unknown>)
    if (entries.length === 0) return null
    const [node, diff] = entries[0]
    const d = diff as Record<string, unknown> | null

    // final_output signals run completion — translate to completed event
    if (node === "final_output") {
      return {
        event: "completed",
        data: { output: (d?.final_response as string) ?? "" },
      } as StreamEvent
    }

    return {
      event: "node",
      data: {
        node,
        // Expose intent/phase from the node diff for the node timeline UI
        phase: (d?.router_intent as string) ?? (d?.phase as string) ?? null,
      },
    } as StreamEvent
  }

  // Token streaming from LLM nodes.
  // messages mode: data is a list of [AIMessageChunk_dict, metadata_dict] pairs.
  if (event === "messages" || (typeof event === "string" && event.startsWith("messages"))) {
    const delta = extractMessageDelta(data)
    if (delta) {
      return { event: "message_delta", data: { delta } } as StreamEvent
    }
    return null
  }

  // Custom events dispatched from nodes via adispatch_custom_event (Wave 2).
  if (event === "custom" && data && typeof data === "object") {
    const d = data as Record<string, unknown>
    if (typeof d.event === "string") return data as StreamEvent
  }

  return null
}

function extractMessageDelta(data: unknown): string {
  if (!Array.isArray(data) || data.length === 0) return ""
  const item = data[0]
  // item can be [msg_chunk, metadata] tuple or the chunk dict directly
  const msg = Array.isArray(item) ? item[0] : item
  if (!msg || typeof msg !== "object") return ""
  const content = (msg as Record<string, unknown>).content
  if (typeof content === "string") return content
  if (Array.isArray(content)) {
    for (const block of content as Array<Record<string, unknown>>) {
      if (block.type === "text" && typeof block.text === "string") return block.text
    }
  }
  return ""
}

// ---------------------------------------------------------------------------
// Public streaming API
// ---------------------------------------------------------------------------

// Drain a LangGraph event stream, mapping each chunk and forwarding it.
// Shared by the new-send path and the reconnect/join path.
async function pumpStream(
  stream: AsyncIterable<unknown>,
  lgThreadId: string,
  onEvent: (e: StreamEvent) => void,
  signal?: AbortSignal,
): Promise<void> {
  for await (const chunk of stream) {
    if (signal?.aborted) break
    const mapped = mapChunk(chunk as LgChunk, lgThreadId)
    if (mapped) {
      onEvent(mapped)
      if (mapped.event === "completed") break
    }
  }
}

// New user send: start a background run that keeps executing even if this
// client disconnects (onDisconnect: "continue"), then stream its events.
// The run_id is persisted so a refresh/reconnect can resume instead of
// starting a second turn.
export async function streamLangGraph(
  lgThreadId: string,
  userInput: string,
  onEvent: (e: StreamEvent) => void,
  signal?: AbortSignal,
): Promise<void> {
  const c = getClient()
  const assistantId = await getRouterAssistantId()

  const run = await c.runs.create(lgThreadId, assistantId, {
    input: { user_input: userInput },
    streamMode: [...STREAM_MODES],
    onDisconnect: "continue",
    streamResumable: true,
  })
  persistRunMapping(lgThreadId, run.run_id)

  // joinStream doesn't reliably emit the server's `metadata` event up front, so
  // synthesize the run-start signal here. This drives the UI's run-start
  // side-effects (the "Thinking…" bubble, the worked timer, run/thread ids)
  // that previously rode on the metadata→start mapping.
  onEvent({
    event: "start",
    data: { thread_id: lgThreadId, run_id: run.run_id },
  } as StreamEvent)

  await pumpStream(
    c.runs.joinStream(lgThreadId, run.run_id, { signal, streamMode: [...STREAM_MODES] }),
    lgThreadId,
    onEvent,
    signal,
  )
}

// Resume: start a run with empty input so the graph continues from its last
// checkpoint (the next pending nodes) instead of restarting the turn, and
// stream that fresh run live. Used by both the Resume button (thread already
// idle after a cancel) and refresh (thread still running). multitaskStrategy
// "interrupt" makes the running case work: it interrupts any in-flight run —
// keeping its completed checkpoint — and starts the continuation. It's a no-op
// when the thread is already idle, so the Resume button is unaffected.
export async function resumeLangGraph(
  lgThreadId: string,
  onEvent: (e: StreamEvent) => void,
  signal?: AbortSignal,
): Promise<void> {
  const c = getClient()
  const assistantId = await getRouterAssistantId()

  const run = await c.runs.create(lgThreadId, assistantId, {
    input: null,
    streamMode: [...STREAM_MODES],
    onDisconnect: "continue",
    streamResumable: true,
    multitaskStrategy: "interrupt",
  })
  persistRunMapping(lgThreadId, run.run_id)

  // Surface the NEW run id (resume creates a fresh run that replaces the one the
  // snapshot restored). Without this the UI's runId state stays pinned to the
  // old run, so Stop would cancel a run that no longer exists (404) and leave
  // the live one running. The synthetic start drives setRunId to the new id.
  onEvent({
    event: "start",
    data: { thread_id: lgThreadId, run_id: run.run_id },
  } as StreamEvent)

  await pumpStream(
    c.runs.joinStream(lgThreadId, run.run_id, { signal, streamMode: [...STREAM_MODES] }),
    lgThreadId,
    onEvent,
    signal,
  )
}
