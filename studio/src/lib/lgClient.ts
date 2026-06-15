/**
 * LangGraph Server client for Wave 1 (chat/clarify intents).
 *
 * Wraps @langchain/langgraph-sdk and adapts its stream events into the
 * existing StreamEvent protocol so handleEvent() in SessionsPage needs no
 * changes.  The two hard requirements from Phase 0 are both enforced here:
 *   1. streamResumable: true on every run creation.
 *   2. chunk.id persisted to sessionStorage so it survives a real page refresh.
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
    // First try sessionStorage mapping.
    let lgThreadId = loadThreadMapping(conversationId)
    if (!lgThreadId) {
      // Fall back to server-side metadata search.
      const results = await c.threads.search({ metadata: { conversation_id: conversationId }, limit: 1 })
      lgThreadId = results[0]?.thread_id ?? null
    }
    // If the sidebar row came from a thread without SolidCue metadata,
    // conversationId is already the LangGraph thread id.
    lgThreadId = lgThreadId ?? conversationId
    if (!lgThreadId) return false
    await c.threads.delete(lgThreadId)
    sessionStorage.removeItem(THREAD_KEY + conversationId)
    clearPersistedRun(lgThreadId)
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
// Run-state persistence (survives real page refresh)
// ---------------------------------------------------------------------------

const RUN_KEY = "lg_run:"        // lg_run:<lgThreadId>       → {runId, lastEventId}
const THREAD_KEY = "lg_thread:"  // lg_thread:<conversationId> → lgThreadId

export function persistThreadMapping(conversationId: string, lgThreadId: string): void {
  sessionStorage.setItem(THREAD_KEY + conversationId, lgThreadId)
}

export function loadThreadMapping(conversationId: string): string | null {
  return sessionStorage.getItem(THREAD_KEY + conversationId)
}

interface PersistedRun {
  runId: string
  lastEventId: string | null
}

function persistRunState(lgThreadId: string, runId: string, lastEventId: string | null): void {
  sessionStorage.setItem(RUN_KEY + lgThreadId, JSON.stringify({ runId, lastEventId }))
}

export function loadPersistedRun(lgThreadId: string): PersistedRun | null {
  const raw = sessionStorage.getItem(RUN_KEY + lgThreadId)
  if (!raw) return null
  try {
    return JSON.parse(raw) as PersistedRun
  } catch {
    return null
  }
}

export function clearPersistedRun(lgThreadId: string): void {
  sessionStorage.removeItem(RUN_KEY + lgThreadId)
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

/**
 * Start a new run on an existing LangGraph thread and stream it.
 * Persists last_event_id to sessionStorage after every chunk so a page refresh
 * can rejoin via joinLangGraph() without losing gap events.
 */
export async function streamLangGraph(
  lgThreadId: string,
  userInput: string,
  onEvent: (e: StreamEvent) => void,
  signal?: AbortSignal,
): Promise<void> {
  const c = getClient()
  const assistantId = await getRouterAssistantId()
  let runId: string | null = null
  let completed = false

  for await (const chunk of c.runs.stream(lgThreadId, assistantId, {
    input: { user_input: userInput },
    streamMode: [...STREAM_MODES],
    streamResumable: true,
    onDisconnect: "continue",
    signal,
  })) {
    if (signal?.aborted) break
    const lgChunk = chunk as LgChunk

    if (lgChunk.event === "metadata") {
      runId = (asRecord(lgChunk.data).run_id as string) ?? null
    }
    if (lgChunk.id != null && runId) {
      persistRunState(lgThreadId, runId, lgChunk.id)
    }

    const mapped = mapChunk(lgChunk, lgThreadId)
    if (mapped) {
      if (mapped.event === "completed") {
        completed = true
        onEvent(mapped)
        break
      }
      onEvent(mapped)
    }
  }

  if (completed) clearPersistedRun(lgThreadId)
}

/**
 * Re-attach to a run that was interrupted by a page refresh.
 * Passes last_event_id so the server replays only events after the last
 * received one — zero gap-event loss guaranteed by Phase 0 proof.
 */
export async function joinLangGraph(
  lgThreadId: string,
  runId: string,
  lastEventId: string | null,
  onEvent: (e: StreamEvent) => void,
  signal?: AbortSignal,
): Promise<void> {
  const c = getClient()
  let completed = false

  // Synthetic start event so handleEvent() initialises UI state correctly
  onEvent({ event: "start", data: { thread_id: lgThreadId, run_id: runId } } as StreamEvent)

  for await (const chunk of c.runs.joinStream(lgThreadId, runId, {
    streamMode: [...STREAM_MODES],
    lastEventId: lastEventId ?? undefined,
    signal,
  })) {
    if (signal?.aborted) break
    const lgChunk = chunk as LgChunk

    if (lgChunk.id != null) {
      persistRunState(lgThreadId, runId, lgChunk.id)
    }

    const mapped = mapChunk(lgChunk, lgThreadId)
    if (mapped) {
      if (mapped.event === "completed") {
        completed = true
        onEvent(mapped)
        break
      }
      onEvent(mapped)
    }
  }

  if (completed) clearPersistedRun(lgThreadId)
}
