// Thin typed client over the solidcue API. In dev, Vite proxies /api -> :8000
// (see vite.config.ts). Override with VITE_API_BASE_URL for non-proxied setups.

import type {
  AgentConfig,
  ConversationMetadataResponse,
  ConversationThreadResponse,
  DiscoveredTool,
  LiveStateResponse,
  MCPServerConfig,
  RunStatusResponse,
  StreamEvent,
  ThreadSummary,
  ToolConfig,
  UserProfileConfig,
} from "./types"

const BASE = import.meta.env.VITE_API_BASE_URL ?? "/api"

export class ApiError extends Error {
  status: number
  constructor(status: number, message: string) {
    super(message)
    this.status = status
    this.name = "ApiError"
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${BASE}${path}`, {
    headers: { "Content-Type": "application/json", ...(init?.headers ?? {}) },
    ...init,
  })
  if (!res.ok) {
    let detail = res.statusText
    try {
      const body = await res.json()
      if (body?.detail) detail = typeof body.detail === "string" ? body.detail : JSON.stringify(body.detail)
    } catch {
      /* ignore */
    }
    throw new ApiError(res.status, detail)
  }
  if (res.status === 204) return undefined as T
  return res.json() as Promise<T>
}

export const api = {
  health: () => request<{ status: string }>("/health"),

  // agents
  listAgents: () => request<AgentConfig[]>("/agents"),
  createAgent: (body: unknown) =>
    request<AgentConfig>("/agents", { method: "POST", body: JSON.stringify(body) }),
  updateAgent: (key: string, body: unknown) =>
    request<AgentConfig>(`/agents/${key}`, { method: "PUT", body: JSON.stringify(body) }),
  deleteAgent: (key: string) => request<void>(`/agents/${key}`, { method: "DELETE" }),
  newThread: () => request<{ thread_id: string }>("/agents/threads", { method: "POST" }),

  // tools
  listTools: () => request<ToolConfig[]>("/tools"),
  createMcpTool: (body: unknown) =>
    request<ToolConfig>("/tools/mcp", { method: "POST", body: JSON.stringify(body) }),
  createApiTool: (body: unknown) =>
    request<ToolConfig>("/tools/api", { method: "POST", body: JSON.stringify(body) }),
  createRagTool: (body: unknown) =>
    request<ToolConfig>("/tools/rag", { method: "POST", body: JSON.stringify(body) }),
  updateTool: (key: string, body: ToolConfig) =>
    request<ToolConfig>(`/tools/${key}`, { method: "PUT", body: JSON.stringify(body) }),
  setToolEnabled: (key: string, enabled: boolean) =>
    request<ToolConfig>(`/tools/${key}/enabled`, {
      method: "PATCH",
      body: JSON.stringify({ enabled }),
    }),
  discoveredTools: (serverKey: string) =>
    request<DiscoveredTool[]>(`/tools/mcp-servers/${serverKey}/discovered`),
  toolMcpServers: () => request<MCPServerConfig[]>("/tools/mcp-servers"),

  // threads / state
  listThreads: () => request<ThreadSummary[]>(`/state/threads`),
  deleteThread: (threadId: string) => request<void>(`/state/threads/${threadId}`, { method: "DELETE" }),
  deleteConversation: (conversationId: string) => request<void>(`/state/conversations/${conversationId}`, { method: "DELETE" }),
  liveState: (threadId: string, keys: string[]) =>
    request<LiveStateResponse>(
      `/state/live/${threadId}?${keys.map((k) => `key=${k}`).join("&")}`,
    ),
  runStatus: (threadId: string) => request<RunStatusResponse>(`/state/runs/${threadId}`),
  getInterrupt: (threadId: string) => request<{ interrupt: import("./types").InterruptPayload | null }>(`/state/interrupt/${threadId}`),
  isResumable: (threadId: string) => request<{ resumable: boolean; next_nodes: string[] }>(`/state/resumable/${threadId}`),
  conversationLatestThread: (conversationId: string) =>
    request<ConversationThreadResponse>(`/state/conversations/${conversationId}/latest-thread`),
  conversationMetadata: (conversationId: string) =>
    request<ConversationMetadataResponse>(`/state/conversations/${conversationId}/metadata`),
  conversationLiveState: (conversationId: string, keys: string[]) =>
    request<LiveStateResponse>(
      `/state/conversations/${conversationId}/live?${keys.map((k) => `key=${k}`).join("&")}`,
    ),
  conversationRunStatus: (conversationId: string) =>
    request<RunStatusResponse>(`/state/conversations/${conversationId}/runs`),
  conversationInterrupt: (conversationId: string) =>
    request<{ interrupt: import("./types").InterruptPayload | null }>(`/state/conversations/${conversationId}/interrupt`),
  conversationResumable: (conversationId: string) =>
    request<{ resumable: boolean; next_nodes: string[]; thread_id: string | null }>(`/state/conversations/${conversationId}/resumable`),
  cancelRun: (agentKey: string, runId: string) =>
    request<{ run_id: string; cancelled: boolean }>(`/agents/${agentKey}/runs/${runId}/cancel`, { method: "POST" }),

  // mcp servers
  listMcpServers: () => request<MCPServerConfig[]>("/mcp/servers"),
  createMcpServer: (body: MCPServerConfig) =>
    request<MCPServerConfig>("/mcp/servers", { method: "POST", body: JSON.stringify(body) }),

  // profile
  getProfile: () => request<UserProfileConfig>("/profile"),
  updateProfile: (body: UserProfileConfig) =>
    request<UserProfileConfig>("/profile", { method: "PUT", body: JSON.stringify(body) }),
}

// Stream a run/resume over SSE-via-POST. Calls onEvent for each parsed frame.
export async function streamAgent(
  agentKey: string,
  body: { thread_id?: string; conversation_id?: string; user_input?: string; resume_value?: string },
  onEvent: (e: StreamEvent) => void,
  signal?: AbortSignal,
): Promise<void> {
  const res = await fetch(`${BASE}/agents/${agentKey}/stream`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
    signal,
  })
  if (!res.ok || !res.body) {
    throw new ApiError(res.status, res.statusText)
  }

  const reader = res.body.getReader()
  const decoder = new TextDecoder()
  let buffer = ""

  while (true) {
    const { value, done } = await reader.read()
    if (done) break
    buffer += decoder.decode(value, { stream: true })

    let sep: number
    while ((sep = buffer.indexOf("\n\n")) !== -1) {
      const frame = buffer.slice(0, sep)
      buffer = buffer.slice(sep + 2)
      const parsed = parseFrame(frame)
      if (parsed) onEvent(parsed)
    }
  }
}

function parseFrame(frame: string): StreamEvent | null {
  let event = "message"
  const dataLines: string[] = []
  for (const line of frame.split("\n")) {
    if (line.startsWith("event:")) event = line.slice(6).trim()
    else if (line.startsWith("data:")) dataLines.push(line.slice(5).trim())
  }
  if (dataLines.length === 0) return null
  try {
    return { event, data: JSON.parse(dataLines.join("\n")) } as StreamEvent
  } catch {
    return null
  }
}
