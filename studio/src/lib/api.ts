// Thin typed client over the solidcue API. In dev, Vite proxies /api -> :8000
// (see vite.config.ts). Override with VITE_API_BASE_URL for non-proxied setups.

import type {
  AgentConfig,
  DiscoveredTool,
  LiveStateResponse,
  MCPServerConfig,
  RuntimeProviderConfig,
  RunStatusResponse,
  ThreadSummary,
  ToolConfig,
  UpdateProfileRequest,
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

  // conversation state
  conversationSnapshot: (conversationId: string) =>
    request<LiveStateResponse>(`/state/conversations/${conversationId}/snapshot`),
  conversationRunStatus: (conversationId: string) =>
    request<RunStatusResponse>(`/state/conversations/${conversationId}/runs`),
  conversationInterrupt: (conversationId: string) =>
    request<{ interrupt: import("./types").InterruptPayload | null }>(`/state/conversations/${conversationId}/interrupt`),
  conversationResumable: (conversationId: string) =>
    request<{ resumable: boolean; next_nodes: string[]; thread_id: string | null }>(`/state/conversations/${conversationId}/resumable`),

  // mcp servers
  listMcpServers: () => request<MCPServerConfig[]>("/mcp/servers"),
  createMcpServer: (body: MCPServerConfig) =>
    request<MCPServerConfig>("/mcp/servers", { method: "POST", body: JSON.stringify(body) }),

  // profile
  getProfile: () => request<UserProfileConfig>("/profile"),
  updateProfile: (body: UpdateProfileRequest) =>
    request<UserProfileConfig>("/profile", { method: "PUT", body: JSON.stringify(body) }),
}
