// Types mirroring the solidcue API schemas (see solidcue/agents/configs/schema.py,
// solidcue/tools/schema.py, solidcue/user/schema.py, solidcue/api/schemas.py).

export interface ProviderConfig {
  type: string
  base_url: string | null
  api_key_env: string
  model: string
  temperature: number | null
}

export interface RuntimeProviderConfig {
  type: "openai_compatible" | "anthropic" | "openrouter"
  base_url: string | null
  api_key: string
  model: string
  temperature: number | null
}

export interface RouterProviderConfig {
  type: "openai_compatible" | "anthropic" | "openrouter"
  base_url: string | null
  api_key_env: string
  model: string
  temperature: number | null
}

export interface AgentConfig {
  agent_id: string
  agent_key: string
  name: string
  description: string
  provider: ProviderConfig
  lite_provider: ProviderConfig | null
  reviewer_provider: ProviderConfig | null
  writer_provider: ProviderConfig | null
  tools: string[]
  allowed_tasks: string[]
  style: Record<string, unknown>
  constraints: Record<string, unknown>
  validation_policy: string | null
}

export interface MCPAuthConfig {
  type: "none" | "api_key" | "bearer" | "oauth"
  token_env: string | null
  location: "header" | "query"
  header_name: string
  prefix: string
  param_name: string
  oauth_provider: string | null
  scopes: string[]
}

export interface MCPServerConfig {
  server_key: string
  name: string
  description: string
  transport: "streamable_http"
  url: string
  auth: MCPAuthConfig
  enabled: boolean
}

export interface MCPToolConfig {
  server_key: string
  tool_name: string
  input_schema: Record<string, unknown> | null
}

export interface APIToolConfig {
  base_url: string
  method: "GET" | "POST"
  headers: Record<string, string>
  auth: MCPAuthConfig
}

export type ToolType = "mcp" | "rag" | "api"
export type ApprovalMode = "never" | "always" | "conditional"
export type ApprovalRisk = "low" | "medium" | "high"

export interface ToolConfig {
  tool_key: string
  name: string
  description: string
  type: ToolType
  enabled: boolean
  approval_mode: ApprovalMode
  approval_risk: ApprovalRisk
  approval_prompt: string | null
  mcp: MCPToolConfig | null
  api: APIToolConfig | null
}

export interface UserProfileConfig {
  location: string | null
  timezone: string | null
  display_name: string | null
  personality: string | null
  router_provider: RouterProviderConfig | null
  preferences: Record<string, unknown>
}

export interface UpdateProfileRequest {
  location: string | null
  timezone: string | null
  display_name: string | null
  personality: string | null
  router_provider: {
    type: "openai_compatible" | "anthropic" | "openrouter"
    base_url: string | null
    model: string
    temperature: number | null
  } | null
  preferences: Record<string, unknown>
  router_api_key?: string | null
}

export interface DiscoveredTool {
  name?: string
  title?: string
  description?: string
  input_schema?: Record<string, unknown>
}

export interface ThreadSummary {
  conversation_id?: string
  thread_id: string
  agent_key: string | null
  step_count: number
}

export interface LiveStateResponse {
  thread_id: string | null
  state: Record<string, unknown>
}

export interface ConversationThreadResponse {
  conversation_id: string
  thread_id: string | null
}

export interface ConversationMetadataResponse {
  conversation_id: string
  agent_key: string | null
  worked_seconds: number
  last_thread_id: string | null
  last_run_id: string | null
  last_run_status: "idle" | "running" | "interrupted" | "completed" | "error" | "cancelled" | "disconnected" | null
  created_at: string | null
  updated_at: string | null
}

export interface RunStatusResponse {
  thread_id: string
  run_id: string | null
  agent_key: string | null
  status: "idle" | "running" | "interrupted" | "completed" | "error" | "cancelled" | "disconnected"
  error: string | null
  updated_at: string | null
}

// --- run / streaming ---

export interface InterruptPreviewSection {
  label?: string
  content?: string
}

export interface InterruptPayload {
  mode?: string
  prompt?: string
  preview?: {
    title?: string
    summary?: string
    sections?: InterruptPreviewSection[]
  }
  options?: string[]
}

export type StreamEvent =
  | {
      event: "start";
      data: {
        thread_id: string;
        run_id?: string;
        conversation_id?: string;
        agent_key?: string;
      };
    }
  | { event: "node"; data: { node: string; phase: string | null; tokens?: { input: number; output: number; total: number } } }
  | { event: "message_start"; data: { thread_id: string; phase: string | null } }
  | { event: "message_delta"; data: { thread_id: string; delta: string } }
  | {
      event: "handoff";
      data: {
        thread_id: string;
        conversation_id?: string;
        target_agent_key: string;
        agent_thread_id: string;
      };
    }
  | {
      event: "plan";
      data: {
        thread_id: string;
        conversation_id?: string;
        intro: string;
        route_reason: string;
        step_count: number;
        steps: { agent_key: string; sub_task: string; step_index: number }[];
      };
    }
  | {
      event: "subagent";
      data: {
        thread_id: string;
        conversation_id?: string;
        agent_key: string;
        agent_thread_id: string;
        sub_task: string;
        step_index: number;
        step_count: number;
        status: "running" | "completed" | "failed" | "interrupted";
        output?: string;
      };
    }
  | {
      event: "subagent_delta";
      data: {
        thread_id: string;
        agent_key: string;
        step_index: number;
        delta: string;
      };
    }
  | { event: "interrupt"; data: { thread_id: string; interrupt: InterruptPayload } }
  | {
      event: "completed";
      data: {
        thread_id: string;
        output: string;
        phase: string | null;
        conversation_id?: string;
      };
    }
  | { event: "cancelled"; data: { thread_id: string; run_id: string } }
  | { event: "error"; data: { message: string } }
