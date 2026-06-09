// Mirror of solidcue/app/utils/normalize.py and solidcue/providers/config.py
// so the wizard derives keys and base-url rules exactly like the CLI/backend.

export function normalizeKey(value: string): string {
  return value
    .trim()
    .toLowerCase()
    .replace(/[\s-]+/g, "_")
    .replace(/[^a-z0-9_]/g, "")
    .replace(/_+/g, "_")
    .replace(/^_+|_+$/g, "")
}

export type ProviderType = "openai_compatible" | "anthropic" | "openrouter"

export const PROVIDER_META: Record<
  ProviderType,
  { label: string; needsBaseUrl: boolean; defaultBaseUrl: string | null }
> = {
  openai_compatible: {
    label: "OpenAI Compatible",
    needsBaseUrl: true,
    defaultBaseUrl: null,
  },
  anthropic: {
    label: "Anthropic (Claude)",
    needsBaseUrl: false,
    defaultBaseUrl: "https://api.anthropic.com",
  },
  openrouter: {
    label: "OpenRouter",
    needsBaseUrl: false,
    defaultBaseUrl: "https://openrouter.ai/api/v1",
  },
}

export interface RoleForm {
  provider_type: ProviderType
  base_url: string
  api_key: string
  model: string
  temperature: string // kept as string for the input; parsed on submit
}

export function emptyRole(temperature = ""): RoleForm {
  return {
    provider_type: "openai_compatible",
    base_url: "",
    api_key: "",
    model: "",
    temperature,
  }
}

// Resolve the base_url to send: user-entered when the provider needs one,
// otherwise the provider's default (matches the CLI).
export function resolveBaseUrl(role: RoleForm): string | null {
  const meta = PROVIDER_META[role.provider_type]
  if (meta.needsBaseUrl) return role.base_url.trim() || null
  return meta.defaultBaseUrl
}
