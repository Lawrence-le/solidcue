import type { MCPAuthConfig } from "@/lib/types"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select"

export const ENV_VAR_RE = /^[A-Z_][A-Z0-9_]*$/

export function emptyAuth(): MCPAuthConfig {
  return {
    type: "none",
    token_env: null,
    location: "header",
    header_name: "Authorization",
    prefix: "Bearer",
    param_name: "api_key",
    oauth_provider: null,
    scopes: [],
  }
}

// Returns true when the auth config is complete enough to submit.
export function authValid(auth: MCPAuthConfig): boolean {
  if (auth.type === "none") return true
  return !!auth.token_env && ENV_VAR_RE.test(auth.token_env)
}

export function AuthBuilder({
  value,
  onChange,
}: {
  value: MCPAuthConfig
  onChange: (next: MCPAuthConfig) => void
}) {
  const set = (patch: Partial<MCPAuthConfig>) => onChange({ ...value, ...patch })
  const envBad = value.type !== "none" && !!value.token_env && !ENV_VAR_RE.test(value.token_env)

  return (
    <div className="space-y-4">
      <div className="space-y-1.5">
        <Label>Auth type</Label>
        <Select value={value.type} onValueChange={(v) => set({ type: v as MCPAuthConfig["type"] })}>
          <SelectTrigger className="w-full">
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value="none">None</SelectItem>
            <SelectItem value="api_key">API key</SelectItem>
            <SelectItem value="bearer">Bearer</SelectItem>
            <SelectItem value="oauth">OAuth</SelectItem>
          </SelectContent>
        </Select>
      </div>

      {value.type !== "none" && (
        <>
          <div className="space-y-1.5">
            <Label>Token env var name</Label>
            <Input
              value={value.token_env ?? ""}
              onChange={(e) => set({ token_env: e.target.value || null })}
              placeholder="SERPAPI_API_KEY"
              className={envBad ? "border-destructive" : ""}
            />
            <p className="text-xs text-muted-foreground">
              The env var name, not the raw key. Uppercase letters, digits, underscores.
            </p>
            {envBad && <p className="text-xs text-destructive">Invalid env var name.</p>}
          </div>

          <div className="space-y-1.5">
            <Label>Token location</Label>
            <Select
              value={value.location}
              onValueChange={(v) => set({ location: v as MCPAuthConfig["location"] })}
            >
              <SelectTrigger className="w-full">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="header">Header</SelectItem>
                <SelectItem value="query">Query param</SelectItem>
              </SelectContent>
            </Select>
          </div>

          {value.location === "header" ? (
            <div className="grid grid-cols-2 gap-4">
              <div className="space-y-1.5">
                <Label>Header name</Label>
                <Input value={value.header_name} onChange={(e) => set({ header_name: e.target.value })} />
              </div>
              <div className="space-y-1.5">
                <Label>Prefix</Label>
                <Input value={value.prefix} onChange={(e) => set({ prefix: e.target.value })} />
              </div>
            </div>
          ) : (
            <div className="space-y-1.5">
              <Label>Query param name</Label>
              <Input value={value.param_name} onChange={(e) => set({ param_name: e.target.value })} />
            </div>
          )}
        </>
      )}
    </div>
  )
}
